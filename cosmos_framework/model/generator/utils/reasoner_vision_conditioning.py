# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Current-observation ViT conditioning on the packed, causal UND stream."""

from types import SimpleNamespace

import torch
from PIL import Image

from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import get_rope_index


def current_observation(clip: torch.Tensor) -> Image.Image:
    """Read only t=0 of uint8 or normalized [C,T,H,W] RGB (never a future label).

    Legacy WAM batches stay uint8 until VAE encoding; unified vision batches
    are normalized earlier. Do not normalize or transfer the future frames.
    """
    if clip.ndim == 5 and clip.shape[0] == 1:
        clip = clip[0]
    if clip.ndim != 4 or clip.shape[0] != 3:
        raise ValueError("Reasoner expects an RGB [3,T,H,W] clip")
    frame = clip[:, 0].detach()
    if frame.dtype == torch.uint8:
        pixels = frame
    elif frame.is_floating_point():
        pixels = ((frame.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    else:
        raise ValueError("Reasoner pixels must be uint8 or normalized floating point")
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())


def pack_reasoner_vision(
    packed: PackedSequence,
    conditions: list[dict[str, torch.Tensor]],
    multimodal_config: SimpleNamespace,
) -> None:
    """Replace image placeholders and their positions without changing sample splits.

    Compute native Edge mRoPE on the *final* causal span, including the packer's
    BOS/EOS/BOV wrappers. GEN retains its existing modality coordinates and mask.
    The usual MoT forward then computes per-layer image+text K/V on UND, visible
    to GEN within that same sample; UND cannot read generated/future tokens.
    """
    if len(conditions) != len(packed.text_caption_lens):
        raise ValueError("One reasoner observation is required per packed WAM sample")
    indexes, features = [], []
    cursor = 0
    for condition, lengths in zip(conditions, packed.text_caption_lens, strict=True):
        if len(lengths) != 1:
            raise ValueError("Reasoner observation conditioning requires one caption per sample")
        length = lengths[0]
        ids = packed.text_ids[cursor : cursor + length].cpu()
        image_indexes = torch.where(ids == multimodal_config.image_token_id)[0]
        feature = condition["features"]
        grid = condition["grid"].cpu()
        expected = int(grid.prod(-1).sum()) // multimodal_config.vision_config.spatial_merge_size**2
        if image_indexes.numel() != expected or feature.shape[0] != expected:
            raise ValueError("Native image grid, placeholder count and ViT features disagree")
        positions, _ = get_rope_index(
            SimpleNamespace(config=multimodal_config), input_ids=ids[None], image_grid_thw=grid
        )
        global_indexes = packed.text_indexes[cursor : cursor + length]
        offset = packed.position_ids[:, global_indexes[0]].clone()
        packed.position_ids[:, global_indexes] = positions[:, 0].to(packed.position_ids) + offset[:, None]
        indexes.append(image_indexes + cursor)
        features.append(feature)
        cursor += length
    if cursor != packed.text_ids.numel():
        raise ValueError("Reasoner observation packing did not consume every causal token")
    packed.reasoner_image_indexes = torch.cat(indexes)
    packed.reasoner_image_features = torch.cat(features)
