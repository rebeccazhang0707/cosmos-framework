# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.reasoner_vision_conditioning import (
    current_observation,
    pack_reasoner_vision,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_frozen_siglip_encoder_supports_bfloat16_position_interpolation():
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import (
        NemotronSiglip2VisionEncoder,
    )

    config = Siglip2VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_patches=16,
        patch_size=2,
    )
    config._attn_implementation = "eager"
    projector = SimpleNamespace(spatial_merge_size=2, input_hidden_size=8, merger_intermedia=16, out_hidden_size=8)
    encoder = NemotronSiglip2VisionEncoder(config, projector).bfloat16().requires_grad_(False).eval()
    # Resize the learned 4x4 positional grid to a rectangular 4x6 observation.
    features = encoder.get_image_features(torch.randn(24, 12).bfloat16(), torch.tensor([[1, 4, 6]]))
    assert features.shape == (6, 8)
    assert features.dtype == torch.bfloat16
    assert torch.isfinite(features).all()
    assert not features.requires_grad


def _pack():
    plans = [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]) for _ in range(2)]
    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 4, 2, 2, 2) for _ in plans],
        fps_vision=torch.tensor([25.0, 25.0]),
    )
    # Each native image expands into a 2x2 merged grid, surrounded by normal text.
    ids = [[7, 20, 19, 19, 19, 19, 21, 8], [9, 20, 19, 19, 19, 19, 21, 10, 11]]
    packed = pack_input_sequence(
        plans,
        ids,
        clean,
        torch.tensor([0.5, 0.6]),
        special_tokens={"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
    )
    config = SimpleNamespace(
        image_token_id=19,
        video_token_id=18,
        vision_start_token_id=20,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    conditions = [{"grid": torch.tensor([[1, 4, 4]]), "features": torch.full((4, 8), float(i + 1))} for i in range(2)]
    return packed, config, conditions


def test_only_current_frame_enters_reasoner():
    clip = torch.rand(3, 17, 32, 64) * 2 - 1
    reference = np.asarray(current_observation(clip))
    clip[:, 1:] = float("nan")
    np.testing.assert_array_equal(np.asarray(current_observation(clip)), reference)
    clip[:, 0] = -1
    assert not np.array_equal(np.asarray(current_observation(clip)), reference)


def test_native_grid_and_features_preserve_sample_boundaries_and_gen_positions():
    packed, config, conditions = _pack()
    original_positions = packed.position_ids.clone()
    original_splits = packed.split_lens.copy()
    original_samples = packed.sample_lens.copy()
    pack_reasoner_vision(packed, conditions, config)
    assert packed.split_lens == original_splits
    assert packed.sample_lens == original_samples
    assert packed.text_caption_lens[0] != packed.text_caption_lens[1]
    image_indexes = packed.reasoner_image_indexes
    assert image_indexes is not None
    first_grid = packed.position_ids[:, packed.text_indexes[image_indexes[:4]]]
    # Native spatial mRoPE: image rows have shared time but different H/W coordinates.
    assert first_grid[0].unique().numel() == 1
    assert first_grid[1].unique().numel() == first_grid[2].unique().numel() == 2
    is_gen = torch.ones(packed.sequence_length, dtype=torch.bool)
    is_gen[packed.text_indexes] = False
    torch.testing.assert_close(packed.position_ids[:, is_gen], original_positions[:, is_gen])
    embedding = torch.nn.Embedding(64, 8)
    embedding.requires_grad_(False)
    network = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)), hidden_size=8
    )
    encoded, _ = Cosmos3VFMNetwork._encode_text(network, packed)
    torch.testing.assert_close(encoded[packed.text_indexes[image_indexes[:4]]], conditions[0]["features"])
    torch.testing.assert_close(encoded[packed.text_indexes[image_indexes[4:]]], conditions[1]["features"])
    assert not encoded.requires_grad
    assert torch.count_nonzero(encoded[is_gen]) == 0


def test_mismatched_visual_tokens_fail_instead_of_spilling_into_next_sample():
    packed, config, conditions = _pack()
    conditions[0]["features"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="disagree"):
        pack_reasoner_vision(packed, conditions, config)


def test_image_condition_changes_generator_and_backpropagates_only_to_gen():
    """Run a real Edge MoT decoder; substitute CPU SDPA for the CUDA-only kernel."""
    from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq
    from cosmos_framework.model.generator.mot.attention import build_packed_sequence
    from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
        Nemotron3DenseVLTextConfig,
    )
    from cosmos_framework.model.generator.utils.memory import MemoryValue

    torch.manual_seed(42)
    packed, config, conditions = _pack()
    for item in conditions:
        item["features"] = torch.randn(4, 8)
    pack_reasoner_vision(packed, conditions, config)
    embedding = torch.nn.Embedding(64, 8).requires_grad_(False)
    network = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)), hidden_size=8
    )
    layer = MoTDecoderLayer(
        Nemotron3DenseVLTextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            num_hidden_layers=1,
        ),
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
    )
    for name, parameter in layer.named_parameters():
        parameter.requires_grad_("moe_gen" in name)

    def sdpa(q, k, v, mask, *, packed_key_states_normalized, **kwargs):
        # Keep native per-sample UND-causal / GEN-to-(UND+GEN) visibility.
        und_out, gen_out = [], []
        gen_k = k if packed_key_states_normalized is None else packed_key_states_normalized
        for i in range(q["_causal_seq_offsets"].numel() - 1):
            u0, u1 = q["_causal_seq_offsets"][i : i + 2]
            g0, g1 = q["_full_only_seq_offsets"][i : i + 2]

            def attention(query, key, value, causal=False):
                return (
                    torch.nn.functional.scaled_dot_product_attention(
                        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), is_causal=causal
                    )
                    .transpose(0, 1)
                    .flatten(-2)
                )

            und_out.append(attention(get_und_seq(q)[u0:u1], get_und_seq(k)[u0:u1], get_und_seq(v)[u0:u1], True))
            gen_out.append(
                attention(
                    get_gen_seq(q)[g0:g1],
                    torch.cat((get_und_seq(gen_k)[u0:u1], get_gen_seq(gen_k)[g0:g1])),
                    torch.cat((get_und_seq(v)[u0:u1], get_gen_seq(v)[g0:g1])),
                )
            )
        return from_und_gen_splits(torch.cat(und_out), torch.cat(gen_out), q), None

    layer.self_attn.dispatch_attention_fn = sdpa
    gen_indexes = torch.where(~torch.isin(torch.arange(packed.sequence_length), packed.text_indexes))[0]
    gen_input = torch.randn(len(gen_indexes), 8)

    def forward():
        hidden, _ = Cosmos3VFMNetwork._encode_text(network, packed)
        hidden[gen_indexes] = gen_input
        seq, mask, _ = build_packed_sequence(
            "two_way",
            packed_sequence=hidden,
            attn_modes=packed.attn_modes,
            split_lens=packed.split_lens,
            sample_lens=packed.sample_lens,
            packed_und_token_indexes=packed.text_indexes,
            packed_gen_token_indexes=gen_indexes,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            text_caption_lens=packed.text_caption_lens,
        )
        und_len, gen_len = len(get_und_seq(seq)), len(get_gen_seq(seq))
        cos = from_und_gen_splits(torch.ones(und_len, 4), torch.ones(gen_len, 4), seq)
        sin = from_und_gen_splits(torch.zeros(und_len, 4), torch.zeros(gen_len, 4), seq)
        result, _, _ = layer(seq, mask, (cos, sin), memory_value=MemoryValue())
        return get_gen_seq(result)[: len(gen_indexes)]

    baseline = forward()
    conditions[0]["features"] = torch.randn(4, 8)
    pack_reasoner_vision(packed, conditions, config)
    changed = forward()
    sample_gen_len = packed.split_lens[1]
    assert not torch.allclose(changed[:sample_gen_len], baseline[:sample_gen_len])
    torch.testing.assert_close(changed[sample_gen_len:], baseline[sample_gen_len:])
    changed.square().mean().backward()
    assert all(p.grad is None for n, p in layer.named_parameters() if "moe_gen" not in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in layer.named_parameters() if "moe_gen" in n)
