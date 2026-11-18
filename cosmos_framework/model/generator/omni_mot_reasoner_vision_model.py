# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in Edge WAM training with frozen image+text reasoner prefill (no AR)."""

from pathlib import Path
from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.mot.unified_mot import _load_bundled_vision_tower
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.reasoner_vision_conditioning import (
    current_observation,
    pack_reasoner_vision,
)


class OmniMoTReasonerVisionModel(OmniMoTModel):
    """Frozen SigLIP2 -> native image tokens -> frozen UND -> trainable GEN.

    ViT is a local, frozen preprocessing module outside the sharded net, like
    the VAE. Its released weights must remain available when resuming; training
    checkpoints keep the existing net.* layout and do not duplicate the ViT.
    """

    def __init__(self, config, reasoner_checkpoint_path: str, reasoner_max_pixels: int = 400384):
        bundle = _load_bundled_vision_tower(reasoner_checkpoint_path)
        if bundle is None:
            raise FileNotFoundError(f"Missing Edge vision_encoder/model.safetensors in {reasoner_checkpoint_path}")
        weights, vision_config, top_config = bundle
        super().__init__(config)
        self._reasoner_vision_bundle = (weights, vision_config)
        self.reasoner_max_pixels = reasoner_max_pixels
        self.reasoner_vision_encoder = None
        self.reasoner_multimodal_config = SimpleNamespace(
            image_token_id=top_config["image_token_id"],
            video_token_id=top_config["video_token_id"],
            vision_start_token_id=top_config["vision_start_token_id"],
            vision_config=SimpleNamespace(spatial_merge_size=vision_config["projector_config"]["spatial_merge_size"]),
        )
        # Catch mismatched tokenizer/vision releases before the first training step.
        image_id = self.vlm_tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_id != self.reasoner_multimodal_config.image_token_id:
            raise ValueError("The native Edge tokenizer and vision checkpoint must use the same image token ID")

    def _get_reasoner_vision_encoder(self):
        if self.reasoner_vision_encoder is None:
            from safetensors.torch import load_file
            from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

            from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import (
                NemotronSiglip2VisionEncoder,
            )

            weights_file, config = self._reasoner_vision_bundle
            vision = dict(config["vision_config"])
            for key in ("model_type", "transformers_version", "spatial_merge_size"):
                vision.pop(key, None)
            vcfg = Siglip2VisionConfig(**vision)
            vcfg._attn_implementation = "eager"
            projector = config["projector_config"]
            pcfg = SimpleNamespace(
                spatial_merge_size=projector["spatial_merge_size"],
                input_hidden_size=projector["input_hidden_size"],
                merger_intermedia=projector.get("merger_intermedia", projector.get("merger_intermediate_size")),
                out_hidden_size=projector["out_hidden_size"],
            )
            # Meta construction avoids consuming training RNG or allocating an FP32 ViT.
            with torch.device("meta"):
                encoder = NemotronSiglip2VisionEncoder(vcfg, pcfg)
            state = load_file(str(Path(weights_file)))
            state = {k.removeprefix("model."): v for k, v in state.items()}
            encoder.load_state_dict(state, strict=True, assign=True)
            self.reasoner_vision_encoder = encoder.to(**self.tensor_kwargs).requires_grad_(False).eval()
        return self.reasoner_vision_encoder.eval()

    @torch.no_grad()
    def _load_and_tokenize_text_data(self, data_batch, iteration):
        if "text_token_lengths" in data_batch:
            raise ValueError("Use the stitched-camera, single-caption RoboDojo WAM dataset")
        encoder = self._get_reasoner_vision_encoder()
        captions = data_batch[self.input_caption_key]
        clips = data_batch[self.input_video_key]
        if len(captions) != len(clips):
            raise ValueError("Reasoner captions and observations must have the same batch size")
        tokens, conditions = [], []
        for caption, clip in zip(captions, clips, strict=True):
            # Dataset-side JSON formatting and CFG text dropout have already run.
            if not isinstance(caption, str):
                raise ValueError("Reasoner expects the dataset's serialized JSON caption")
            if isinstance(clip, (list, tuple)):
                if len(clip) != 1:
                    raise ValueError("Only one stitched camera canvas per WAM sample is supported")
                clip = clip[0]
            inputs = self.vlm_processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": current_observation(clip),
                                "max_pixels": self.reasoner_max_pixels,
                            },
                            {"type": "text", "text": caption},
                        ],
                    }
                ],
                add_generation_prompt=True,
            )
            grid = inputs["image_grid_thw"]
            features = encoder.get_image_features(
                inputs["pixel_values"].to(**self.tensor_kwargs), grid.to(self.tensor_kwargs["device"])
            )
            tokens.append(inputs["input_ids"].tolist())
            conditions.append({"grid": grid, "features": features.detach()})
        data_batch["_reasoner_vision"] = conditions
        return tokens

    def _prepare_training_data(self, data_batch, iteration):
        result = super()._prepare_training_data(data_batch, iteration)
        _, plans, clean, *_ = result
        for plan in plans:
            if list(plan.condition_frame_indexes_vision) != [0] or list(plan.condition_frame_indexes_action) != [0]:
                raise ValueError("Reasoner vision training supports current-frame/current-state WAM conditions only")
        clean.reasoner_vision = data_batch.pop("_reasoner_vision")
        return result

    def _pack_input_sequence(self, sequence_plans, input_text_indexes, gen_data_clean, input_timesteps, **kwargs):
        packed = super()._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, input_timesteps, **kwargs
        )
        if gen_data_clean.reasoner_vision is None:
            raise ValueError("Missing current-observation reasoner features")
        pack_reasoner_vision(packed, gen_data_clean.reasoner_vision, self.reasoner_multimodal_config)
        return packed

    def _prepare_inference_data(self, *args, **kwargs):
        raise NotImplementedError(
            "This recipe implements training prefill only. Policy serving must also supply current-image UND tokens; "
            "the text-only WAM inference path cannot evaluate this checkpoint correctly."
        )
