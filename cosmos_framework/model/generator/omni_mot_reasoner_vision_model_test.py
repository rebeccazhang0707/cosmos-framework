"""Inference must preserve training's image conditioning across the sampler boundary."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.omni_mot_reasoner_vision_model import OmniMoTReasonerVisionModel
from cosmos_framework.model.generator.utils.reasoner_vision_conditioning import current_observation

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def model_and_batch():
    model = OmniMoTReasonerVisionModel.__new__(OmniMoTReasonerVisionModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(input_caption_key="ai_caption", input_video_key="video")

    # Substitute only the expensive released processor/encoder. Both training
    # and inference are required to call this exact same method.
    def tokenize(self, batch, iteration):
        image = np.asarray(current_observation(batch["video"][0][0]))
        value = float(image.mean())
        batch["_reasoner_vision"] = [{"features": torch.tensor([[value]]), "grid": torch.tensor([[1, 1, 1]])}]
        return [[17, 19, 21, int(value)]]

    model._load_and_tokenize_text_data = MethodType(tokenize, model)
    batch = {
        "video": [[torch.full((3, 17, 4, 4), 25, dtype=torch.uint8)]],
        "ai_caption": ['{"task":"stack bowls"}'],
        "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0], condition_frame_indexes_action=[0])],
    }
    return model, batch


def parent_preparation(self, batch, seed, has_negative_prompt=False):
    clean = self.get_data_and_condition(batch)
    # The base sampler can release raw vision before token and mask packing.
    del batch["video"]
    tokens, unused_unconditional = self._get_inference_text_tokens(batch, has_negative_prompt)
    assert tokens == unused_unconditional
    assert clean.reasoner_vision is not None
    # Later denoising reuses clean and must retain the visual feature tensors.
    return clean, tokens, seed


def infer(model, batch):
    with (
        patch.object(OmniMoTModel, "get_data_and_condition", return_value=SimpleNamespace(reasoner_vision=None)),
        patch.object(OmniMoTModel, "_prepare_inference_data", parent_preparation),
    ):
        return model._prepare_inference_data(batch, [42])


def test_current_image_features_survive_raw_release_and_future_pixels_are_ignored():
    model, batch = model_and_batch()
    first, tokens, seed = infer(model, batch)
    assert seed == [42]
    assert "video" in batch and "_reasoner_vision" not in batch and "_reasoner_inference_tokens" not in batch
    batch["video"][0][0][:, 1:] = 200
    unchanged, unchanged_tokens, _ = infer(model, batch)
    torch.testing.assert_close(first.reasoner_vision[0]["features"], unchanged.reasoner_vision[0]["features"])
    assert tokens == unchanged_tokens
    batch["video"][0][0][:, 0] = 99
    changed, changed_tokens, _ = infer(model, batch)
    assert not torch.equal(changed.reasoner_vision[0]["features"], first.reasoner_vision[0]["features"])
    assert tokens != changed_tokens


def test_inference_does_not_change_training_attachment_or_leak_request_state_on_failure():
    model, batch = model_and_batch()
    batch["_reasoner_vision"] = [{"features": torch.ones(1, 1)}]
    with patch.object(OmniMoTModel, "get_data_and_condition", return_value=SimpleNamespace(reasoner_vision=None)):
        assert model.get_data_and_condition(batch).reasoner_vision is None
        assert "_reasoner_vision" in batch
    del batch["_reasoner_vision"]
    with patch.object(OmniMoTModel, "_prepare_inference_data", side_effect=RuntimeError("sampler failed")):
        with pytest.raises(RuntimeError, match="sampler failed"):
            model._prepare_inference_data(batch, [42])
    assert "_reasoner_vision" not in batch and "_reasoner_inference_tokens" not in batch


@pytest.mark.parametrize(
    "overrides", [{"guidance": 2}, {"use_batched_cfg": True}, {"has_negative_prompt": True}, {"upsample_task": "i2v"}]
)
def test_rejects_unsupported_cfg_or_prompt_rewriting(overrides):
    model, batch = model_and_batch()
    with pytest.raises(ValueError, match="guidance1"):
        model.generate_samples_from_batch(batch, seed=[42], **overrides)


def test_native_guidance_one_sampling_delegates_to_existing_sampler():
    model, batch = model_and_batch()
    with patch.object(OmniMoTModel, "generate_samples_from_batch", return_value={"action": []}) as sample:
        assert model.generate_samples_from_batch(batch, seed=[42], num_steps=30) == {"action": []}
        sample.assert_called_once_with(batch, guidance=1.0, seed=[42], num_steps=30)
