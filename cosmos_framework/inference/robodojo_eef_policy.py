"""Normalized EEF20 inference; outputs are decoded with one fixed chunk-start anchor."""

import json
import os
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transforms import (
    build_sequence_plan_from_mode,
    reflection_pad_to_target,
)

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAPTION_SUFFIX = (
    " The top centre shows the high camera; the bottom left and right show the left and right wrist cameras."
)


def inference_mode(service):
    """Use the mode frozen with this checkpoint, including older WAM configs."""
    node = service.experiment_config
    for key in ("dataloader_train", "dataloader", "datasets", "robodojo", "dataset"):
        node = node[key] if isinstance(node, Mapping) else getattr(node, key)
    mode = node.get("mode", "wam") if isinstance(node, Mapping) else getattr(node, "mode", "wam")
    expected = os.environ.get("ROBODOJO_EXPECT_MODE")
    if mode not in ("wam", "policy") or (expected is not None and expected != mode):
        raise ValueError(f"Checkpoint task mode {mode!r} disagrees with requested mode {expected!r}")
    return mode


def inference_geometry(service):
    node = service.experiment_config
    for key in ("dataloader_train", "dataloader", "datasets", "robodojo", "dataset"):
        node = node[key] if isinstance(node, Mapping) else getattr(node, key)

    def value(key, default):
        return node.get(key, default) if isinstance(node, Mapping) else getattr(node, key, default)

    return dict(canvas_width=int(value("canvas_width", 960)), canvas_height=int(value("canvas_height", 736)))


def prepare_observation(observation, state, mode="wam", canvas_width=960, canvas_height=736):
    if mode not in ("wam", "policy"):
        raise ValueError(f"Unsupported RoboDojo inference mode: {mode!r}")
    views = []
    for camera in CAMERAS:
        rgb = np.asarray(observation["vision"][camera]["color"])
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError(f"Expected native 640x480 RGB from {camera}, got {rgb.shape}/{rgb.dtype}")
        views.append(np.asarray(Image.fromarray(rgb).resize((480, 360), Image.Resampling.BILINEAR)))
    canvas = np.zeros((720, 960, 3), dtype=np.uint8)
    canvas[:360, 240:720] = views[0]
    canvas[360:, :480] = views[1]
    canvas[360:, 480:] = views[2]
    video_frames = 1 if mode == "policy" else 17
    video = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(1).repeat(1, video_frames, 1, 1)
    padded = reflection_pad_to_target({"video": video}, ["video"], True, canvas_width, canvas_height)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 20D current state, got {state.shape}")
    actions = torch.zeros((17, 64), dtype=torch.float32)
    actions[0, :20] = torch.from_numpy(state.copy())
    caption = observation.get("instruction", observation.get("instructions"))
    if isinstance(caption, (list, tuple)) and len(caption) == 1:
        caption = caption[0]
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("RoboDojo instruction must be a nonempty string")
    data = {
        **padded,
        "action": actions,
        "ai_caption": caption + CAPTION_SUFFIX,
        "viewpoint": "concat_view",
        "conditioning_fps": torch.tensor(25),
        "mode": mode,
        "idle_frames": torch.tensor(0),
    }
    if mode == "policy":
        # Match the training prompt before its RGB window is compacted to T=1.
        data.update(video_num_frames=17, idle_frames_total=17)
    prompt = ActionPromptJsonFormatter(caption_key="ai_caption")(data)["ai_caption"]
    if isinstance(prompt, dict):
        prompt = json.dumps(prompt)
    plan = build_sequence_plan_from_mode(
        mode=mode,
        video_length=video_frames,
        action_length=17,
        has_text=True,
        num_history_actions=1 if mode == "policy" else 0,
    )
    if list(plan.condition_frame_indexes_action) != [0]:
        raise ValueError("Current EEF state must condition the first action slot")
    return {**padded, "action": actions, "prompt": prompt, "plan": plan, "state": state}


def predict_normalized_actions(service, observations, states):
    if not observations or len(observations) != len(states):
        raise ValueError("Observation and state batch sizes must agree")
    mode = inference_mode(service)
    geometry = inference_geometry(service)
    items = [prepare_observation(obs, state, mode=mode, **geometry) for obs, state in zip(observations, states)]
    if not getattr(service, "_robodojo_input_audited", False):
        print(
            json.dumps(
                {
                    "robodojo_inference_mode": mode,
                    "video_frames": items[0]["video"].shape[1],
                    "canvas_wh": [items[0]["video"].shape[-1], items[0]["video"].shape[-2]],
                    "action_rows": 17,
                    "condition_action": list(items[0]["plan"].condition_frame_indexes_action),
                    "condition_vision": list(items[0]["plan"].condition_frame_indexes_vision),
                    "action_start_frame_offset": items[0]["plan"].action_start_frame_offset,
                }
            ),
            flush=True,
        )
        service._robodojo_input_audited = True
    count = len(items)
    batch = {
        service.model.input_video_key: [[item["video"]] for item in items],
        **make_batched_action_processing_fields(
            ActionProcessingRecord(raw_action_dim=20, action_normalizer=None), batch_size=count
        ),
        "action": [[item["action"]] for item in items],
        "mode": [mode] * count,
        "ai_caption": [item["prompt"] for item in items],
        "prompt": [item["prompt"] for item in items],
        "conditioning_fps": [torch.tensor(25) for _ in items],
        "image_size": torch.stack([item["image_size"] for item in items]).cuda(),
        "domain_id": [torch.tensor(17) for _ in items],
        "sequence_plan": [item["plan"] for item in items],
    }
    with service._lock, torch.inference_mode():
        samples = service.model.generate_samples_from_batch(
            batch,
            guidance=service.cfg.guidance,
            seed=[service.cfg.seed] * count,
            num_steps=service.cfg.num_steps,
            has_negative_prompt=False,
        )
    chunks = []
    for item, generated in zip(items, samples["action"], strict=True):
        action = generated.float().squeeze(0).detach().cpu().numpy()
        if action.ndim != 2 or action.shape[0] != 17 or action.shape[1] < 20:
            raise ValueError(f"Expected state plus 16 future actions, got {action.shape}")
        action = action[:, :20]
        if not np.isfinite(action).all() or not np.allclose(action[0], item["state"], atol=0.025, rtol=0):
            raise ValueError("Nonfinite actions or initial state condition was not preserved")
        chunks.append(action[1:].copy())
    return chunks
