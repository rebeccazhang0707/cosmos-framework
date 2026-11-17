"""RoboDojo joint/EEF labels, synchronized native-rate clips and fixed camera layout."""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline, reflection_pad_to_target

CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")


class RoboDojoPolicyDataset(Dataset):
    def __init__(
        self,
        prepared_root: str,
        split: str = "train",
        chunk_length: int = 16,
        sample_stride: int = 8,
        mode: str = "wam",
    ):
        if mode not in ("wam", "policy"):
            raise ValueError(f"Unsupported RoboDojo mode: {mode}")
        self.mode = mode
        self.prepared = Path(prepared_root)
        self.manifest = json.loads((self.prepared / "manifest.json").read_text())
        self.action_representation = self.manifest.get("action_representation", "joint_absolute")
        if self.action_representation == "eef_relative_chunk" and (
            chunk_length != self.manifest["chunk_length"] or sample_stride != self.manifest["sample_stride"]
        ):
            raise ValueError("Precomputed EEF labels require their frozen chunk length and stride")
        self.root = Path(self.manifest["root"])
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        self.chunk_length = chunk_length
        self.stride = sample_stride
        self.blocks = []
        self.ends = []
        n = 0
        for e in self.episodes:
            count = max(0, (e["length"] - chunk_length - 1) // sample_stride + 1)
            self.blocks.append((n, count))
            n += count
            self.ends.append(n)
        self._arrays = OrderedDict()
        self._decoders = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def get_shuffle_blocks(self):
        return self.blocks

    def __getitem__(self, index):
        import av

        ep = bisect.bisect_right(self.ends, index)
        e = self.episodes[ep]
        start = (index - self.blocks[ep][0]) * self.stride
        eid = e["episode_index"]
        if eid not in self._arrays:
            with np.load(self.prepared / f"episode_{eid:06d}.npz") as a:
                self._arrays[eid] = {k: a[k].copy() for k in a.files}
            if len(self._arrays) > 8:
                self._arrays.popitem(last=False)
        a = self._arrays[eid]
        self._arrays.move_to_end(eid)
        ts = a["timestamp"][start : start + self.chunk_length + 1]
        if self.mode == "policy":
            ts = ts[:1]  # No future RGB is decoded, encoded, or passed to the action-only model.
        views = []
        for cam in CAMERAS:
            path = str(
                self.root
                / f"videos/{cam}/chunk-{e[f'videos/{cam}/chunk_index']:03d}/file-{e[f'videos/{cam}/file_index']:03d}.mp4"
            )
            if path not in self._decoders:
                container = av.open(path)
                container.streams.video[0].thread_count = 1
                self._decoders[path] = container
                if len(self._decoders) > 6:
                    self._decoders.popitem(last=False)[1].close()
            decoder = self._decoders[path]
            self._decoders.move_to_end(path)
            times = (ts.astype(np.float64) + e[f"videos/{cam}/from_timestamp"]).tolist()
            stream = decoder.streams.video[0]
            decoder.seek(max(0, int((times[0] - 0.5 / 25) / float(stream.time_base))), stream=stream, backward=True)
            frames = []
            for frame in decoder.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base)
                if t < times[len(frames)] - 0.5 / 25:
                    continue
                if abs(t - times[len(frames)]) > 0.5 / 25 + 1e-4:
                    raise ValueError(f"Camera timestamp mismatch: {path}, wanted {times[len(frames)]}, got {t}")
                rgb = frame.reformat(width=480, height=360, format="rgb24").to_ndarray()
                frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
                if len(frames) == len(times):
                    break
            if len(frames) != len(times):
                raise ValueError(f"Incomplete camera window: {path}")
            views.append(torch.stack(frames))
        video = torch.zeros((len(ts), 3, 720, 960), dtype=torch.uint8)
        video[:, :, :360, 240:720] = views[0]
        video[:, :, 360:, :480] = views[1]
        video[:, :, 360:, 480:] = views[2]
        if self.action_representation == "eef_relative_chunk":
            row = start // self.stride
            if int(a["starts"][row]) != start:
                raise ValueError("EEF target window does not match the camera/state anchor")
            action = np.concatenate([a["state"][start : start + 1], a["action"][row]])
        else:
            action = np.concatenate([a["state"][start : start + 1], a["action"][start : start + self.chunk_length]])
        result = dict(
            video=video.permute(1, 0, 2, 3).contiguous(),
            action=torch.from_numpy(action.copy()),
            ai_caption=e["instruction"]
            + " The top centre shows the high camera; the bottom left and right show the left and right wrist cameras.",
            mode=self.mode,
            domain_id=torch.tensor(self.manifest.get("domain_id", 17)),
            viewpoint="concat_view",
            conditioning_fps=torch.tensor(25),
            idle_frames=torch.tensor(0),
            __key__=f"{eid}/{start}",
        )
        if self.mode == "policy":
            # Native Policy keeps history_action clean. Transport the current state
            # through that field so token0 remains a condition, followed by16 targets.
            # This does not add historical observations or a memory mechanism.
            result["history_action"] = result["action"][:1].clone()
            result["action"] = result["action"][1:]
            result["video_num_frames"] = self.chunk_length + 1  # Preserve prompt duration before RGB compaction.
            result["idle_frames_total"] = self.chunk_length + 1  # Match the WAM prompt including its state row.
        return result


class RoboDojoTransform(ActionTransformPipeline):
    def __init__(self, *args, canvas_width=960, canvas_height=736, **kwargs):
        super().__init__(*args, **kwargs)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def _prepare_video(self, data_dict, resolution):
        # The original720 recipe keeps960x720 content intact. Smaller canvases
        # use the same aspect-preserving downscale and reflection padding.
        return reflection_pad_to_target(data_dict, ["video"], True, self.canvas_width, self.canvas_height), None


def get_robodojo_policy_sft_dataset(
    prepared_root: str,
    tokenizer_config: dict,
    max_action_dim: int = 64,
    split: str = "train",
    chunk_length: int = 16,
    canvas_width: int = 960,
    canvas_height: int = 736,
    mode: str = "wam",
    resolution: str = "720",
):
    dataset = RoboDojoPolicyDataset(prepared_root, split, chunk_length, mode=mode)
    transform = RoboDojoTransform(
        tokenizer_config=tokenizer_config,
        max_action_dim=max_action_dim,
        cfg_dropout_rate=0.1,
        format_prompt_as_json=True,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    return ActionIterableShuffleDataset(sft, seed=42) if split == "train" else sft
