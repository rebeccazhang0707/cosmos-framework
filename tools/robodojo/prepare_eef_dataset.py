"""Prepare normalized20D chunk-relative EEF labels aligned to existing joint RGB.

Only training windows fit the two independent state/action normalizers. Stored
action arrays have shape(windows,16,20); state arrays have shape(frames,20).
Raw16D EEF poses and pre-normalization labels are retained for replay/audit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_framework.data.generator.action.utils.robodojo_eef import (
    absolute20,
    denormalize,
    fit_normalizer,
    normalize,
    pose_errors,
    recover16,
    relative20,
)
from tools.robodojo.prepare_dataset import DEFAULT_SPLIT, load_plan


def build(eef_root, joint_prepared, output, split_file):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((joint_prepared / "manifest.json").read_text())
    split = json.loads(split_file.read_text())
    ee_episodes = load_plan(eef_root, split)
    joint = {e["episode_index"]: e for e in original["episodes"]}
    if set(joint) != {e["episode_index"] for e in ee_episodes}:
        raise ValueError("Joint/EEF episode IDs differ")
    groups = {}
    for e in ee_episodes:
        source = joint[e["episode_index"]]
        for key in ("length", "task_index", "instruction", "split"):
            if e[key] != source[key]:
                raise ValueError(f"Episode alignment mismatch:{e['episode_index']}/{key}")
        groups.setdefault((e["data/chunk_index"], e["data/file_index"]), []).append(e)
    state_train, action_train = [], []
    totals = dict(
        frames=0,
        train_windows=0,
        val_windows=0,
        max_gripper_alignment_error=0.0,
        next_state_position_sum=0.0,
        next_state_rotation_sum=0.0,
        next_state_count=0,
    )
    for (chunk, file), episodes in sorted(groups.items()):
        table = pq.read_table(
            eef_root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet",
            columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"],
        ).combine_chunks()
        ids = table["episode_index"].to_numpy()
        bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        blocks = {int(ids[a]): (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)}
        for e in episodes:
            eid = e["episode_index"]
            begin, end = blocks[eid]
            part = table.slice(begin, end - begin)
            values = {}
            for key, name in (("action", "eef_action_raw"), ("observation.state", "eef_state_raw")):
                column = part[key].chunk(0)
                a = column.values.to_numpy().reshape(-1, 16)
                values[name] = a[column.offset : column.offset + len(part)].astype(np.float32, copy=False)
            state, action = values["eef_state_raw"], values["eef_action_raw"]
            timestamp = part["timestamp"].to_numpy()
            if (
                state.shape != (e["length"], 16)
                or action.shape != state.shape
                or not np.isfinite(state).all()
                or not np.isfinite(action).all()
            ):
                raise ValueError(f"Invalid16D EEF values:{eid}")
            with np.load(joint_prepared / f"episode_{eid:06d}.npz") as old:
                if not np.array_equal(timestamp, old["timestamp"]):
                    raise ValueError(f"Timestamp mismatch:{eid}")
                gap = max(
                    float(abs(state[:, [7, 15]] - old["state"][:, [6, 13]]).max()),
                    float(abs(action[:, [7, 15]] - old["action"][:, [6, 13]]).max()),
                )
            if gap > 1e-6:
                raise ValueError(f"EEF/joint gripper sequence mismatch:{eid}/{gap}")
            totals["max_gripper_alignment_error"] = max(totals["max_gripper_alignment_error"], gap)
            if not np.array_equal(part["frame_index"].to_numpy(), np.arange(e["length"])):
                raise ValueError(f"Frame index mismatch:{eid}")
            starts = np.arange(0, len(state) - 16, 8, dtype=np.int64)
            targets = action[starts[:, None] + np.arange(16)]
            raw = relative20(state[starts, None, :], targets).astype(np.float32)
            abs_state = absolute20(state).astype(np.float32)
            np.savez(
                output / f"episode_{eid:06d}.npz",
                **values,
                timestamp=timestamp,
                starts=starts,
                state_abs20=abs_state,
                action_relative_raw=raw,
            )
            if e["split"] == "train":
                state_train.append(abs_state[starts])
                action_train.append(raw)
            totals[e["split"] + "_windows"] += len(starts)
            totals["frames"] += len(state)
            p, r = pose_errors(action[:-1], state[1:])
            totals["next_state_position_sum"] += float(p.sum())
            totals["next_state_rotation_sum"] += float(r.sum())
            totals["next_state_count"] += p.size
        print(f"Encoded EEF file{chunk}/{file}: {len(episodes)} episodes", flush=True)
    state_stats = fit_normalizer(np.concatenate(state_train))
    del state_train
    action_stats = fit_normalizer(np.concatenate(action_train))
    del action_train
    normalizers = dict(
        state=state_stats,
        action=action_stats,
        fit_split="train",
        fit_state_rows=totals["train_windows"],
        fit_action_rows=totals["train_windows"] * 16,
        validation_used=False,
    )
    (output / "normalization.json").write_text(json.dumps(normalizers, indent=2) + "\n")
    audit = dict(
        max_position_error_m=0.0, max_rotation_error_rad=0.0, max_gripper_error=0.0, max_state_roundtrip_error=0.0
    )
    for i, e in enumerate(ee_episodes):
        path = output / f"episode_{e['episode_index']:06d}.npz"
        with np.load(path) as saved:
            a = {k: saved[k].copy() for k in saved.files}
        abs_state = a.pop("state_abs20")
        a["state"] = normalize(abs_state, state_stats)
        a["action"] = normalize(a["action_relative_raw"], action_stats)
        recovered = recover16(a["eef_state_raw"][a["starts"], None, :], denormalize(a["action"], action_stats))
        target = a["eef_action_raw"][a["starts"][:, None] + np.arange(16)]
        p, r = pose_errors(recovered, target)
        audit["max_position_error_m"] = max(audit["max_position_error_m"], float(p.max()))
        audit["max_rotation_error_rad"] = max(audit["max_rotation_error_rad"], float(r.max()))
        audit["max_gripper_error"] = max(
            audit["max_gripper_error"], float(abs(recovered[..., [7, 15]] - target[..., [7, 15]]).max())
        )
        audit["max_state_roundtrip_error"] = max(
            audit["max_state_roundtrip_error"], float(abs(denormalize(a["state"], state_stats) - abs_state).max())
        )
        np.savez(path, **a)
        if i % 500 == 0:
            print("Normalized and roundtrip-checked", i + 1, flush=True)
    if audit["max_position_error_m"] > 1e-6 or audit["max_rotation_error_rad"] > 1e-5:
        raise ValueError(f"EEF codec roundtrip failed:{audit}")
    manifest = {
        k: v for k, v in original.items() if k not in ("episodes", "training_action_stats", "alignment_diagnostics")
    }
    manifest.update(
        episodes=[joint[e["episode_index"]] for e in ee_episodes],
        action_dim=20,
        domain_id=18,
        action_representation="eef_relative_chunk",
        action_normalization="preapplied_q01_q99_no_clip",
        normalization_applied=True,
        normalization_file="normalization.json",
        chunk_length=16,
        sample_stride=8,
        eef_source_root=str(eef_root),
        video_source_root=original["root"],
        action_contract="Per arm:[p_target-p_chunk_start,rot6d(R_target@R_chunk_start.T),absolute_gripper];world axes",
        state_contract="Absolute EEF xyz/rot6d/gripper;separate training-only state normalizer",
        rotation_6d_convention="first_two_columns_concatenated",
        quaternion_order="wxyz",
        audit=dict(**totals, **audit),
        source_revision=split["source_revision"],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected = []
    for spec in split["tasks"]:
        ids = sorted(spec["validation_episodes"])[::2]
        if len(ids) != 5:
            raise ValueError("Expected5 selected held-out episodes per task")
        selected.extend(dict(episode_index=eid, task_index=spec["task_index"], task_name=spec["name"]) for eid in ids)
    (output / "replay_plan.json").write_text(
        json.dumps(
            dict(
                episodes=selected,
                count=len(selected),
                selection="Sorted held-out episode IDs at positions0,2,4,6,8;fixed before IK outcomes",
                task_success_rate=None,
                task_success_unavailable_reason="Original train scene layouts are not released",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(output=str(output), **totals, **audit)), flush=True)


def main():
    # This sequential converter needs no machine-sized Arrow thread pools.
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eef-root", type=Path, required=True)
    p.add_argument("--joint-prepared", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    a = p.parse_args()
    build(a.eef_root, a.joint_prepared, a.output_dir, a.split_file)


if __name__ == "__main__":
    main()
