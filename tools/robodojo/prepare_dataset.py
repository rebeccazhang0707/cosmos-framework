"""Prepare14D joint-action RoboDojo NPZ episodes without re-encoding videos.

Run from the repository root:
    python tools/robodojo/prepare_dataset.py --dataset-root /path/to/lerobot \
        --output-dir /path/to/prepared [--plan-only]
The versioned split file preserves the exact35-task train/held-out partition.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "examples/robodojo/split.json"


def assign_splits(episodes, split):
    specs = {s["task_index"]: s for s in split["tasks"]}
    if len(specs) != len(split["tasks"]) or set(specs) != {e["task_index"] for e in episodes}:
        raise ValueError("Split tasks must match dataset tasks exactly")
    if len({e["episode_index"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs")
    result = []
    for index, spec in sorted(specs.items()):
        group = [e for e in episodes if e["task_index"] == index]
        ids = {e["episode_index"] for e in group}
        heldout = set(spec["validation_episodes"])
        if len(group) != spec["episodes"] or len(heldout) != len(spec["validation_episodes"]) or not heldout <= ids:
            raise ValueError(f"Invalid episode split for task{index}")
        if not heldout or len(heldout) == len(group):
            raise ValueError("Every task needs disjoint nonempty training and validation sets")
        result.extend(
            dict(e, task_name=spec["name"], split="val" if e["episode_index"] in heldout else "train")
            for e in sorted(group, key=lambda e: e["episode_index"])
        )
    return result


def load_plan(root, split):
    import pyarrow.parquet as pq

    instructions = {
        r["__index_level_0__"]: r["task_index"] for r in pq.read_table(root / "meta/tasks.parquet").to_pylist()
    }
    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index"]
    columns += [f"videos/{c}/{key}" for c in CAMERAS for key in ("chunk_index", "file_index", "from_timestamp")]
    episodes = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            texts = row.pop("tasks")
            if len(texts) != 1:
                raise ValueError("Expected one task instruction per episode")
            episodes.append(dict(row, task_index=instructions[texts[0]], instruction=texts[0]))
    return assign_splits(episodes, split)


def summary(episodes):
    return dict(
        tasks=len({e["task_index"] for e in episodes}),
        demos=dict(Counter(e["split"] for e in episodes)),
        windows={
            s: sum(max(0, (e["length"] - 17) // 8 + 1) for e in episodes if e["split"] == s) for s in ("train", "val")
        },
    )


def prepare(root, output, split, episodes):
    import pyarrow.parquet as pq

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared output must be empty: {output}")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 25:
        raise ValueError("The RoboDojo recipe requires25 Hz data")
    if any(info["features"][key]["shape"] != [14] for key in ("action", "observation.state")):
        raise ValueError("This recipe expects14D joint labels; EEF labels require a separate adapter")
    output.mkdir(parents=True, exist_ok=True)
    videos = {
        root
        / info["video_path"].format(
            video_key=c, chunk_index=e[f"videos/{c}/chunk_index"], file_index=e[f"videos/{c}/file_index"]
        )
        for e in episodes
        for c in CAMERAS
    }
    for path in videos:
        if not path.is_file():
            raise FileNotFoundError(path)
    groups = {}
    for e in episodes:
        groups.setdefault((e["data/chunk_index"], e["data/file_index"]), []).append(e)
    training = []
    max_alignment_error = 0.0
    for (chunk, file), group in sorted(groups.items()):
        table = pq.read_table(
            root / info["data_path"].format(chunk_index=chunk, file_index=file),
            columns=["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"],
        ).combine_chunks()
        ids = table["episode_index"].to_numpy()
        bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        blocks = {int(ids[a]): (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)}
        if len(blocks) != len(bounds) - 1:
            raise ValueError("Noncontiguous episode storage")
        for e in group:
            start, end = blocks[e["episode_index"]]
            part = table.slice(start, end - start)
            arrays = {}
            for source, name in (("action", "action"), ("observation.state", "state")):
                column = part[source].chunk(0)
                values = column.values.to_numpy().reshape(-1, 14)
                arrays[name] = values[column.offset : column.offset + len(part)].astype(np.float32, copy=False)
            action, state = arrays["action"], arrays["state"]
            timestamp = part["timestamp"].to_numpy()
            if action.shape != (e["length"], 14) or state.shape != action.shape:
                raise ValueError(f"Invalid action/state shape: episode{e['episode_index']}")
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError("Nonfinite action/state")
            if not np.array_equal(part["frame_index"].to_numpy(), np.arange(e["length"])) or not np.all(
                np.diff(timestamp) > 0
            ):
                raise ValueError("Frame ordering/timestamps differ from episode metadata")
            if set(part["task_index"].to_pylist()) != {e["task_index"]}:
                raise ValueError("Task ID mismatch")
            if not np.allclose(action[:-1], state[1:], atol=1e-5):
                raise ValueError(f"Action does not match next state: episode{e['episode_index']}")
            max_alignment_error = max(max_alignment_error, float(abs(action[:-1] - state[1:]).max()))
            np.savez(output / f"episode_{e['episode_index']:06d}.npz", **arrays, timestamp=timestamp)
            if e["split"] == "train":
                training.append(action)
        print(f"Prepared data file{chunk}/{file}: {len(group)} episodes", flush=True)
    a = np.concatenate(training)
    stats = {
        k: v.tolist()
        for k, v in dict(
            min=a.min(0),
            max=a.max(0),
            mean=a.mean(0),
            std=a.std(0),
            q01=np.quantile(a, 0.01, axis=0),
            q99=np.quantile(a, 0.99, axis=0),
        ).items()
    }
    manifest = dict(
        root=str(root.resolve()),
        fps=25,
        seed=split["seed"],
        action_dim=14,
        domain_id=17,
        action_normalization=None,
        action_contract="action[t] controls the transition to state[t+1]",
        content_size_wh=[960, 720],
        model_canvas_wh=[736, 544],
        view_size_wh=[480, 360],
        resolution="480",
        episodes=episodes,
        training_action_stats=stats,
        source_revision=split["source_revision"],
        preparation=dict(**summary(episodes), max_action_next_state_error=max_alignment_error),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--plan-only", action="store_true", help="Check task/split metadata without writing data")
    args = parser.parse_args()
    split = json.loads(args.split_file.read_text())
    episodes = load_plan(args.dataset_root, split)
    if not args.plan_only:
        prepare(args.dataset_root, args.output_dir, split, episodes)
    print(json.dumps(summary(episodes)), flush=True)


if __name__ == "__main__":
    main()
