"""Select the exact exp1 four-task demos from the frozen EEF20 dataset.

Preserve the reference episode order/splits. Use official ee_lerobot RGB files
and their own episode offsets, with frozen EEF20 normalizers and NPZ labels.
Domain17 pairs the initial embedding with the independent joint control.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

TASKS = {"classify_objects", "insert_tubes", "push_T", "stack_bowls"}


def prepare(source, reference, eef_video_root, output):
    import pyarrow.parquet as pq

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    full = json.loads((source / "manifest.json").read_text())
    ref = json.loads((reference / "manifest.json").read_text())
    if full.get("action_representation") != "eef_relative_chunk" or full["action_dim"] != 20:
        raise ValueError("Expected the frozen normalized EEF20 source")
    if ref.get("action_dim", 14) != 14 or len(ref["episodes"]) != 400:
        raise ValueError("Expected the original400-demo joint reference")
    info = json.loads((eef_video_root / "meta/info.json").read_text())
    if info["features"]["action"]["shape"] != [16] or info["fps"] != 25:
        raise ValueError("Expected the official16D ee_lerobot source metadata at25Hz")
    if eef_video_root.name != "RoboDojo_ee_lerobot_v30_video":
        raise ValueError("EEF control requires the official ee_lerobot RGB directory")
    if Counter(e["task_name"] for e in ref["episodes"]) != Counter(dict.fromkeys(TASKS, 100)):
        raise ValueError("Reference must contain exactly the original four tasks")
    by_id = {e["episode_index"]: e for e in full["episodes"]}
    video_episodes = {}
    for path in sorted((eef_video_root / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.ParquetFile(path).read(use_threads=False).to_pylist():
            video_episodes[row["episode_index"]] = row
    episodes = []
    for old in ref["episodes"]:
        e = dict(by_id[old["episode_index"]])
        if any(old[k] != e.get(k) for k in old):
            raise ValueError(f"Source/reference episode metadata mismatch: {old['episode_index']}")
        video = video_episodes[old["episode_index"]]
        if video["length"] != old["length"] or video["tasks"] != [old["instruction"]]:
            raise ValueError(f"Official EEF episode identity/length mismatch: {old['episode_index']}")
        for k in e:
            if k.startswith("videos/") or k in ("data/chunk_index", "data/file_index"):
                e[k] = video[k]
        episodes.append(e)
    counts = Counter(e["split"] for e in episodes)
    if counts != {"train": 360, "val": 40}:
        raise ValueError(f"Unexpected split: {counts}")
    stats = source / full["normalization_file"]
    normalizers = json.loads(stats.read_text())
    if normalizers["validation_used"] or normalizers["fit_split"] != "train":
        raise ValueError("Source statistics must exclude validation")
    videos = {
        eef_video_root
        / info["video_path"].format(
            video_key=cam, chunk_index=e[f"videos/{cam}/chunk_index"], file_index=e[f"videos/{cam}/file_index"]
        )
        for e in episodes
        for cam in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    }
    for path in videos:
        if not path.is_file():
            raise FileNotFoundError(f"Download official EEF RGB first: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        filename = f"episode_{e['episode_index']:06d}.npz"
        target = (source / filename).resolve(strict=True)
        (output / filename).symlink_to(target)
    shutil.copyfile(stats, output / full["normalization_file"])
    manifest = {k: v for k, v in full.items() if k not in ("episodes", "audit", "preparation")}
    manifest.update(
        episodes=episodes,
        root=str(eef_video_root.resolve()),
        video_source_root=str(eef_video_root.resolve()),
        video_source_dataset="RoboDojo_ee_lerobot_v30_video",
        video_source_revision=full["source_revision"],
        domain_id=17,
        resolution="720",
        model_canvas_wh=[960, 736],
        source_prepared_root=str(source.resolve()),
        reference_joint_manifest=str((reference / "manifest.json").resolve()),
        source_domain_id=full["domain_id"],
        domain_id_reason="Pair the initial Edge domain17 embedding with the independent exp1 joint control",
        source_dataset_audit=full["audit"],
        normalization_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
        normalization_fit_scope="Frozen all35-task training-only statistics, reused without refitting",
        preparation=dict(
            tasks=sorted(TASKS),
            demos=dict(counts),
            windows={s: sum((e["length"] - 17) // 8 + 1 for e in episodes if e["split"] == s) for s in counts},
            reference_episode_order_preserved=True,
            official_eef_video_offsets=True,
            rgb_files=len(videos),
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), **manifest["preparation"])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--eef-video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.reference, a.eef_video_root, a.output)


if __name__ == "__main__":
    main()
