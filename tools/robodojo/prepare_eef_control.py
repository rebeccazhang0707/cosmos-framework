"""Select the exact exp1 four-task demos from the frozen EEF20 dataset.

Preserve the reference episode order/splits and video offsets. Reuse the source
training-only normalizers and NPZ files. Domain17 pairs the initial embedding
with the independent joint control; the all35 source manifest is unchanged.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

TASKS = {"classify_objects", "insert_tubes", "push_T", "stack_bowls"}


def prepare(source, reference, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    full = json.loads((source / "manifest.json").read_text())
    ref = json.loads((reference / "manifest.json").read_text())
    if full.get("action_representation") != "eef_relative_chunk" or full["action_dim"] != 20:
        raise ValueError("Expected the frozen normalized EEF20 source")
    if ref.get("action_dim", 14) != 14 or len(ref["episodes"]) != 400:
        raise ValueError("Expected the original400-demo joint reference")
    if Path(full["root"]).resolve() != Path(ref["root"]).resolve():
        raise ValueError("Reference and source must use the same RGB dataset")
    if Counter(e["task_name"] for e in ref["episodes"]) != Counter(dict.fromkeys(TASKS, 100)):
        raise ValueError("Reference must contain exactly the original four tasks")
    by_id = {e["episode_index"]: e for e in full["episodes"]}
    episodes = []
    for old in ref["episodes"]:
        e = by_id[old["episode_index"]]
        if any(old[k] != e.get(k) for k in old):
            raise ValueError(f"Source/reference episode metadata mismatch: {old['episode_index']}")
        episodes.append(e)
    counts = Counter(e["split"] for e in episodes)
    if counts != {"train": 360, "val": 40}:
        raise ValueError(f"Unexpected split: {counts}")
    stats = source / full["normalization_file"]
    normalizers = json.loads(stats.read_text())
    if normalizers["validation_used"] or normalizers["fit_split"] != "train":
        raise ValueError("Source statistics must exclude validation")
    output.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        filename = f"episode_{e['episode_index']:06d}.npz"
        target = (source / filename).resolve(strict=True)
        (output / filename).symlink_to(target)
    shutil.copyfile(stats, output / full["normalization_file"])
    manifest = {k: v for k, v in full.items() if k not in ("episodes", "audit", "preparation")}
    manifest.update(
        episodes=episodes,
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
            reference_video_offsets_preserved=True,
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), **manifest["preparation"])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.reference, a.output)


if __name__ == "__main__":
    main()
