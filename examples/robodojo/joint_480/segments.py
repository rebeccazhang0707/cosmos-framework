"""Strict fresh initialization, full-state continuation, and pinned evaluation snapshots."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

TRAIN_RELATIVE = Path("train/robodojo/four_tasks_wam/edge_joint14_480_shift5_16gpu")
GOAL = 800


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def latest(train):
    pointer = train / "checkpoints/latest_checkpoint.txt"
    path = train / "checkpoints" / Path(pointer.read_text().strip()).name
    step = int(path.name.removeprefix("iter_"))
    for key in ("model", "optim", "scheduler", "trainer"):
        if not (path / key / ".metadata").is_file():
            raise ValueError(f"Incomplete full-state resume: {path / key}")
    return step, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finish"))
    parser.add_argument("run_root", type=Path)
    parser.add_argument("segment", type=int)
    args = parser.parse_args()
    root, stage = args.run_root, args.segment
    train = root / TRAIN_RELATIVE
    chain = root / "continuation"
    if args.command == "prepare":
        audit = json.loads((root / "control/config_audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("Configuration audit has not passed")
        if (train / "checkpoints/latest_checkpoint.txt").exists():
            step, checkpoint = latest(train)
            state = "COMPLETE" if step >= GOAL else "RESUME"
        elif stage == 1 and not list((train / "checkpoints").glob("iter_*")):
            step, checkpoint, state = 0, None, "FRESH"
        else:
            raise ValueError("Continuation cannot silently fall back to fresh initialization")
        atomic(
            chain / f"segment{stage:02d}_start.json",
            dict(
                stage=stage,
                job_id=os.environ.get("SLURM_JOB_ID"),
                resume_step=step,
                resume_checkpoint=str(checkpoint) if checkpoint else None,
                start_epoch=time.time(),
                no_op=state == "COMPLETE",
            ),
        )
        print(state, flush=True)
        return
    start = json.loads((chain / f"segment{stage:02d}_start.json").read_text())
    step, checkpoint = latest(train)
    if not start["no_op"] and step <= start["resume_step"]:
        raise ValueError("Training made no progress")
    snapshot = chain / f"snapshots/segment{stage:02d}"
    (snapshot / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "config.pkl"):
        shutil.copyfile(train / name, snapshot / name)
    link = snapshot / "checkpoints" / checkpoint.name
    if not link.exists():
        link.symlink_to(checkpoint, target_is_directory=True)
    record = dict(
        **start,
        final_step=step,
        final_checkpoint=str(checkpoint),
        snapshot_train_dir=str(snapshot),
        finish_epoch=time.time(),
        new_checkpoint=step > start["resume_step"],
    )
    atomic(chain / f"segment{stage:02d}_finish.json", record)
    if step >= GOAL:
        atomic(chain / "goal_reached.json", record)
        plan = json.loads((chain / "plan.json").read_text())
        jobs = [r["job_id"] for r in plan["jobs"] if r["segment"] > stage]
        if jobs:
            subprocess.run(["scancel", *jobs], check=True)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
