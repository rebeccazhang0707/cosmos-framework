"""Pair each scheduled SR checkpoint with one independent held-out loss job.

Completed SR checkpoints are backfilled in one allocation. Future validation
jobs depend on the corresponding training boundary, and never gate training
or SR. SR phase2 resumes do not duplicate held-out evaluation of identical EMA.
"""

import argparse
import fcntl
import json
import subprocess
from pathlib import Path


def submission_groups(plan, root):
    paired = set()
    for row in plan["jobs"]:
        if row["kind"] == "validation":
            paired.update(row.get("segments", [row["segment"]]))
    stages = sorted({row["segment"] for row in plan["jobs"] if row["kind"] == "eval"} - paired)
    ready, future = [], []
    for stage in stages:
        result = root / f"continuation/evaluations/segment{stage:02d}/validation/result.json"
        if result.is_file() and json.loads(result.read_text()).get("complete"):
            continue
        finish = root / f"continuation/segment{stage:02d}_finish.json"
        if finish.is_file():
            if json.loads(finish.read_text())["new_checkpoint"]:
                ready.append(stage)
        else:
            training = next(row for row in plan["jobs"] if row["kind"] == "train" and row["segment"] == stage)
            future.append(([stage], f"afterok:{training['job_id']}"))
    return ([(ready, None)] if ready else []) + future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    chain = args.run_root / "continuation"
    with (chain / ".validation_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan_path = chain / "plan.json"
        plan = json.loads(plan_path.read_text())
        groups = submission_groups(plan, args.run_root)
        if args.dry_run:
            print(json.dumps(groups))
            return
        for stages, dependency in groups:
            tag = "_".join(map(str, stages))
            command = [
                "sbatch",
                "--parsable",
                "--qos=user-restrictions",
                "--nice=0",
                f"--job-name=general_sa-rdojo.eef4.val{tag}",
                f"--output={chain}/validation-{tag}-%j.log",
            ]
            if dependency:
                command.append(f"--dependency={dependency}")
            command += [
                str(args.repo / "examples/robodojo/eef_policy/validation.sbatch"),
                str(args.run_root),
                str(args.repo),
                ",".join(map(str, stages)),
            ]
            identifier = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not identifier.isdigit():
                raise ValueError(f"Invalid job ID: {identifier!r}")
            record = dict(
                kind="validation",
                segment=max(stages),
                segments=stages,
                job_id=identifier,
                dependency=dependency,
                nodes=1,
                gpu_workers=4,
                hours=4,
                source_commit=args.source_commit,
                source_repo=str(args.repo),
            )
            plan["jobs"].append(record)
            plan["validation_protocol"] = dict(
                weights="EMA",
                split="val",
                noise_seed=42,
                batch_size=32,
                coverage="all held-out windows",
                refresh_curves_after_each_checkpoint=True,
            )
            temp = plan_path.with_suffix(".tmp")
            temp.write_text(json.dumps(plan, indent=2) + "\n")
            temp.replace(plan_path)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
