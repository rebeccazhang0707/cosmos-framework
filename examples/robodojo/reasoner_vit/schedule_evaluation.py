"""Queue one evaluation only after the requested complete checkpoint exists.

Runs on a login-node CPU. No GPU allocation is consumed waiting for training.
Insert the new evaluation between the running and pending baseline evaluations,
so validation and SR across both experiments use at most one evaluation node.
"""

import argparse
import fcntl
import getpass
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TRAIN = Path("train/robodojo/four_tasks_wam/edge_joint14_480_shift5_reasoner_vit_16gpu")


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def pin_checkpoint(root, step):
    train = root / TRAIN
    checkpoint = train / f"checkpoints/iter_{step:09d}"
    if not all((checkpoint / part / ".metadata").is_file() for part in ("model", "optim", "scheduler", "trainer")):
        return None
    snapshot = root / f"continuation/evaluation_snapshots/step{step:06d}"
    (snapshot / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "config.pkl"):
        if not (snapshot / name).exists():
            shutil.copyfile(train / name, snapshot / name)
    link = snapshot / "checkpoints" / checkpoint.name
    if not link.exists():
        link.symlink_to(checkpoint, target_is_directory=True)
    target = dict(
        final_step=step,
        final_checkpoint=str(checkpoint),
        snapshot_train_dir=str(snapshot),
        new_checkpoint=True,
        selection="Exact complete checkpoint at the user-requested first evaluation step",
    )
    atomic(root / f"continuation/evaluation_targets/step{step:06d}.json", target)
    return target


def schedule(root, base_root, step):
    status_path = root / "continuation/evaluation_schedule.json"
    with (root.parent / ".robodojo_evaluation_submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous_status.get("job_id"):
            return previous_status
        if pin_checkpoint(root, step) is None:
            value = dict(
                state="waiting_for_checkpoint",
                target_step=step,
                nodes=1,
                gpus=8,
                hours=4,
                training_waits_for_evaluation=False,
                watch_interval_seconds=60,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            atomic(status_path, value)
            return value
        base_path = base_root / "continuation/plan.json"
        base = json.loads(base_path.read_text())
        evaluations = sorted([r for r in base["jobs"] if r["kind"] == "eval"], key=lambda r: r["segment"])
        query = subprocess.check_output(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%R"], text=True
        )
        states = {parts[0]: parts[1:] for line in query.splitlines() if (parts := line.split("|", 2))}
        running = [
            r["job_id"]
            for r in evaluations
            if states.get(r["job_id"], [None])[0] in ("RUNNING", "CONFIGURING", "COMPLETING")
        ]
        pending = [r for r in evaluations if states.get(r["job_id"], [None])[0] == "PENDING"]
        following = pending[0] if pending else None
        original_base = json.dumps(base)
        original_following_dependency = following["dependency"] if following else None
        vit_plan_path = root / "continuation/plan.json"
        original_vit = vit_plan_path.read_text()
        submission_path = (
            base_root / f"continuation/evaluations/segment{following['segment']:02d}/submission.json"
            if following
            else None
        )
        original_submission = submission_path.read_text() if submission_path else None
        held_here = following is not None and states[following["job_id"]][1] != "(JobHeldUser)"
        if held_here:
            subprocess.run(["scontrol", "hold", following["job_id"]], check=True)
        job_id = None
        try:
            dependency = "afterany:" + ":".join(running) if running else None
            command = [
                "sbatch",
                "--parsable",
                "--hold",
                "--qos=user-restrictions",
                "--nice=0",
                f"--job-name=general_sa-rdojo.j480vit.eval{step}",
                f"--output={root}/continuation/eval-step{step}-%j.log",
            ]
            if dependency:
                command.append("--dependency=" + dependency)
            command.extend([str(root / "control/eval/evaluate.sbatch"), str(root), str(step)])
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError("Unexpected sbatch output")
            if following:
                parts = [x for x in following["dependency"].split(",") if not x.startswith("afterany:")]
                parts.append("afterany:" + job_id)
                dependency_after = ",".join(parts)
                subprocess.run(
                    ["scontrol", "update", "JobId=" + following["job_id"], "Dependency=" + dependency_after], check=True
                )
                following["dependency"] = dependency_after
                atomic(base_path, base)
                submission = base_root / f"continuation/evaluations/segment{following['segment']:02d}/submission.json"
                value = json.loads(submission.read_text())
                for row in value["jobs"]:
                    if row["job_id"] == following["job_id"]:
                        row["dependency"] = dependency_after
                atomic(submission, value)
            vit_plan = json.loads(vit_plan_path.read_text())
            vit_plan["jobs"].append(
                dict(
                    kind="eval",
                    segment=0,
                    target_step=step,
                    job_id=job_id,
                    dependency=dependency,
                    nodes=1,
                    gpus=8,
                    hours=4,
                )
            )
            atomic(vit_plan_path, vit_plan)
            result = dict(
                state="submitted",
                job_id=job_id,
                target_step=step,
                nodes=1,
                gpus=8,
                hours=4,
                dependency=dependency,
                next_baseline_eval=following["job_id"] if following else None,
                episodes_per_task=15,
                retained_video_bundles_per_task=5,
                maximum_concurrent_evaluation_jobs=1,
                submitted_at=datetime.now(timezone.utc).isoformat(),
            )
            atomic(
                root / f"continuation/evaluations/step{step:06d}/submission.json",
                dict(
                    jobs=[dict(job_id=job_id, kind="combined_eval", nodes=1, gpus=8, dependency=dependency)],
                    goal_episodes=60,
                    episodes_per_task=15,
                    videos_per_task=5,
                    validation_included_in_job=job_id,
                    validation_shifts=[5, 10],
                ),
            )
            atomic(status_path, result)
            subprocess.run(["scontrol", "release", job_id], check=True)
            return result
        except Exception as exc:
            if following:
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        "JobId=" + following["job_id"],
                        "Dependency=" + original_following_dependency,
                    ],
                    check=False,
                )
                atomic(base_path, json.loads(original_base))
                atomic(submission_path, json.loads(original_submission))
            atomic(vit_plan_path, json.loads(original_vit))
            if job_id:
                subprocess.run(["scancel", job_id], check=False)
            atomic(
                status_path, dict(state="submission_error", target_step=step, cancelled_attempt=job_id, error=str(exc))
            )
            raise
        finally:
            if held_here:
                subprocess.run(["scontrol", "release", following["job_id"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        result = schedule(args.run_root, args.base_run, args.step)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once or result.get("job_id"):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
