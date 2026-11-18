"""Refresh an existing SR run while its native episode workers keep running.

This monitor performs CPU-only publication and never restarts inference or
re-encodes video. The publisher owns native-result aggregation and run-scoped
video retention. A per-run lock prevents duplicate monitors.
"""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "REVOKED",
    "DEADLINE",
    "OUT_OF_MEMORY",
    "SPECIAL_EXIT",
}


def publication_state(report):
    expected = report.get("manifest", {}).get("retained_episodes_per_family", 5)
    media_complete = all(len(row["videos"]) >= expected for row in report.get("families", {}).values())
    if report.get("complete") and report.get("families") and media_complete:
        return "complete"
    wanted = {job["job_id"] for job in (report.get("submission") or {}).get("jobs", [])}
    states = {row["job_id"]: row["state"].split()[0].rstrip("+") for row in report.get("job_states", [])}
    if wanted and wanted <= states.keys() and all(states[job] in TERMINAL for job in wanted):
        return "finished_incomplete"
    return "running"


def publish_once(run_dir, command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Publisher exit {result.returncode}: {result.stderr[-2000:]}")
    report = json.loads((run_dir / "public/report.json").read_text())
    return {
        "state": publication_state(report),
        "episodes": report["episodes"],
        "target": report["target"],
        "videos_by_task": {task: len(row["videos"]) for task, row in report["families"].items()},
        "report_updated_at": report["updated_at"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError("Publication interval must be 1–60 seconds")
    with (args.run_dir / ".publication-watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A publication monitor is already active for this run", flush=True)
            return
        public = args.run_dir / "public"
        public.mkdir(exist_ok=True)
        while True:
            started = time.monotonic()
            try:
                status = publish_once(args.run_dir, [args.python, str(args.publisher), "--publish"])
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                status = {"state": "publication_error", "error": str(exc)}
            status.update(
                pid=os.getpid(), heartbeat_utc=datetime.now(timezone.utc).isoformat(), interval_seconds=args.interval
            )
            temporary = public / "publication_status.json.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(public / "publication_status.json")
            print(json.dumps(status), flush=True)
            if args.once or status["state"] in {"complete", "finished_incomplete"}:
                return
            time.sleep(max(1, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
