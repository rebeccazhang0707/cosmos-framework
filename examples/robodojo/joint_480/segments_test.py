"""Continuation boundaries must preserve state and pin evaluation inputs."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from examples.robodojo.joint_480.segments import TRAIN_RELATIVE, latest, main


class SegmentTests(unittest.TestCase):
    def call(self, command, root, stage):
        stream = StringIO()
        with patch("sys.argv", ["segments.py", command, str(root), str(stage)]), redirect_stdout(stream):
            main()
        return stream.getvalue()

    def checkpoint(self, train, step):
        checkpoint = train / f"checkpoints/iter_{step:09d}"
        for key in ("model", "optim", "scheduler", "trainer"):
            (checkpoint / key).mkdir(parents=True, exist_ok=True)
            (checkpoint / key / ".metadata").touch()
        (train / "checkpoints/latest_checkpoint.txt").write_text(checkpoint.name)
        for name in ("config.yaml", "config.pkl"):
            (train / name).write_text(f"step{step}")
        return checkpoint

    def audit(self, root):
        (root / "control").mkdir()
        (root / "control/config_audit.json").write_text('{"passed":true}')

    def test_continuation_never_silently_reinitializes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.audit(root)
            self.assertEqual(self.call("prepare", root, 1).strip(), "FRESH")
            with self.assertRaises(ValueError):
                self.call("prepare", root, 2)
            train = root / TRAIN_RELATIVE
            path = self.checkpoint(train, 255)
            self.assertEqual(self.call("prepare", root, 2).strip(), "RESUME")
            (path / "optim/.metadata").unlink()
            with self.assertRaises(ValueError):
                latest(train)

    def test_snapshot_is_pinned_and_goal_cancels_only_future_segments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.audit(root)
            train = root / TRAIN_RELATIVE
            self.call("prepare", root, 1)
            first = self.checkpoint(train, 255)
            self.call("finish", root, 1)
            snapshot = root / "continuation/snapshots/segment01"
            self.call("prepare", root, 2)
            self.checkpoint(train, 1600)
            plan = dict(
                jobs=[
                    dict(segment=1, job_id="old_eval"),
                    dict(segment=2, job_id="current_eval"),
                    dict(segment=3, job_id="future_train"),
                    dict(segment=3, job_id="future_eval"),
                ]
            )
            (root / "continuation/plan.json").write_text(json.dumps(plan))
            with patch("subprocess.run") as cancel:
                self.call("finish", root, 2)
            self.assertEqual((snapshot / "config.yaml").read_text(), "step255")
            self.assertEqual((snapshot / "checkpoints" / first.name).resolve(), first)
            self.assertEqual(cancel.call_args.args[0], ["scancel", "future_train", "future_eval"])
            self.assertTrue((root / "continuation/goal_reached.json").is_file())


if __name__ == "__main__":
    unittest.main()
