"""CPU checks for conditioning isolation and clean deadline termination."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.callbacks.robodojo import RoboDojoTrainingMonitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.monitor = RoboDojoTrainingMonitor()
        self.monitor.root = Path(self.temporary.name)
        self.monitor.checked = False
        self.monitor.iteration = None
        self.model = SimpleNamespace(config=SimpleNamespace(input_video_key="video"))
        self.plan = SimpleNamespace(condition_frame_indexes_vision=[0], condition_frame_indexes_action=[0])
        self.data = dict(
            sequence_plan=[self.plan],
            video=[[torch.zeros(3, 17, 544, 736, dtype=torch.uint8)]],
            action=[[torch.zeros(17, 64)]],
        )

    def test_wrong_clean_state_or_canvas_is_rejected_before_training(self):
        self.plan.condition_frame_indexes_action = [1]
        with self.assertRaises(ValueError):
            self.monitor.on_training_step_start(self.model, self.data)
        self.plan.condition_frame_indexes_action = [0]
        self.data["video"] = [[torch.zeros(3, 17, 480, 736, dtype=torch.uint8)]]
        with self.assertRaises(ValueError):
            self.monitor.on_training_step_start(self.model, self.data)
        with self.assertRaises(RuntimeError):
            self.monitor.on_before_backward(self.model, torch.tensor(float("nan")))

    def test_deadline_requests_native_checkpoint_exit(self):
        self.monitor.deadline = 1
        self.monitor.last_end = 0
        self.monitor.trainer = SimpleNamespace(max_iterations=2000)
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original_tensor(*args, **kwargs)

        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.mem_get_info", return_value=(20, 100)),
            patch("torch.cuda.memory_reserved", return_value=30),
            patch("torch.cuda.max_memory_reserved", return_value=40),
            patch("torch.cuda.max_memory_allocated", return_value=25),
            patch("torch.tensor", side_effect=cpu_tensor),
        ):
            self.monitor.on_training_step_end(
                self.model,
                {},
                dict(flow_matching_loss_action=original_tensor(0.1), flow_matching_loss_vision=original_tensor(0.2)),
                original_tensor(1.0),
                iteration=19,
            )
        self.assertEqual(self.monitor.trainer.max_iterations, 19)
        self.assertEqual(json.loads((self.monitor.root / "budget_stop.json").read_text())["step"], 19)


if __name__ == "__main__":
    unittest.main()
