"""Check held-out coverage, weighting, and idempotent SR/validation pairing."""

import json
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.evaluation.robodojo_eef_validation import merge_results, validation_plan
from examples.robodojo.eef_policy.submit_validations import submission_groups


class ValidationTests(unittest.TestCase):
    def test_window_offsets_exclude_training_episodes(self):
        manifest = {
            "episodes": [
                dict(split="train", task_name="a", length=1000),
                dict(split="val", task_name="a", length=33),
                dict(split="train", task_name="b", length=1000),
                dict(split="val", task_name="b", length=25),
                dict(split="val", task_name="a", length=17),
            ]
        }
        self.assertEqual(validation_plan(manifest), {"a": [0, 1, 2, 5], "b": [3, 4]})

    def fixture(self, root):
        output = root / "segment02/validation"
        (output / "results").mkdir(parents=True)
        checkpoint = root / "iter_000000511"
        plan = dict(
            worker_count=2,
            noise_seed=42,
            batch_size=32,
            validation_windows=12,
            validation_episodes=2,
            windows_by_task={"a": 10, "b": 2},
            metrics=["total_loss"],
            metric_definition="native FM",
            manifest_sha256="manifest",
            normalization_sha256="normalizer",
        )
        for worker, (task, windows, loss) in enumerate([("a", 10, 1.0), ("b", 2, 7.0)]):
            part = dict(
                checkpoint=str(checkpoint),
                worker=worker,
                seed=42,
                batch_size=32,
                weights="EMA",
                smoke_only=False,
                manifest_sha256="manifest",
                normalization_sha256="normalizer",
                per_task={task: dict(windows=windows, total_loss=loss)},
                elapsed_seconds=1,
            )
            (output / f"results/{checkpoint.name}.worker_{worker}.json").write_text(json.dumps(part))
        return output, checkpoint, plan

    def test_unequal_windows_are_weighted_and_partial_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output, checkpoint, plan = self.fixture(Path(temp))
            result = merge_results(output, checkpoint, plan)
            self.assertEqual(result["overall"]["total_loss"], 2.0)
            self.assertEqual(result["task_macro_mean"]["total_loss"], 4.0)
            self.assertTrue((output.parent / "public/validation.json").is_file())
            part_path = output / f"results/{checkpoint.name}.worker_1.json"
            part = json.loads(part_path.read_text())
            part["per_task"]["b"]["windows"] = 1
            part_path.write_text(json.dumps(part))
            with self.assertRaises(ValueError):
                merge_results(output, checkpoint, plan)

    def test_reject_wrong_weights_or_normalizers(self):
        with tempfile.TemporaryDirectory() as temp:
            output, checkpoint, plan = self.fixture(Path(temp))
            path = output / f"results/{checkpoint.name}.worker_0.json"
            part = json.loads(path.read_text())
            for key, value in [("weights", "online"), ("normalization_sha256", "different"), ("smoke_only", True)]:
                wrong = {**part, key: value}
                path.write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):
                    merge_results(output, checkpoint, plan)

    def test_completed_backfill_and_future_jobs_pair_once_per_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "continuation").mkdir()
            jobs = []
            for stage in (2, 3, 4):
                jobs += [
                    dict(kind="train", segment=stage, job_id=str(100 + stage)),
                    dict(kind="eval", segment=stage, phase=1),
                    dict(kind="eval", segment=stage, phase=2),
                ]
            plan = dict(jobs=jobs)
            for stage in (2, 3):
                (root / f"continuation/segment{stage:02d}_finish.json").write_text('{"new_checkpoint":true}')
            self.assertEqual(submission_groups(plan, root), [([2, 3], None), ([4], "afterok:104")])
            plan["jobs"].append(dict(kind="validation", segment=3, segments=[2, 3]))
            self.assertEqual(submission_groups(plan, root), [([4], "afterok:104")])
            (root / "continuation/segment04_finish.json").write_text('{"new_checkpoint":false}')
            self.assertEqual(submission_groups(plan, root), [])


if __name__ == "__main__":
    unittest.main()
