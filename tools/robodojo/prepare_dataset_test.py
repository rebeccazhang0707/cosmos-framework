"""Episode isolation and deterministic split checks for the standalone preparer."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.robodojo.prepare_dataset import CAMERAS, assign_splits, prepare


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [dict(episode_index=i, task_index=i // 3) for i in range(6)]
        self.spec = dict(
            tasks=[
                dict(task_index=0, name="a", episodes=3, validation_episodes=[1]),
                dict(task_index=1, name="b", episodes=3, validation_episodes=[4]),
            ]
        )

    def test_exact_heldouts_and_order_do_not_depend_on_metadata_order(self):
        result = assign_splits(self.episodes, self.spec)
        self.assertEqual(result, assign_splits(list(reversed(self.episodes)), self.spec))
        train = {e["episode_index"] for e in result if e["split"] == "train"}
        val = {e["episode_index"] for e in result if e["split"] == "val"}
        self.assertEqual(val, {1, 4})
        self.assertEqual(train | val, set(range(6)))
        self.assertFalse(train & val)

    def test_missing_heldout_or_duplicate_episode_is_rejected(self):
        wrong = copy.deepcopy(self.spec)
        wrong["tasks"][0]["validation_episodes"] = [8]
        with self.assertRaises(ValueError):
            assign_splits(self.episodes, wrong)
        with self.assertRaises(ValueError):
            assign_splits(self.episodes + [self.episodes[0]], self.spec)

    def test_task_coverage_and_nonempty_train_are_required(self):
        wrong = copy.deepcopy(self.spec)
        wrong["tasks"][0]["validation_episodes"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            assign_splits(self.episodes, wrong)
        with self.assertRaises(ValueError):
            assign_splits(self.episodes[:3], self.spec)

    def test_parquet_offsets_alignment_and_train_only_statistics(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            (root / "meta").mkdir(parents=True)
            info = dict(
                fps=25,
                data_path="data.parquet",
                video_path="{video_key}.mp4",
                features={key: dict(shape=[14]) for key in ("action", "observation.state")},
            )
            (root / "meta/info.json").write_text(json.dumps(info))
            for camera in CAMERAS:
                (root / f"{camera}.mp4").touch()
            episodes = assign_splits(self.episodes, self.spec)
            rows = []
            for e in episodes:
                e.update(length=19, instruction="Move.", **{"data/chunk_index": 0, "data/file_index": 0})
                for camera in CAMERAS:
                    e.update({f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": 0})
                for frame in range(19):
                    value = e["episode_index"] * 100 + frame
                    rows.append(
                        dict(
                            episode_index=e["episode_index"],
                            frame_index=frame,
                            timestamp=frame / 25,
                            task_index=e["task_index"],
                            action=[value + 1.0] * 14,
                            **{"observation.state": [float(value)] * 14},
                        )
                    )
            pq.write_table(pa.Table.from_pylist(rows), root / "data.parquet")
            output = Path(temporary) / "prepared"
            spec = dict(self.spec, seed=42, source_revision="test")
            prepare(root, output, spec, episodes)
            for e in episodes:
                with np.load(output / f"episode_{e['episode_index']:06d}.npz") as saved:
                    np.testing.assert_array_equal(saved["state"][:, 0], e["episode_index"] * 100 + np.arange(19))
                    np.testing.assert_array_equal(saved["action"][:-1], saved["state"][1:])
            manifest = json.loads((output / "manifest.json").read_text())
            train_ids = [e["episode_index"] for e in episodes if e["split"] == "train"]
            self.assertAlmostEqual(manifest["training_action_stats"]["mean"][0], np.mean(train_ids) * 100 + 10)
            self.assertEqual(manifest["preparation"]["max_action_next_state_error"], 0)
            with self.assertRaises(FileExistsError):
                prepare(root, output, spec, episodes)


if __name__ == "__main__":
    unittest.main()
