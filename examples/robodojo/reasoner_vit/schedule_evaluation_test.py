"""Do not consume GPUs waiting for a checkpoint or duplicate scheduled evaluation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from examples.robodojo.reasoner_vit.schedule_evaluation import TRAIN, pin_checkpoint, schedule


def fixture(root):
    (root / "continuation").mkdir(parents=True)
    (root / "continuation/plan.json").write_text('{"jobs":[]}')
    train = root / TRAIN
    train.mkdir(parents=True)
    (train / "config.yaml").write_text("model: image_conditioned\n")
    (train / "config.pkl").write_bytes(b"config")
    return train


def test_waits_for_every_checkpoint_component_without_contacting_slurm():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vit"
        train = fixture(root)
        checkpoint = train / "checkpoints/iter_000001000"
        for part in ("model", "optim", "scheduler"):
            (checkpoint / part).mkdir(parents=True)
            (checkpoint / part / ".metadata").touch()
        with patch("subprocess.check_output") as query:
            assert schedule(root, Path(tmp) / "base", 1000)["state"] == "waiting_for_checkpoint"
            query.assert_not_called()
        assert not (root / "continuation/evaluation_snapshots").exists()
        (checkpoint / "trainer").mkdir()
        (checkpoint / "trainer/.metadata").touch()
        target = pin_checkpoint(root, 1000)
        pinned = Path(target["snapshot_train_dir"])
        assert target["final_step"] == 1000
        assert (pinned / "checkpoints/iter_000001000").resolve() == checkpoint
        assert (pinned / "config.yaml").read_text() == (train / "config.yaml").read_text()


def test_submits_once_between_running_and_pending_baseline_evaluations():
    with tempfile.TemporaryDirectory() as tmp:
        root, base = Path(tmp) / "vit", Path(tmp) / "base"
        train = fixture(root)
        for part in ("model", "optim", "scheduler", "trainer"):
            (train / f"checkpoints/iter_000001000/{part}").mkdir(parents=True)
            (train / f"checkpoints/iter_000001000/{part}/.metadata").touch()
        (base / "continuation/evaluations/segment04").mkdir(parents=True)
        (base / "continuation/plan.json").write_text(
            json.dumps(
                dict(
                    jobs=[
                        dict(kind="eval", segment=2, job_id="111", dependency="afterok:333"),
                        dict(kind="eval", segment=4, job_id="222", dependency="afterok:444,afterany:111"),
                    ]
                )
            )
        )
        (base / "continuation/evaluations/segment04/submission.json").write_text('{"jobs":[{"job_id":"222"}]}')
        with (
            patch(
                "subprocess.check_output", side_effect=["111|RUNNING|node1\n222|PENDING|(Dependency)\n", "999\n"]
            ) as query,
            patch("subprocess.run") as mutate,
        ):
            first = schedule(root, base, 1000)
            again = schedule(root, base, 1000)
            assert first == again
            assert first["dependency"] == "afterany:111" and first["job_id"] == "999"
            assert query.call_count == 2
            submission = query.call_args_list[1].args[0]
            assert "--hold" in submission and "--dependency=afterany:111" in submission
            calls = [call.args[0] for call in mutate.call_args_list]
            assert ["scontrol", "hold", "222"] in calls
            assert ["scontrol", "update", "JobId=222", "Dependency=afterok:444,afterany:999"] in calls
            assert ["scontrol", "release", "999"] in calls
            assert ["scontrol", "release", "222"] in calls
        updated = json.loads((base / "continuation/plan.json").read_text())
        assert updated["jobs"][1]["dependency"] == "afterok:444,afterany:999"
        new_plan = json.loads((root / "continuation/plan.json").read_text())
        assert new_plan["jobs"] == [
            dict(
                kind="eval",
                segment=0,
                target_step=1000,
                job_id="999",
                dependency="afterany:111",
                nodes=1,
                gpus=8,
                hours=4,
            )
        ]


def test_additional_checkpoint_keeps_milestone_and_serializes_evaluations():
    with tempfile.TemporaryDirectory() as tmp:
        root, base = Path(tmp) / "vit", Path(tmp) / "base"
        train = fixture(root)
        fixture(base)
        original_base = dict(jobs=[dict(kind="eval", segment=4, job_id="111", dependency="afterok:444")])
        (base / "continuation/plan.json").write_text(json.dumps(original_base))
        milestone = root / "continuation/evaluation_schedule.json"
        milestone.write_text('{"state":"waiting_for_checkpoint","target_step":1000}')
        for step in (665, 1000):
            for part in ("model", "optim", "scheduler", "trainer"):
                p = train / f"checkpoints/iter_{step:09d}/{part}"
                p.mkdir(parents=True)
                (p / ".metadata").touch()
        extra = root / "continuation/evaluation_schedule_step000665.json"
        with (
            patch("subprocess.check_output", side_effect=["111|PENDING|(Priority)\n", "999\n"]),
            patch("subprocess.run") as mutate,
        ):
            result = schedule(root, base, 665, extra, after_eval="111")
        assert result["job_id"] == "999" and result["target_step"] == 665
        assert result["dependency"] == "afterany:111"
        assert json.loads((base / "continuation/plan.json").read_text()) == original_base
        assert not any(call.args[0][1] == "hold" for call in mutate.call_args_list)
        assert json.loads(milestone.read_text()) == dict(state="waiting_for_checkpoint", target_step=1000)
        with patch("subprocess.check_output", return_value="999|RUNNING|node1\n") as query:
            result = schedule(root, base, 1000)
            assert query.call_count == 1  # Queue query only; no second sbatch allocation.
        assert result["state"] == "waiting_for_prior_evaluation"
        assert result["prior_evaluation_jobs"] == ["999"] and result["target_step"] == 1000
        assert not result.get("job_id")
