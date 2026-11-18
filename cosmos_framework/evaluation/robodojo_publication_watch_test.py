"""Do not stop publishing while videos or a continuation job remain pending."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.evaluation.robodojo_publication_watch import publication_state, publish_once


class PublicationTests(unittest.TestCase):
    def report(self):
        return dict(
            complete=False,
            episodes=10,
            target=600,
            updated_at="now",
            manifest=dict(retained_episodes_per_family=5),
            families={"task": {"videos": [1, 2]}},
            submission=dict(jobs=[dict(job_id="1"), dict(job_id="2")]),
            job_states=[dict(job_id="1", state="COMPLETED"), dict(job_id="2", state="PENDING")],
        )

    def test_continue_through_job_boundary_and_missing_media(self):
        report = self.report()
        self.assertEqual(publication_state(report), "running")
        report["complete"] = True
        self.assertEqual(publication_state(report), "running")
        report["families"]["task"]["videos"] = list(range(5))
        self.assertEqual(publication_state(report), "complete")

    def test_missing_accounting_is_not_treated_as_finished(self):
        report = self.report()
        report["job_states"] = []
        self.assertEqual(publication_state(report), "running")
        report["job_states"] = [dict(job_id="1", state="UNKNOWN"), dict(job_id="2", state="UNKNOWN")]
        self.assertEqual(publication_state(report), "running")
        report["job_states"] = [dict(job_id="1", state="COMPLETED"), dict(job_id="2", state="CANCELLED by 123")]
        self.assertEqual(publication_state(report), "finished_incomplete")

    def test_cpu_publisher_updates_existing_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir()
            publisher = root / "publish.py"
            payload = json.dumps(self.report())
            publisher.write_text(
                "from pathlib import Path\nPath(__file__).parent.joinpath('public/report.json').write_text("
                + repr(payload)
                + ")\n"
            )
            status = publish_once(root, [sys.executable, str(publisher)])
            self.assertEqual(status["episodes"], 10)
            self.assertEqual(status["videos_by_task"], {"task": 2})
            self.assertEqual(status["state"], "running")


if __name__ == "__main__":
    unittest.main()
