"""Registration must resolve both adapters and preserve existing workspace edits."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("install_vit_policy", Path(__file__).with_name("install_vit_policy.py"))
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bench = Path(self.temporary.name) / "RoboDojo"
        self.policies = self.bench / "XPolicyLab/policy"
        self.policies.mkdir(parents=True)
        self.client = self.bench / "src/eval_client/main.py"
        self.client.parent.mkdir(parents=True)
        self.client.write_text("import argparse\nparser = argparse.ArgumentParser()\n")

    def test_registers_both_import_locations_and_patches_native_client(self):
        result = INSTALLER.install(self.bench)
        self.assertTrue(result["client_patched"])
        self.assertEqual(len(result["installed_links"]), 2)
        for name, relative in [
            ("Cosmos3_Edge_RoboDojo", "joint_policy"),
            ("Cosmos3_Edge_RoboDojo_ViT", "reasoner_vit/policy"),
        ]:
            link = self.policies / name
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.resolve(), Path(__file__).parent.resolve() / relative)
            for file in ("__init__.py", "model.py", "deploy.py", "deploy.yml"):
                self.assertTrue((link / file).is_file())
        self.assertIn("allow_abbrev=False", self.client.read_text())

    def test_repeated_installation_preserves_fixed_client_and_links(self):
        INSTALLER.install(self.bench)
        self.client.write_text(self.client.read_text() + "# unrelated local change\n")
        before = self.client.read_bytes()
        result = INSTALLER.install(self.bench)
        self.assertEqual(result["installed_links"], [])
        self.assertFalse(result["client_patched"])
        self.assertEqual(self.client.read_bytes(), before)

    def test_existing_adapter_prevents_all_changes(self):
        existing = self.policies / "Cosmos3_Edge_RoboDojo_ViT"
        existing.mkdir()
        (existing / "model.py").write_text("# user's adapter\n")
        before = self.client.read_bytes()
        with self.assertRaises(FileExistsError):
            INSTALLER.install(self.bench)
        self.assertEqual(self.client.read_bytes(), before)
        self.assertFalse((self.policies / "Cosmos3_Edge_RoboDojo").exists())
        self.assertEqual((existing / "model.py").read_text(), "# user's adapter\n")

    def test_dangling_old_machine_link_is_preserved_for_explicit_repair(self):
        existing = self.policies / "Cosmos3_Edge_RoboDojo_ViT"
        existing.symlink_to("/unavailable/old-machine/policy", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            INSTALLER.install(self.bench)
        self.assertEqual(os.readlink(existing), "/unavailable/old-machine/policy")
        self.assertFalse((self.policies / "Cosmos3_Edge_RoboDojo").exists())

    def test_unrecognized_native_parser_prevents_registration(self):
        self.client.write_text("import argparse\nparser = argparse.ArgumentParser(description='custom')\n")
        before = self.client.read_bytes()
        with self.assertRaises(ValueError):
            INSTALLER.install(self.bench)
        self.assertEqual(self.client.read_bytes(), before)
        self.assertEqual(list(self.policies.iterdir()), [])

    def test_missing_native_client_prevents_registration(self):
        self.client.unlink()
        with self.assertRaises(FileNotFoundError):
            INSTALLER.install(self.bench)
        self.assertEqual(list(self.policies.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
