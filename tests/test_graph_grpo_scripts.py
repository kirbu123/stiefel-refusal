import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_GRPO_SCRIPT = PROJECT_ROOT / "scripts" / "run_graph_grpo.sh"


class TestGraphGrpoScripts(unittest.TestCase):
    def _build_env(self):
        temp_dir = tempfile.TemporaryDirectory()
        fake_python = Path(temp_dir.name) / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "echo \"ARGS:$*\"\n"
            "echo \"CATEGORY_DATASET_SOURCE=${CATEGORY_DATASET_SOURCE:-}\"\n"
            "echo \"CATEGORY_FILTER=${CATEGORY_FILTER:-}\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

        env = os.environ.copy()
        env["PATH"] = f"{temp_dir.name}:{env['PATH']}"
        return temp_dir, env

    def test_script_accepts_jailbreakbench_category_flags(self):
        temp_dir, env = self._build_env()
        with temp_dir:
            result = subprocess.run(
                [
                    "bash",
                    str(GRAPH_GRPO_SCRIPT),
                    "--category-dataset-source",
                    "jailbreakbench",
                    "--category",
                    "Physical harm",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("CATEGORY_DATASET_SOURCE=jailbreakbench", result.stdout)
        self.assertIn("CATEGORY_FILTER=Physical harm", result.stdout)

    def test_script_accepts_combined_category_flags(self):
        temp_dir, env = self._build_env()
        with temp_dir:
            result = subprocess.run(
                [
                    "bash",
                    str(GRAPH_GRPO_SCRIPT),
                    "--category-dataset-source",
                    "combined",
                    "--category",
                    "Privacy",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("CATEGORY_DATASET_SOURCE=combined", result.stdout)
        self.assertIn("CATEGORY_FILTER=Privacy", result.stdout)

    def test_script_rejects_unknown_dataset_source(self):
        result = subprocess.run(
            [
                "bash",
                str(GRAPH_GRPO_SCRIPT),
                "--category-dataset-source",
                "harmbench_semantic",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid category dataset source", result.stderr)

    def test_script_rejects_unknown_category(self):
        result = subprocess.run(
            [
                "bash",
                str(GRAPH_GRPO_SCRIPT),
                "--category",
                "unknown_category",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid category", result.stderr)

    def test_script_defaults_to_combined_physical_harm(self):
        temp_dir, env = self._build_env()
        with temp_dir:
            result = subprocess.run(
                [
                    "bash",
                    str(GRAPH_GRPO_SCRIPT),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("CATEGORY_DATASET_SOURCE=combined", result.stdout)
        self.assertIn("CATEGORY_FILTER=Physical harm", result.stdout)


if __name__ == "__main__":
    unittest.main()
