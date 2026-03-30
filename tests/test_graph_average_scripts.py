import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_AVERAGE_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "run_graph_average.sh",
    PROJECT_ROOT / "scripts" / "blocking" / "run_graph_average.sh",
)


class TestGraphAverageScripts(unittest.TestCase):
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

    def test_scripts_accept_jailbreakbench_category_flags(self):
        for script_path in GRAPH_AVERAGE_SCRIPTS:
            temp_dir, env = self._build_env()
            with temp_dir:
                result = subprocess.run(
                    [
                        "bash",
                        str(script_path),
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

    def test_scripts_reject_category_without_jailbreakbench_source(self):
        for script_path in GRAPH_AVERAGE_SCRIPTS:
            result = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--category",
                    "Physical harm",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "--category requires --category-dataset-source jailbreakbench",
                result.stderr,
            )

    def test_scripts_reject_unknown_category(self):
        for script_path in GRAPH_AVERAGE_SCRIPTS:
            result = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--category-dataset-source",
                    "jailbreakbench",
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

    def test_scripts_reject_old_harmbench_interface(self):
        for script_path in GRAPH_AVERAGE_SCRIPTS:
            result = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--category-dataset-source",
                    "harmbench_semantic",
                    "--semantic-category",
                    "illegal",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(
                "invalid category dataset source" in result.stderr
                or "unknown argument '--semantic-category'" in result.stderr
            )


if __name__ == "__main__":
    unittest.main()
