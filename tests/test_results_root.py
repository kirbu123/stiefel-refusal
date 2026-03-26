import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

import config
from cli.config_loader import apply_config_to_env
from cli.runner import PROJECT_ROOT, run_method


class TestResultsRoot(unittest.TestCase):
    def test_apply_config_to_env_sets_and_resets_results_root(self):
        with patch.dict(os.environ, {}, clear=False):
            apply_config_to_env({"model": {"name": "model-a"}, "output": {"results_root": "results/blocking"}})
            self.assertEqual(os.environ["RESULTS_ROOT"], "results/blocking")

            apply_config_to_env({"model": {"name": "model-a"}})
            self.assertEqual(os.environ["RESULTS_ROOT"], "results")

    def test_run_method_returns_results_dir_from_configured_root(self):
        fake_module = types.SimpleNamespace(main=lambda: None)
        with patch.dict(os.environ, {}, clear=False):
            with patch("cli.runner.importlib.import_module", return_value=fake_module):
                results_dir = run_method(
                    "basic_refusal",
                    {
                        "model": {"name": "model-a"},
                        "output": {"results_root": "results/blocking"},
                    },
                    "model-a",
                )

        self.assertEqual(results_dir, PROJECT_ROOT / "results" / "blocking" / "basic_refusal")

    def test_get_method_results_dir_uses_env_driven_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"RESULTS_ROOT": tmpdir}, clear=False):
                reloaded_config = importlib.reload(config)
                method_dir = reloaded_config.get_method_results_dir("basic_refusal")
                self.assertEqual(method_dir, Path(tmpdir) / "basic_refusal")
                self.assertTrue(method_dir.exists())

            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
