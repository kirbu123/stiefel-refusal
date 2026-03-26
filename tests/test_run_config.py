import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from cli.run_config import main


class TestRunConfig(unittest.TestCase):
    def test_main_uses_config_model_when_override_missing(self):
        config = {"model": {"name": "config-model"}}

        with patch("cli.run_config.load_config", return_value=config) as load_mock:
            with patch("cli.run_config.print_config_summary") as summary_mock:
                with patch("cli.run_config.run_method", return_value=Path("results/basic_refusal")) as run_mock:
                    exit_code = main(
                        ["--method", "basic_refusal", "--config", "configs/basic_refusal.toml"]
                    )

        self.assertEqual(exit_code, 0)
        load_mock.assert_called_once_with("basic_refusal", "configs/basic_refusal.toml")
        summary_mock.assert_called_once_with(config, "config-model")
        run_mock.assert_called_once_with("basic_refusal", config, "config-model")

    def test_main_prefers_explicit_model_override(self):
        config = {"model": {"name": "config-model"}}

        with patch("cli.run_config.load_config", return_value=config):
            with patch("cli.run_config.print_config_summary"):
                with patch("cli.run_config.run_method", return_value=Path("results/basic_refusal")) as run_mock:
                    exit_code = main(
                        [
                            "--method",
                            "basic_refusal",
                            "--config",
                            "configs/basic_refusal.toml",
                            "--model",
                            "override-model",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with("basic_refusal", config, "override-model")


if __name__ == "__main__":
    unittest.main()
