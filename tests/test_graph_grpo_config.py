import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from cli.config_loader import apply_config_to_env, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGraphGrpoConfig(unittest.TestCase):
    def test_graph_grpo_config_uses_w_sampling_knobs(self):
        config = load_config("graph_grpo")
        grpo = config["grpo"]

        self.assertEqual(grpo["n_groups"], 4)
        self.assertEqual(grpo["noise_scale"], 0.1)
        self.assertNotIn("alphas", grpo)

    def test_apply_config_to_env_sets_new_grpo_env_vars_without_alphas(self):
        config = load_config("graph_grpo")

        with patch.dict(os.environ, {}, clear=True):
            apply_config_to_env(config)

            self.assertEqual(os.environ["GRPO_N_GROUPS"], "4")
            self.assertEqual(os.environ["GRPO_NOISE_SCALE"], "0.1")
            self.assertEqual(os.environ["GRPO_REF_ALPHA"], "1.0")
            self.assertNotIn("GRPO_ALPHAS", os.environ)

    def test_run_graph_grpo_script_uses_new_rollout_knobs(self):
        script_text = (PROJECT_ROOT / "scripts" / "run_graph_grpo.sh").read_text(encoding="utf-8")

        self.assertIn("GRPO_N_GROUPS", script_text)
        self.assertIn("GRPO_NOISE_SCALE", script_text)
        self.assertIn("DEBUG=false", script_text)
        self.assertNotIn("GRPO_ALPHAS", script_text)


if __name__ == "__main__":
    unittest.main()
