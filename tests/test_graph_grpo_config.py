import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from baselines.graph_grpo.runtime_config import (
    resolve_graph_grpo_debug_noise_scale,
    resolve_graph_grpo_debug_question_count,
    resolve_graph_grpo_weights_mode,
    resolve_graph_grpo_weights_init_type,
    validate_graph_grpo_weights_mode,
    validate_graph_grpo_weights_init_type,
)
from cli.config_loader import apply_config_to_env, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGraphGrpoConfig(unittest.TestCase):
    def test_graph_grpo_config_uses_w_sampling_knobs(self):
        config = load_config("graph_grpo")
        model = config["model"]
        grpo = config["grpo"]
        weights = config["weights"]

        self.assertEqual(model["batch_size"], 32)
        self.assertEqual(grpo["n_groups"], 4)
        self.assertEqual(grpo["noise_scale"], 0.1)
        self.assertNotIn("alphas", grpo)
        self.assertEqual(weights["mode"], "scalar")
        self.assertEqual(weights["init_type"], "average")

    def test_apply_config_to_env_sets_new_grpo_env_vars_without_alphas(self):
        config = load_config("graph_grpo")

        with patch.dict(os.environ, {}, clear=True):
            apply_config_to_env(config)

            self.assertEqual(os.environ["BATCH_SIZE"], "32")
            self.assertEqual(os.environ["GRPO_N_GROUPS"], "4")
            self.assertEqual(os.environ["GRPO_NOISE_SCALE"], "0.1")
            self.assertEqual(os.environ["GRPO_REF_ALPHA"], "1.0")
            self.assertEqual(os.environ["WEIGHTS_MODE"], "scalar")
            self.assertEqual(os.environ["WEIGHTS_INIT_TYPE"], "average")
            self.assertNotIn("GRPO_ALPHAS", os.environ)

    def test_run_graph_grpo_script_uses_average_init_and_debug_false(self):
        script_text = (PROJECT_ROOT / "scripts" / "run_graph_grpo.sh").read_text(encoding="utf-8")

        self.assertIn("GRPO_N_GROUPS", script_text)
        self.assertIn("GRPO_NOISE_SCALE", script_text)
        self.assertIn("BATCH_SIZE=32", script_text)
        self.assertIn("DEBUG=false", script_text)
        self.assertIn("DEBUG_N_QUESTIONS=4", script_text)
        self.assertIn("DEBUG_NOISE_SCALE=0.02", script_text)
        self.assertIn('WEIGHTS_MODE="scalar"', script_text)
        self.assertIn('WEIGHTS_INIT_TYPE="average"', script_text)
        self.assertNotIn("GRPO_ALPHAS", script_text)

    def test_graph_grpo_runtime_defaults_weights_mode_to_scalar(self):
        self.assertEqual(resolve_graph_grpo_weights_mode(None), "scalar")
        self.assertEqual(resolve_graph_grpo_weights_mode(""), "scalar")
        self.assertEqual(resolve_graph_grpo_weights_mode("dense"), "dense")

    def test_graph_grpo_runtime_validates_weights_mode(self):
        validate_graph_grpo_weights_mode("scalar")
        validate_graph_grpo_weights_mode("dense")

        with self.assertRaisesRegex(
            ValueError,
            "graph_grpo only supports WEIGHTS_MODE",
        ):
            validate_graph_grpo_weights_mode("weird")

    def test_graph_grpo_runtime_defaults_weights_init_to_average(self):
        self.assertEqual(resolve_graph_grpo_weights_init_type(None), "average")
        self.assertEqual(resolve_graph_grpo_weights_init_type(""), "average")
        self.assertEqual(resolve_graph_grpo_weights_init_type("topic"), "topic")

    def test_graph_grpo_runtime_rejects_zero_init(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not support WEIGHTS_INIT_TYPE='zero'",
        ):
            validate_graph_grpo_weights_init_type("zero")

        validate_graph_grpo_weights_init_type("average")
        validate_graph_grpo_weights_init_type("topic")

    def test_graph_grpo_runtime_debug_defaults(self):
        self.assertEqual(resolve_graph_grpo_debug_question_count(None), 4)
        self.assertEqual(resolve_graph_grpo_debug_question_count("8"), 8)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.1, None), 0.02)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.01, None), 0.01)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.1, "0.005"), 0.005)

        with self.assertRaisesRegex(ValueError, "DEBUG_N_QUESTIONS must be >= 1"):
            resolve_graph_grpo_debug_question_count("0")

        with self.assertRaisesRegex(ValueError, "DEBUG_NOISE_SCALE must be >= 0"):
            resolve_graph_grpo_debug_noise_scale(0.1, "-1")

    def test_graph_grpo_debug_mode_uses_configurable_question_subset(self):
        main_text = (PROJECT_ROOT / "baselines" / "graph_grpo" / "__main__.py").read_text(encoding="utf-8")
        trainer_text = (PROJECT_ROOT / "baselines" / "graph_grpo" / "trainer.py").read_text(encoding="utf-8")

        self.assertIn("model=MODEL_NAME, batch_size=MODEL_BATCH_SIZE", main_text)
        self.assertIn("effective_question_count = min(MODEL_BATCH_SIZE, len(all_category_questions))", main_text)
        self.assertIn("category_questions = random.sample(all_category_questions, k=effective_question_count)", main_text)
        self.assertIn("if DEBUG:", main_text)
        self.assertIn("effective_question_count = min(effective_question_count, debug_question_count)", main_text)
        self.assertIn("effective_noise_scale = debug_noise_scale if DEBUG else GRPO_CONFIG[\"noise_scale\"]", main_text)
        self.assertIn("resolve_graph_grpo_debug_question_count(os.getenv(\"DEBUG_N_QUESTIONS\"))", main_text)
        self.assertIn("resolve_graph_grpo_debug_noise_scale(", main_text)
        self.assertIn("resolve_graph_grpo_weights_mode(os.getenv(\"WEIGHTS_MODE\"))", main_text)
        self.assertIn("resolve_graph_grpo_weights_init_type(os.getenv(\"WEIGHTS_INIT_TYPE\"))", main_text)
        self.assertIn("Loss: {accumulated_loss:.6e}", trainer_text)
        self.assertIn("Grad norm: {grad_norm:.6e}", trainer_text)


if __name__ == "__main__":
    unittest.main()
