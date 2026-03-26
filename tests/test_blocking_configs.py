import sys
import types
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from cli.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOCKING_CONFIGS_DIR = PROJECT_ROOT / "configs" / "blocking"
BLOCKING_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "blocking"


class TestBlockingConfigs(unittest.TestCase):
    def test_blocking_configs_have_expected_output_and_weights(self):
        expected_model = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
        expected = {
            "basic_refusal": {
                "max_weight": [-1.0],
                "min_weight": [-1.0],
            },
            "topic_ablation": {
                "max_weight": [-2.5, -3.0],
                "min_weight": [0.0, -1.0],
            },
            "tag_ablation": {
                "max_weight": [-2.5, -3.0],
                "min_weight": [0.0, -1.0],
            },
            "graph_average": {
                "max_weight": [-2.5, -3.0],
                "min_weight": [0.0, -1.0],
            },
        }

        for method, expected_grid in expected.items():
            config = load_config(method, str(BLOCKING_CONFIGS_DIR / f"{method}.toml"))
            self.assertEqual(config["model"]["name"], expected_model)
            self.assertEqual(config["output"]["results_root"], "results/blocking")
            self.assertEqual(config["evaluation"]["backend"], "llamaguard")
            self.assertEqual(config["grid_search"]["max_weight"], expected_grid["max_weight"])
            self.assertEqual(config["grid_search"]["min_weight"], expected_grid["min_weight"])

    def test_standard_configs_default_to_main_results_root(self):
        config = load_config("basic_refusal")
        self.assertNotIn("output", config)

    def test_blocking_scripts_target_matching_config_files(self):
        expected = {
            "run_basic_refusal.sh": ("basic_refusal", "configs/blocking/basic_refusal.toml"),
            "run_topic_ablation.sh": ("topic_ablation", "configs/blocking/topic_ablation.toml"),
            "run_tag_ablation.sh": ("tag_ablation", "configs/blocking/tag_ablation.toml"),
            "run_graph_average.sh": ("graph_average", "configs/blocking/graph_average.toml"),
        }

        for script_name, (method, config_path) in expected.items():
            script_text = (BLOCKING_SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
            self.assertIn(f"--method {method}", script_text)
            self.assertIn(f"--config {config_path}", script_text)


if __name__ == "__main__":
    unittest.main()
