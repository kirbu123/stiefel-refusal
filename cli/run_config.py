"""
Non-interactive config runner.

Usage:
    python -m cli.run_config --method basic_refusal --config configs/basic_refusal.toml
"""

from __future__ import annotations

import argparse

from .config_loader import METHODS, load_config, print_config_summary
from .runner import run_method

try:
    from .ui import print
except ModuleNotFoundError:
    from builtins import print


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a baseline method from a TOML config.")
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Baseline method identifier.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name or local path override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.method, args.config)
    model_name = args.model or config.get("model", {}).get("name")
    if not model_name:
        parser.error("Model name is required via --model or [model].name in the config.")

    config.setdefault("model", {})["name"] = model_name
    print_config_summary(config, model_name)

    results_dir = run_method(args.method, config, model_name)
    print()
    print(f"[bold green]Results directory:[/] {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
