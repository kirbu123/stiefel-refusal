"""
LLM Editing Interactive CLI -- main entry point.

Usage:
    python -m cli
"""

import os
import sys
import warnings
from pathlib import Path

import torch
import transformers
from questionary import Choice
from rich.traceback import install

from .config_loader import (
    METHODS,
    DEFAULT_CONFIG_FILES,
    load_config,
    apply_config_to_env,
    print_config_summary,
)
from .runner import run_method, find_latest_results
from .post_processing import post_process_menu
from .ui import (
    detect_gpu,
    print,
    print_banner,
    print_memory_usage,
    prompt_confirm,
    prompt_select,
    prompt_text,
)


def run():
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print_banner()
    detect_gpu()
    print()

    while True:
        model_name = prompt_text(
            "Enter HuggingFace model name or local path",
            default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        )
        if not model_name:
            print("[red]Model name is required.[/]")
            continue
        break

    print(f"\nSelected model: [bold]{model_name}[/]")

    while True:
        print()
        method_choices = [
            Choice(title=display_name, value=method_id)
            for method_id, display_name in METHODS.items()
        ]
        method_choices.append(Choice(title="Exit", value="exit"))

        method = prompt_select("Select an editing method:", method_choices)

        if method == "exit" or method is None:
            print("\n[cyan]Goodbye![/]")
            return

        default_config = DEFAULT_CONFIG_FILES[method]
        print(f"\nDefault config: [bold]{default_config}[/]")

        use_custom = prompt_confirm(
            "Use default configuration?",
            default=True,
        )

        config_path = None
        if not use_custom:
            config_path = prompt_text(
                "Enter path to custom config file:",
                default=str(default_config),
            )

        try:
            config = load_config(method, config_path)
        except FileNotFoundError as e:
            print(f"[red]{e}[/]")
            continue

        config.setdefault("model", {})["name"] = model_name
        print_config_summary(config, model_name)

        proceed = prompt_confirm("Proceed with these settings?", default=True)
        if not proceed:
            continue

        try:
            results_dir = run_method(method, config, model_name)
        except KeyboardInterrupt:
            print("\n[yellow]Method execution interrupted by user.[/]")
            results_dir = Path(__file__).parent.parent / "results" / method
        except Exception as e:
            print(f"\n[red]Error during method execution: {e}[/]")
            import traceback
            traceback.print_exc()
            results_dir = Path(__file__).parent.parent / "results" / method

        results = find_latest_results(results_dir)

        if results:
            post_process_menu(model_name, method, results_dir, results)
        else:
            print("[yellow]No result files found. Post-processing unavailable.[/]")

        again = prompt_confirm("\nRun another method?", default=False)
        if not again:
            print("\n[cyan]Goodbye![/]")
            return


def main():
    install()

    try:
        run()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise


if __name__ == "__main__":
    main()
