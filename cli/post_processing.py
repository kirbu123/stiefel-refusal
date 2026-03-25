"""
Post-processing menu: save model, upload to HuggingFace, validate,
generate plots, and chat -- adapted from heretic's post-optimization flow.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import ModelCard, ModelCardData
from questionary import Choice

from .ui import (
    empty_cache,
    print,
    print_memory_usage,
    prompt_confirm,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


PROJECT_ROOT = Path(__file__).parent.parent


def _load_model_for_post_processing(model_name: str):
    """Load a fresh heretic Model for post-processing."""
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        from heretic.config import Settings
        from heretic.model import Model

        settings = Settings(
            model=model_name,
            batch_size=16,
            max_response_length=2048,
            system_prompt="You are a helpful assistant.",
        )
        model = Model(settings)
        return model, settings
    finally:
        sys.argv = original_argv


def _apply_best_params(model, result: dict[str, Any]):
    """Re-apply the editing parameters from a result to the model."""
    from model_utils import apply_abliteration_with_hyperparams
    from refusal_directions import compute_refusal_direction
    from heretic.utils import load_prompts as heretic_load_prompts
    from heretic.config import DatasetSpecification

    experiment = result.get("experiment_config", result.get("experiment_info", {}))
    hyperparams = experiment.get("hyperparameters", experiment.get("abliteration_params", {}))

    if not hyperparams:
        print("[yellow]No hyperparameters found in result, skipping re-application.[/]")
        return

    n_layers = len(model.get_layers())

    max_weight = hyperparams.get("max_weight", 2.0)
    max_weight_position = hyperparams.get("max_weight_position", 0.7)
    min_weight = hyperparams.get("min_weight", 0.1)
    min_weight_distance = hyperparams.get("min_weight_distance", 0.3)

    good_prompts_spec = DatasetSpecification(
        dataset="mlabonne/harmless_alpaca",
        split="train[:400]",
        column="text",
    )
    good_prompts = heretic_load_prompts(good_prompts_spec)

    questions = result.get("questions", [])
    harmful_questions = result.get("harmful_questions", {})
    if not questions and isinstance(harmful_questions, dict):
        questions = harmful_questions.get("questions", [])

    if not questions:
        print("[yellow]No questions found in result for re-computing refusal direction.[/]")
        return

    print("  Computing refusal direction...")
    refusal_directions = compute_refusal_direction(model, questions[:50], good_prompts)

    print("  Applying abliteration...")
    apply_abliteration_with_hyperparams(
        model, refusal_directions,
        max_weight, max_weight_position,
        min_weight, min_weight_distance,
        n_layers,
    )


def save_model_locally(model, settings):
    """Save the model to a local directory."""
    save_directory = prompt_path("Path to save the model:")
    if not save_directory:
        return

    print(f"Saving merged model to [bold]{save_directory}[/]...")
    try:
        merged_model = model.get_merged_model()
        merged_model.save_pretrained(save_directory)
        del merged_model
        empty_cache()
        model.tokenizer.save_pretrained(save_directory)
        print(f"[green]Model saved to [bold]{save_directory}[/][/]")
    except Exception as e:
        print(f"[red]Error saving model: {e}[/]")


def upload_to_huggingface(model, settings):
    """Upload the model to Hugging Face Hub."""
    token = huggingface_hub.get_token()
    if not token:
        token = prompt_password("Hugging Face access token:")
    if not token:
        return

    try:
        user = huggingface_hub.whoami(token)
        fullname = user.get("fullname", user.get("name", "unknown user"))
        email = user.get("email", "no email found")
        print(f"Logged in as [bold]{fullname} ({email})[/]")
    except Exception as e:
        print(f"[red]Authentication failed: {e}[/]")
        return

    default_repo = f"{user['name']}/{Path(settings.model).name}-edited"
    repo_id = prompt_text("Name of repository:", default=default_repo)
    if not repo_id:
        return

    visibility = prompt_select(
        "Should the repository be public or private?",
        ["Public", "Private"],
    )
    private = visibility == "Private"

    print("Uploading merged model...")
    try:
        merged_model = model.get_merged_model()
        merged_model.push_to_hub(repo_id, private=private, token=token)
        del merged_model
        empty_cache()
        model.tokenizer.push_to_hub(repo_id, private=private, token=token)

        model_path = Path(settings.model)
        if model_path.exists():
            card_path = model_path / huggingface_hub.constants.REPOCARD_NAME
            if card_path.exists():
                card = ModelCard.load(card_path)
            else:
                card = None
        else:
            try:
                card = ModelCard.load(settings.model)
            except Exception:
                card = None

        if card is not None:
            if card.data is None:
                card.data = ModelCardData()
            if card.data.tags is None:
                card.data.tags = []
            card.data.tags.extend(["abliterated", "edited", "llm-editing"])
            card.push_to_hub(repo_id, token=token)

        print(f"[green]Model uploaded to [bold]{repo_id}[/][/]")
    except Exception as e:
        print(f"[red]Error uploading model: {e}[/]")


def validate_model(model, result: dict[str, Any]):
    """Run validation on the modified model."""
    print()
    print("[bold]Running validation...[/]")

    questions = result.get("questions", [])
    harmful_questions = result.get("harmful_questions", {})
    if not questions and isinstance(harmful_questions, dict):
        questions = harmful_questions.get("questions", [])

    if not questions:
        print("[yellow]No questions found in result for validation.[/]")
        return

    eval_questions = questions[:20]
    print(f"Generating responses for {len(eval_questions)} questions...")

    from data_utils import extract_response_after_think

    responses_raw = model.get_responses_batched(eval_questions)
    responses = [extract_response_after_think(r) for r in responses_raw]

    print()
    print("[bold]Sample responses:[/]")
    for i, (q, r) in enumerate(zip(eval_questions[:5], responses[:5]), 1):
        print(f"\n  [bold]Q{i}:[/] {q[:100]}...")
        resp_preview = r[:200] + "..." if len(r) > 200 else r
        print(f"  [bold]A{i}:[/] {resp_preview}")

    try:
        from evaluate.judges import evaluate_harmfulness
        print()
        print("[bold]Evaluating harmfulness...[/]")
        scores = []
        for q, r in zip(eval_questions, responses):
            if not r:
                continue
            try:
                score_data = evaluate_harmfulness(q, r, classifier_categories=[])
                score = score_data.get("score")
                if score is not None:
                    scores.append(int(score))
            except Exception:
                pass

        if scores:
            print(f"  Mean harmfulness score: [bold]{np.mean(scores):.2f}[/]")
            print(f"  Median: [bold]{np.median(scores):.2f}[/]")
            print(f"  Evaluated: {len(scores)} / {len(eval_questions)} responses")
    except ImportError:
        print("[yellow]Evaluation module not available. Skipping harmfulness scoring.[/]")

    print_memory_usage()


def generate_plots(results_dir: Path, result: dict[str, Any], method: str):
    """Generate plots from experiment results using the visualization module."""
    print()
    print("[bold]Generating plots...[/]")

    try:
        from visualization.plots import (
            plot_harmfulness_heatmap,
            plot_locality_heatmap,
            plot_harmfulness_distribution,
            plot_locality_distribution,
            generate_plot_filename,
        )
    except ImportError:
        print("[red]visualization.plots module not available.[/]")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiment = result.get("experiment_config", result.get("experiment_info", {}))
    category = experiment.get("category", "Unknown")

    harmful = result.get("harmful_questions", {})
    original_evals = harmful.get("original_evaluations", [])
    modified_evals = harmful.get("modified_evaluations", [])

    original_scores = [
        e.get("score", 0) for e in original_evals if e.get("score") is not None
    ] if original_evals else []
    modified_scores = [
        e.get("score", 0) for e in modified_evals if e.get("score") is not None
    ] if modified_evals else result.get("final_scores", [])

    if not original_scores:
        original_scores = result.get("original_scores", [])
    if not modified_scores:
        modified_scores = result.get("modified_scores", [])

    hyperparams = experiment.get("hyperparameters", experiment.get("abliteration_params", {}))

    plot_choices = prompt_select(
        "Which plots would you like to generate?",
        [
            Choice(title="All plots", value="all"),
            Choice(title="Harmfulness distribution", value="harm_dist"),
            Choice(title="Locality distribution", value="loc_dist"),
            Choice(title="Harmfulness heatmap", value="harm_heat"),
            Choice(title="Locality heatmap", value="loc_heat"),
            Choice(title="Cancel", value="cancel"),
        ],
    )

    if plot_choices == "cancel" or plot_choices is None:
        return

    evaluator_name = method

    if plot_choices in ("all", "harm_dist"):
        if original_scores or modified_scores:
            fname = generate_plot_filename("harmfulness_distribution", category, evaluator_name, ext=".png")
            plot_harmfulness_distribution(
                original_scores, modified_scores,
                plots_dir / fname,
                evaluator_name=evaluator_name, method_name=method,
                category_name=category, hyperparams=hyperparams,
            )
        else:
            print("[yellow]No score data available for harmfulness distribution.[/]")

    if plot_choices in ("all", "loc_dist"):
        harmless = result.get("harmless_questions", {})
        locality_scores = harmless.get("locality_scores", [])
        if locality_scores:
            orig_loc = [s["original_score"] for s in locality_scores]
            mod_loc = [s["modified_score"] for s in locality_scores]
            diffs = [s["difference"] for s in locality_scores if s.get("difference") is not None]
            if diffs:
                fname = generate_plot_filename("locality_distribution", category, evaluator_name, ext=".png")
                plot_locality_distribution(
                    orig_loc, mod_loc, diffs,
                    plots_dir / fname,
                    evaluator_name=evaluator_name, method_name=method,
                    category_name=category, hyperparams=hyperparams,
                )
        else:
            print("[yellow]No locality data available for distribution plot.[/]")

    if plot_choices in ("all", "harm_heat"):
        if modified_scores and hyperparams:
            heatmap_data = [{
                "max_weight": hyperparams.get("max_weight", 0),
                "min_weight": hyperparams.get("min_weight", 0),
                "mean_modified_score": float(np.mean(modified_scores)),
            }]
            fname = generate_plot_filename("harmfulness_heatmap", category, evaluator_name)
            plot_harmfulness_heatmap(
                heatmap_data, plots_dir / fname,
                evaluator_name=evaluator_name, method_name=method,
                category_name=category,
            )
        else:
            print("[yellow]No data available for harmfulness heatmap.[/]")

    if plot_choices in ("all", "loc_heat"):
        harmless = result.get("harmless_questions", {})
        avg_loc = harmless.get("average_locality_change")
        if avg_loc is not None and hyperparams:
            heatmap_data = [{
                "max_weight": hyperparams.get("max_weight", 0),
                "min_weight": hyperparams.get("min_weight", 0),
                "average_locality_change": avg_loc,
            }]
            fname = generate_plot_filename("locality_heatmap", category, evaluator_name)
            plot_locality_heatmap(
                heatmap_data, plots_dir / fname,
                evaluator_name=evaluator_name, method_name=method,
                category_name=category,
            )
        else:
            print("[yellow]No data available for locality heatmap.[/]")

    print(f"[green]Plots saved to: {plots_dir}[/]")


def chat_with_model(model):
    """Interactive chat loop with the modified model."""
    print()
    print("[cyan]Press Ctrl+C at any time to return to the menu.[/]")

    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    while True:
        try:
            message = prompt_text("User:", qmark=">", unsafe=True)
            if not message:
                break
            chat.append({"role": "user", "content": message})

            print("[bold]Assistant:[/] ", end="")
            response = model.stream_chat_response(chat)
            chat.append({"role": "assistant", "content": response})
        except (KeyboardInterrupt, EOFError):
            break


def post_process_menu(
    model_name: str,
    method: str,
    results_dir: Path,
    results: list[dict[str, Any]],
):
    """
    Main post-processing menu loop.
    Lets user select a result and perform actions on the modified model.
    """
    from .runner import summarize_results

    summaries = summarize_results(results)

    if not summaries:
        print("[yellow]No experiment results found. Skipping post-processing.[/]")
        return

    print()
    print("[bold green]Experiment completed![/]")
    print()
    print("The following experiment results are available:")

    choices = []
    for i, s in enumerate(summaries):
        label = f"[{i + 1}] "
        if s["category"] != "N/A":
            label += f"{s['category']} | "
        if s["mean_score"] is not None:
            label += f"Mean score: {s['mean_score']:.2f} | "
        if s["avg_locality_change"] is not None:
            label += f"Locality change: {s['avg_locality_change']:+.2f} | "
        if s["mmlu_original_accuracy"] is not None and s["mmlu_modified_accuracy"] is not None:
            label += (
                f"MMLU: {s['mmlu_original_accuracy']:.2f} -> "
                f"{s['mmlu_modified_accuracy']:.2f} | "
            )
        if s["param_key"]:
            label += f"Params: {s['param_key']}"
        elif s["hyperparams"]:
            hp_str = ", ".join(f"{k}={v}" for k, v in list(s["hyperparams"].items())[:4])
            label += f"Params: {hp_str}"
        choices.append(Choice(title=label.rstrip(" |"), value=i))

    choices.append(Choice(title="Exit", value="exit"))

    while True:
        print()
        selection = prompt_select("Select a result to work with:", choices)

        if selection == "exit" or selection is None:
            return

        selected_result = summaries[selection]["raw"]

        should_load = prompt_confirm(
            "Load the model and re-apply editing for post-processing?",
            default=True,
        )

        model = None
        settings = None
        if should_load:
            print()
            print(f"Loading model [bold]{model_name}[/]...")
            model, settings = _load_model_for_post_processing(model_name)
            print_memory_usage()

            print()
            print("Re-applying editing parameters...")
            _apply_best_params(model, selected_result)
            print("[green]Model editing re-applied.[/]")
            print_memory_usage()

        while True:
            print()
            action_choices = []
            if model is not None:
                action_choices.extend([
                    "Save the model to a local folder",
                    "Upload the model to Hugging Face",
                    "Validate the model",
                    "Chat with the model",
                ])
            action_choices.extend([
                "Generate plots from results",
                "Return to result selection",
            ])

            action = prompt_select(
                "What do you want to do?",
                action_choices,
            )

            if action is None or action == "Return to result selection":
                if model is not None:
                    del model
                    empty_cache()
                    model = None
                break

            try:
                if action == "Save the model to a local folder":
                    save_model_locally(model, settings)
                elif action == "Upload the model to Hugging Face":
                    upload_to_huggingface(model, settings)
                elif action == "Validate the model":
                    validate_model(model, selected_result)
                elif action == "Generate plots from results":
                    generate_plots(results_dir, selected_result, method)
                elif action == "Chat with the model":
                    chat_with_model(model)
            except Exception as e:
                print(f"[red]Error: {e}[/]")
