import torch
import random
import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.select_direction import select_rdo_direction, select_cone_basis, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def _repo_dotenv():
    return Path(__file__).resolve().parent.parent.parent / ".env"


def parse_arguments():
    """Parse arguments for local TensorBoard run directory (RDO checkpoints + hparams)."""
    load_dotenv(_repo_dotenv(), override=True)
    parser = argparse.ArgumentParser(
        description="Evaluate RDO direction from a local TensorBoard run dir (rdo.py output)."
    )
    parser.add_argument(
        "--tb_run_dir",
        type=str,
        required=True,
        help="Path to TensorBoard run directory (contains hparams.json and checkpoints/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha from hparams.json (if missing or not a number)",
    )
    parser.add_argument(
        "--add_layer",
        type=int,
        default=None,
        help="Override add_layer from hparams.json",
    )
    return parser.parse_args()


def load_tb_run(tb_run_dir: Path) -> Tuple[Dict[str, Any], Path, torch.Tensor]:
    """Load hparams.json and lowest_loss_vector.pt from a local rdo.py TensorBoard run."""
    run_dir = tb_run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    hparams_path = run_dir / "hparams.json"
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Missing hparams.json: {hparams_path}")

    ckpt = run_dir / "checkpoints" / "lowest_loss_vector.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    directions = torch.load(ckpt, map_location="cpu")
    return hparams, run_dir, directions


def _resolve_alpha(
    hparams: Dict[str, Any],
    directions: torch.Tensor,
    override: Optional[float],
) -> float:
    if override is not None:
        return float(override)
    a = hparams.get("alpha")
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return float(a)
    if isinstance(a, str):
        try:
            return float(a)
        except ValueError:
            pass
    # rdo.py scales with ||v||; match that when hparams stores a tensor string
    return float(directions.squeeze().float().norm().item())


def _kl_threshold_from_hparams(hparams: Dict[str, Any], run_dir: Path) -> float:
    """Match previous wandb logic: tight KL filter when using retain loss and not a kl_ablation group."""
    use_retain = bool(hparams.get("use_retain_loss", hparams.get("retain_lambda", 0) > 0))
    folder = run_dir.name
    if use_retain and "kl_ablation" not in folder:
        return 0.1
    return 9999.0


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset_split(harmtype="harmful", split="train", instructions_only=True)
    harmless_train = load_dataset_split(harmtype="harmless", split="train", instructions_only=True)[: len(harmful_train)]
    harmful_val = load_dataset_split(harmtype="harmful", split="val", instructions_only=True)
    harmless_val = load_dataset_split(harmtype="harmless", split="val", instructions_only=True)[: len(harmful_val)]
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("Filtering train dataset")
        print(f"Number of harmful examples: {len(harmful_train)}")
        print(f"Number of harmless examples: {len(harmless_train)}")
        harmful_train_scores = get_refusal_scores(
            model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks
        )

        harmless_train_scores = get_refusal_scores(
            model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks
        )
        print(len([score for score in harmful_train_scores.tolist() if score > 0]))
        print(len([score for score in harmless_train_scores.tolist() if score < 0]))
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)[: len(harmful_train)]
        print(f"Filtered {len(harmful_train)} harmful examples and {len(harmless_train)} harmless examples")

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(
            model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks
        )
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def select_and_save_direction(
    cfg,
    model_base,
    harmful_val,
    harmless_val,
    candidate_directions,
    add_layer,
    run_dir: Path,
    hparams: Dict[str, Any],
):
    """Select and save the direction."""
    sel_dir = run_dir / "select_direction"
    sel_dir.mkdir(parents=True, exist_ok=True)

    is_subspace = len(candidate_directions.shape) > 2
    if is_subspace:
        candidate_idx, direction = select_cone_basis(
            model_base,
            harmful_val,
            harmless_val,
            candidate_directions,
            artifact_dir=str(sel_dir),
            add_layer=add_layer,
            n_samples=cfg.subspace_n_samples,
        )
    else:
        kl_threshold = _kl_threshold_from_hparams(hparams, run_dir)
        if candidate_directions.shape[0] > 1:
            candidate_idx, direction = select_rdo_direction(
                model_base,
                harmful_val,
                harmless_val,
                candidate_directions,
                artifact_dir=str(sel_dir),
                add_layer=add_layer,
                kl_threshold=kl_threshold,
            )
        else:
            candidate_idx = 0
            direction = candidate_directions[0]

    with open(run_dir / "direction_metadata.json", "w") as f:
        json.dump({"candidate_idx": candidate_idx}, f, indent=4)

    torch.save(direction, run_dir / "direction.pt")
    return candidate_idx, direction


def generate_and_save_completions_for_dataset(
    cfg,
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    intervention_label,
    dataset_name,
    run_dir: Path,
    dataset=None,
):
    """Generate and save completions for a dataset."""
    comp_dir = run_dir / "completions"
    comp_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.completions_batch_size,
    )

    out_path = comp_dir / f"{dataset_name}_{intervention_label}_completions.json"
    with open(out_path, "w") as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(
    cfg, intervention_label, dataset_name, eval_methodologies, run_dir: Path
):
    """Evaluate completions and save results for a dataset."""
    comp_dir = run_dir / "completions"
    comp_path = comp_dir / f"{dataset_name}_{intervention_label}_completions.json"
    with open(comp_path, "r") as f:
        completions = json.load(f)

    eval_path = comp_dir / f"{dataset_name}_{intervention_label}_evaluations.json"
    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=str(eval_path),
    )

    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=4)


def run_pipeline(tb_run_dir: str, alpha_override: Optional[float] = None, add_layer_override: Optional[int] = None):
    """Run the full evaluation pipeline using a local TensorBoard run directory."""
    load_dotenv(_repo_dotenv(), override=True)

    hparams, run_dir, directions = load_tb_run(Path(tb_run_dir))
    print(f"Loaded TB run: {run_dir}")

    model_path = hparams.get("model")
    if not model_path:
        raise ValueError('"model" not found in hparams.json')

    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    cone_dim = int(hparams.get("cone_dim", 1))
    is_subspace = cone_dim > 1

    directions = directions.squeeze().unsqueeze(0).cuda()
    print(f"Directions shape: {directions.shape}")

    add_layer = add_layer_override if add_layer_override is not None else int(hparams["add_layer"])
    alpha = _resolve_alpha(hparams, directions.cpu(), alpha_override)

    candidate_directions = alpha * directions
    print(f"candidate_directions shape: {candidate_directions.shape}")

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    harmless_test = random.sample(load_dataset_split(harmtype="harmless", split="test"), cfg.n_test)

    candidate_idx, direction = select_and_save_direction(
        cfg,
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        add_layer=add_layer,
        run_dir=run_dir,
        hparams=hparams,
    )
    if not is_subspace:
        eval_vectors = [{"vector": direction, "name_postfix": ""}]
    else:
        subspace_dim = len(direction)
        print(f"Subspace dim: {subspace_dim}")
        eval_vectors = []
        for i in range(subspace_dim):
            eval_vectors.append({"vector": direction[i], "name_postfix": f"_basis_{i+1}"})

    for e in eval_vectors:
        direction = e["vector"]
        name_postfix = e["name_postfix"]
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [
            (model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))
        ], []

        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                generate_and_save_completions_for_dataset(
                    cfg,
                    model_base,
                    ablation_fwd_pre_hooks,
                    ablation_fwd_hooks,
                    f"ablation{name_postfix}",
                    dataset_name,
                    run_dir,
                )
            if cfg.evaluate_actadd:
                generate_and_save_completions_for_dataset(
                    cfg,
                    model_base,
                    actadd_fwd_pre_hooks,
                    actadd_fwd_hooks,
                    f"actadd{name_postfix}",
                    dataset_name,
                    run_dir,
                )

        if cfg.evaluate_harmless:
            print("Evaluating on harmless instructions")
            actadd_refusal_pre_hooks, actadd_refusal_hooks = [
                (model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))
            ], []
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                actadd_refusal_pre_hooks,
                actadd_refusal_hooks,
                f"actadd{name_postfix}",
                "harmless",
                run_dir,
                dataset=harmless_test,
            )

    for e in eval_vectors:
        print("Evaluating completions")
        name_postfix = e["name_postfix"]
        model_base.del_model()
        torch.cuda.empty_cache()

        for dataset_name in cfg.evaluation_datasets:
            cfg.jailbreak_eval_methodologies = ("substring_matching",)

            if cfg.evaluate_ablation:
                evaluate_completions_and_save_results_for_dataset(
                    cfg, f"ablation{name_postfix}", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, run_dir=run_dir
                )
            if cfg.evaluate_actadd:
                evaluate_completions_and_save_results_for_dataset(
                    cfg, f"actadd{name_postfix}", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, run_dir=run_dir
                )

        if cfg.evaluate_harmless:
            evaluate_completions_and_save_results_for_dataset(
                cfg, f"actadd{name_postfix}", "harmless", eval_methodologies=cfg.refusal_eval_methodologies, run_dir=run_dir
            )

    print(f"Outputs written under: {run_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        tb_run_dir=args.tb_run_dir,
        alpha_override=args.alpha,
        add_layer_override=args.add_layer,
    )
