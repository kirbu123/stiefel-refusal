# Automated Discovery of Category-Specific Refusal Axes in LLMs via Knowledge-Graph Retrieval

This repository contains experiments on refusal directions and abliteration for steering model behavior.

## Requirements

```bash
pip install -r requirements.txt
```

## Interactive CLI

The project includes an interactive CLI that guides you through model selection, editing, and post-processing -- no need to manually edit shell scripts or environment variables.

### Quick Start

```bash
python -m cli
```

### CLI Flow

1. **Enter model name** -- HuggingFace model ID (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) or a local path.
2. **Select editing method** -- one of six available methods:
   - **Basic Refusal** -- uses category questions as harmful prompts, uniform abliteration shift
   - **Topic Ablation** -- uses the category name as the harmful prompt, grid search over hyperparameters
   - **Tag Ablation** -- uses tags from `tag_filtered_questions.json`, one refusal direction per tag
   - **Graph Average** -- averages refusal directions from all graph tags, grid search
   - **Graph GRPO (IS)** -- trains learnable direction weights with importance sampling
   - **Graph GRPO (old)** -- trains direction weights with noise-based GRPO
3. **Configuration** -- default parameters are loaded from TOML config files in `configs/`. You can use the defaults or provide a custom config path.
4. **Run** -- the selected method executes with the configured parameters.
5. **Post-processing menu** -- after the experiment completes, choose from:
   - **Save model locally** -- merge and save the edited model to a local directory
   - **Upload to Hugging Face** -- push the model to a HuggingFace repository
   - **Validate** -- generate responses and evaluate harmfulness scores
   - **Generate plots** -- harmfulness/locality heatmaps and distribution histograms
   - **Chat** -- interactive chat with the edited model

Every run can also compute MMLU accuracy before and after editing. For grid-search methods, the original-model MMLU baseline is computed once and cached, while post-edit MMLU is stored per saved configuration.

For non-interactive runs from a TOML file, use:

```bash
python -m cli.run_config --method graph_average --config configs/graph_average.toml
```

### Configuration Files

Each method has a default TOML config in `configs/`:

```
configs/
├── basic_refusal.toml
├── topic_ablation.toml
├── tag_ablation.toml
├── graph_average.toml
├── graph_grpo.toml
└── graph_grpo_old.toml
```

Blocking-only grid-search configs live under `configs/blocking/` and write to a separate results root:

```
configs/blocking/
├── basic_refusal.toml
├── topic_ablation.toml
├── tag_ablation.toml
└── graph_average.toml
```

`graph_grpo` and `graph_grpo_old` are intentionally excluded from this blocking pack because they need a reward/objective change, not just sign-inverted abliteration weights.

Example (`configs/graph_average.toml`):

```toml
[model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
batch_size = 16
max_response_length = 2048

[output]
results_root = "results"

[data]
graph_file = "graph_build/ph_wordnet_graph_25neighbours_actions_terms.txt"

[grid_search]
max_weight = [2.5, 3.0]
max_weight_position = [0.7]
min_weight = [0.0, 1.0]
min_weight_distance = [0.3]

[evaluation]
backend = "llamaguard"
evaluate_locality = true

[mmlu]
enabled = true
dataset = "cais/mmlu"
subset = "all"
split = "test"
mode = "zero_shot"
answer_mode = "generate"
n_shots = 5
sample_size = 100
sample_seed = 42
max_new_tokens = 4
store_predictions = false
```

The model name provided interactively always overrides the value in the config file.

### MMLU Configuration

Each config file can include an optional `[mmlu]` section:

Each config file can also include an optional `[output]` section:

- `results_root` -- root directory for experiment outputs; defaults to `results`

- `enabled` -- turn MMLU evaluation on or off
- `dataset` -- Hugging Face dataset ID, default `cais/mmlu`
- `subset` -- `"all"` for all subjects, or a single subject config
- `split` -- evaluation split, default `test`
- `mode` -- `zero_shot` or `few_shot`
- `answer_mode` -- `generate` to parse the generated `A/B/C/D`, or `logits` to pick the answer by choice log-probabilities
- `n_shots` -- number of demonstrations for `few_shot`
- `sample_size` -- integer sample limit for fast runs, or `null` for full MMLU
- `sample_seed` -- deterministic sampling seed
- `max_new_tokens` -- generation budget for answer letters, default `32` to give reasoning models enough room to reach the final choice
- `store_predictions` -- when `true`, store per-example predictions in `results/<method>/mmlu/`

Notes:

- `sample_size = 100` is the default fast path for regular experiments.
- Set `sample_size = null` to evaluate on the full selected MMLU split.
- `zero_shot` asks the model to answer one question with `A/B/C/D`.
- `few_shot` prepends demonstrations from the corresponding MMLU `dev` split.
- `answer_mode = "logits"` scores `A/B/C/D` directly from the model probabilities instead of parsing generated text.
- For reasoning models that emit `<think>...</think>`, MMLU parsing strips the thinking block before extracting the final answer letter.

### Academic Benchmarks Configuration

`graph_grpo` also supports a dedicated `[academic_benchmarks]` block for `TinyHellaSwag`, `ARC`, `WinoGrande`, `GSM8K`, and `TruthfulQA`.

Example:

```toml
[academic_benchmarks]
enabled = ["tinyhellaswag", "arc", "winogrande", "gsm8k", "truthfulqa"]
sample_seed = 42
store_predictions = true

[academic_benchmarks.tinyhellaswag]
dataset = "tinyBenchmarks/tinyHellaswag"
split = "validation"
sample_size = 100

[academic_benchmarks.arc]
dataset = "allenai/ai2_arc"
split = "validation"
sample_size = 100

[academic_benchmarks.winogrande]
dataset = "allenai/winogrande"
subset = "winogrande_xl"
split = "validation"
sample_size = 100

[academic_benchmarks.gsm8k]
dataset = "openai/gsm8k"
subset = "main"
split = "test"
sample_size = 100
max_new_tokens = 512

[academic_benchmarks.truthfulqa]
dataset = "truthfulqa/truthful_qa"
subset = "multiple_choice"
split = "validation"
sample_size = 100
```

Global keys:

- `enabled` -- list of academic benchmarks to run
- `sample_seed` -- deterministic sampling seed shared by all enabled academic benchmarks
- `store_predictions` -- when `true`, store per-example academic benchmark details in `results/graph_grpo/academic_benchmarks/details/`

Per-benchmark keys:

- `dataset` -- Hugging Face dataset ID used at runtime
- `subset` -- optional dataset subset/config name (`winogrande_xl`, `main`, `multiple_choice`, etc.)
- `split` -- dataset split used for evaluation
- `sample_size` -- integer sample limit for the default fast path, or `null` for the full selected split
- `max_new_tokens` -- generation budget for `gsm8k`

Defaults and notes:

- `sample_size = 100` is the default fast path for all five academic benchmarks.
- Set any benchmark `sample_size = null` to run the full selected split.
- `TinyHellaSwag` uses `IRT++` as the primary reported metric and stores the corresponding field as `irt_plus_plus`.
- `TinyHellaSwag` requires the optional `tinyBenchmarks` Python package. Install it explicitly:

```bash
pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

If `tinyhellaswag` is enabled but `tinyBenchmarks` is not installed, the run fails with a clear dependency error instead of silently skipping the benchmark.

### Academic Evaluation Procedure

`graph_grpo` evaluates academic benchmarks in exactly two places:

1. `clean model` -- one cached run before optimization starts
2. `best-value edited model` -- one final run after the best coefficients are applied

Important details:

- Academic benchmarks are not evaluated on every GRPO epoch.
- Academic benchmarks are not evaluated on every Optuna trial.
- The clean-model academic baseline is cached by `model_name + normalized benchmark config`.
- Final edited-model details are stored under `results/graph_grpo/academic_benchmarks/details/`.
- `mmlu` stays as a separate top-level block in `answers_*.json`; the new academic results live under `academic_benchmarks`.

### Benchmarks, Sources, and Setups

| Benchmark | Runtime dataset source | Original paper / official benchmark source | Setup source | Repo default split / variant | Metric(s) | Repo-specific notes |
|---|---|---|---|---|---|---|
| TinyHellaSwag | [`tinyBenchmarks/tinyHellaswag`](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag) | [`tinyBenchmarks`](https://huggingface.co/papers/2402.14992), original [`HellaSwag`](https://huggingface.co/papers/1905.07830) | `tinyBenchmarks` package / dataset card | `validation`, sampled to `100` by default | `irt_plus_plus` (primary), `accuracy` | Follows the tinyBenchmarks setup. In the Python package this score is exposed as `gpirt`; this repo stores it as `irt_plus_plus`. |
| ARC | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | [`ARC`](https://huggingface.co/papers/1803.05457) | Dataset card + original benchmark split names | `validation` on both `ARC-Easy` and `ARC-Challenge` | `ARC-Easy accuracy`, `ARC-Challenge accuracy`, `macro_accuracy` | The repo reports both variants separately and also logs the simple macro-average across them. |
| WinoGrande | [`allenai/winogrande`](https://huggingface.co/datasets/allenai/winogrande) | [`WinoGrande`](https://huggingface.co/papers/1907.10641) | Dataset card | `winogrande_xl` / `validation` | `accuracy` | The repo scores the two candidate sentence completions via continuation log-probabilities. |
| GSM8K | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | [`Training Verifiers to Solve Math Word Problems`](https://huggingface.co/papers/2110.14168) | Dataset card / original benchmark answer format | `main` / `test` | `exact_match` | The repo evaluates generated answers, first parsing `#### answer`, then falling back to the last numeric value after stripping `<think>...</think>`. |
| TruthfulQA | [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [`TruthfulQA`](https://huggingface.co/papers/2109.07958) | Official repo [`sylinrl/TruthfulQA`](https://github.com/sylinrl/TruthfulQA) and legacy HF reference [`eitanturok/truthful_qa`](https://huggingface.co/datasets/eitanturok/truthful_qa) | `multiple_choice` / `validation` | `MC1` (primary), `MC2` | The repo intentionally keeps the historical `MC1 + MC2` reporting for comparability, even though the official repo recommends the newer 2-option multiple-choice setup as of January 2025. That choice is an implementation decision for this repo, not a claim that the newer setup is invalid. |

### Result Format

When academic benchmarks are enabled, `answers_*.json` includes a top-level `academic_benchmarks` block. Each benchmark stores:

- `original` -- clean-model summary for that benchmark
- `modified` -- final edited-model summary for that benchmark
- `delta_primary_metric` -- `modified.primary_metric_value - original.primary_metric_value`
- `config` -- the exact normalized config snapshot used for that benchmark
- `details_file` -- optional JSON file with the full clean/modified per-example results

Task-specific summary fields:

- `tinyhellaswag` -- `irt_plus_plus`, `accuracy`
- `arc` -- `macro_accuracy`, plus `by_variant.ARC-Easy` and `by_variant.ARC-Challenge`
- `winogrande` -- `accuracy`
- `gsm8k` -- `exact_match`
- `truthfulqa` -- `mc1`, `mc2`

## Project Structure

```
LLM_editing/
├── cli/                    # Interactive CLI (python -m cli)
├── configs/                # TOML config files for each editing method
├── baselines/              # Experiment methods (each runnable independently)
├── evaluate/               # Evaluation functions and batch helpers
├── visualization/          # Plotting utilities (heatmaps, distributions)
├── scripts/                # Shell launch scripts (set CUDA_VISIBLE_DEVICES)
├── config.py               # Project-wide configuration
├── data_utils.py           # Data loading and preprocessing helpers
├── model_utils.py          # Model operations (abliteration, learnable weights)
├── refusal_directions.py   # Computing, saving, and loading refusal directions
├── training.py             # GRPO training epoch
├── dataset/                # Benchmark datasets (raw, processed, splits)
├── graph_build/            # Knowledge-graph construction from Wikidata
├── results/                # Experiment outputs (created at runtime)
├── requirements.txt
└── README.md
```

### `baselines/`

Each file is a self-contained experiment script. Launch via shell scripts (see `scripts/` below) or directly:

```bash
python -m baselines.graph_average
python -m baselines.topic_ablation
python -m baselines.tag_ablation
python -m baselines.graph_grpo
python -m baselines.graph_grpo_old
```

| File | Description |
|---|---|
| `graph_average.py` | Grid search over abliteration hyperparameters using an averaged refusal direction from all graph tags |
| `topic_ablation.py` | Per-category ablation: computes a refusal direction from the category name itself |
| `tag_ablation.py` | Per-tag ablation: loads tags from `tag_filtered_questions.json` and computes a separate direction for each |
| `graph_grpo_old.py` | GRPO training of learnable coefficients for a weighted sum of refusal directions |
| `hyperparams.py` | Shared hyperparameter grid used by the grid-search baselines |

### `evaluate/`

| File | Description |
|---|---|
| `judges.py` | Core scoring functions: `evaluate_harmfulness_with_local_judge()` (LLM-as-a-Judge, 0-4 scale) and `classify_question_category_with_local_llm()` |
| `metrics.py` | Batch helpers: `evaluate_responses()` scores a list of question-response pairs; `evaluate_locality()` measures harmfulness change on harmless questions before/after modification |
| `academic_benchmarks.py` | Shared academic benchmark helpers for `TinyHellaSwag`, `ARC`, `WinoGrande`, `GSM8K`, and `TruthfulQA`, including caching and result serialization |
| `model_scoring.py` | Shared chat-template generation and continuation log-prob scoring helpers used by `mmlu.py` and `academic_benchmarks.py` |
| `mmlu.py` | Shared MMLU evaluation helpers: prompt building, deterministic sampling, caching, and result serialization |
| `runner.py` | Standalone script to evaluate saved ablation results from JSON files |

### `visualization/`

| File | Description |
|---|---|
| `plots.py` | Shared plotting functions: `create_harmfulness_heatmap()`, `create_locality_heatmap()`, `create_distribution_plots()` |
| `draw_graphics.py` | CLI tool to build plots from answer JSON files. Auto-detects baseline type from the file path |

### Shared Modules (top-level)

| File | Description |
|---|---|
| `config.py` | Model name, categories, dataset specs, GRPO/abliteration hyperparameters, API URLs, paths |
| `data_utils.py` | `load_all_datasets_with_categories()`, `extract_response_after_think()` |
| `model_utils.py` | `LearnableDirectionWeights` (nn.Module), `apply_abliteration_with_hyperparams()` |
| `refusal_directions.py` | `compute_refusal_direction()`, `save_refusal_directions()`, `load_refusal_directions()` |
| `training.py` | `train_grpo_epoch()` -- one GRPO training epoch with noise variants, advantage computation, and policy gradient update |

### `dataset/`

- **raw/** -- original benchmark data (AdvBench, HarmBench, JailbreakBench, MaliciousInstruct, StrongReject, TDC2023)
- **processed/** -- datasets converted to a unified JSON format
- **splits/** -- train/val/test splits (harmful vs harmless)
- `load_dataset.py` -- dataset loading functions
- `categorize_datasets.py` -- assigns jailbreakbench categories to all datasets

### `graph_build/`

Knowledge-graph construction and querying:

- `sparql_util.py` -- Wikidata SPARQL queries and rate limiting
- `build_graph_from_string.py` -- builds a graph from a text description
- `extract_paths_from_graph.py`, `extract_subgraph.py` -- path and subgraph extraction

### `scripts/`

Shell scripts to launch each baseline with `CUDA_VISIBLE_DEVICES` configured:

```bash
bash scripts/run_graph_average.sh
bash scripts/run_topic_ablation.sh
bash scripts/run_tag_ablation.sh
bash scripts/run_graph_grpo_old.sh
```

Blocking scripts are kept separately and write to `results/blocking/<method>/...`:

```bash
bash scripts/blocking/run_basic_refusal.sh
bash scripts/blocking/run_topic_ablation.sh
bash scripts/blocking/run_tag_ablation.sh
bash scripts/blocking/run_graph_average.sh
```

Edit the `CUDA_VISIBLE_DEVICES` line inside each script to select the target GPU.

### RDO Refusal Axis (`baselines/rdo_refusal.py`)

This section explains how to run the RDO training/editing baseline using the companion geometry pipeline.

1) Prepare `geometry-of-refusal` environment

- Add the `.env` file expected by `geometry-of-refusal` (inside `LLM-Attack-Defense/geometry-of-refusal/`).

like this example:

```
SAVE_DIR="./../results/rdo_refusal"
DIM_DIR="dim"
WANDB_ENTITY="refusal-representations"
WANDB_PROJECT="refusal_directions"
```

2) Build the geometry artifacts (run the pipeline)

From:
`./geometry-of-refusal/refusal_direction/`

run:
```bash
python -m pipeline.run_pipeline --model_path <model name>
```

3) Train/edit with RDO

From the repo root for this baseline (`LLM-Attack-Defense/`), run:
```bash
./scripts/run_rdo_refusal.sh
```

What `./scripts/run_rdo_refusal.sh` does (script construction)

The script is a thin wrapper that:

- `cd "$(dirname "$0")/.."` to ensure it runs from `LLM-Attack-Defense/`
- sets core hyperparameters in bash variables:
  - `direction_mode` (default: `shtiefel_proj_rot`)
  - `lr` (default: `1e-5`)
  - `max_iters` (exported as `MAX_ITERS`)
  - optional `result_path` (leave empty to train)
- forces a default GPU via `export CUDA_VISIBLE_DEVICES=3` (you can override this when running)
- sets fast evaluation defaults via environment variables:
  - `MMLU_SAMPLE_SIZE`, `MMLU_MAX_NEW_TOKENS`, `MMLU_STORE_PREDICTIONS`
  - `LLAMAGUARD_MAX_NEW_TOKENS`
  - dataset caps: `RDO_MAX_HARMFUL_PER_CATEGORY`, `RDO_MAX_HARMLESS_TOTAL`
- sets `HF_TOKEN` inside the script (needed for gated models / LlamaGuard download if applicable)
- defines and executes a `cmd=(python -m baselines.rdo_refusal ...)` array including flags like:
  - `--train_direction`
  - `--direction_mode ${direction_mode}`
  - `--eval_llamaguard --eval_mmlu --mmlu_store_predictions`
  - `--freeze_order_layers`
  - `--init_mode "random"` (can be changed to `diag_permutation`)
  - `--retain_loss`
- if `result_path` is set, it appends `--result_path "${result_path}"` to reuse an existing run directory
- finally runs the command with `"${cmd[@]}"`

## Adding New Components

**New method:** create `baselines/my_method.py` and import shared modules:

```python
from config import MODEL_NAME, CATEGORIES, RESULTS_DIR, ...
from evaluate.metrics import evaluate_responses, evaluate_locality
from visualization.plots import plot_harmfulness_heatmap, plot_harmfulness_distribution
from model_utils import apply_abliteration_with_hyperparams
from refusal_directions import compute_refusal_direction
from data_utils import load_all_datasets_with_categories
```

**New benchmark:** add data files to `dataset/`, update `data_utils.py` loaders.

**New evaluation function:** add to `evaluate/judges.py` or create a new file in `evaluate/`.

## Results

Each baseline writes outputs to `<results_root>/<method_name>/` with the following structure:

```
results/
├── graph_average/
│   ├── answers/                          # One JSON per hyperparameter combination
│   ├── mmlu/                             # Optional per-example MMLU predictions
│   ├── harmfulness/
│   │   ├── distribution_plots/
│   │   └── heatmap_plots/
│   └── locality/
│       ├── distribution_plots/
│       └── heatmap_plots/
├── graph_grpo/
│   ├── answers/
│   ├── mmlu/                             # Optional per-example MMLU predictions
│   ├── academic_benchmarks/
│   │   ├── cache/                        # Cached clean-model academic evaluations
│   │   └── details/                      # Final edited-model per-benchmark details
│   ├── harmfulness/ ...
│   └── locality/ ...
├── graph_grpo_old/
│   ├── answers/
│   ├── harmfulness/ ...
│   └── locality/ ...
├── tag_ablation/
│   ├── answers/
│   ├── harmfulness/ ...
│   └── locality/ ...
└── topic_ablation/
    ├── answers/
    ├── harmfulness/ ...
    └── locality/ ...
```

For blocking configs, the same structure is created under `results/blocking/`.

Saved answer files can now include a top-level `mmlu` block with:

- `original` -- baseline accuracy before editing
- `modified` -- accuracy after editing
- `delta_accuracy` -- `modified - original`
- `config` -- the exact MMLU settings used
- `answer_comparison_preview` -- a compact before/after preview of how the model answered the same MMLU questions
- `details_file` -- optional JSON with per-example predictions

For `graph_grpo`, saved answer files can also include a top-level `academic_benchmarks` block. Each enabled benchmark stores:

- `original` -- clean-model benchmark summary
- `modified` -- final edited-model benchmark summary
- `delta_primary_metric` -- difference in the benchmark's primary scalar metric
- `config` -- exact benchmark config snapshot used for that run
- `details_file` -- optional JSON with the full clean/modified benchmark outputs

`graph_grpo` also writes academic cache/details files under:

```
results/graph_grpo/academic_benchmarks/
├── cache/                                # Clean-model cached academic evaluations
└── details/                              # Final edited-model per-benchmark details
```

## Verification Notes

The code changes are intended to be verified with lightweight unit tests and static checks only. Long-running editing methods are not required for validating the MMLU and academic benchmark integration.

## Credits

- [Heretic codebase](https://github.com/p-e-w/heretic/)
