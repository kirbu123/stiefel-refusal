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

Example (`configs/graph_average.toml`):

```toml
[model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
batch_size = 16
max_response_length = 2048

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

Edit the `CUDA_VISIBLE_DEVICES` line inside each script to select the target GPU.

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

Each baseline writes outputs to `results/<method_name>/` with the following structure:

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

Saved answer files can now include a top-level `mmlu` block with:

- `original` -- baseline accuracy before editing
- `modified` -- accuracy after editing
- `delta_accuracy` -- `modified - original`
- `config` -- the exact MMLU settings used
- `answer_comparison_preview` -- a compact before/after preview of how the model answered the same MMLU questions
- `details_file` -- optional JSON with per-example predictions

## Verification Notes

The code changes are intended to be verified with lightweight unit tests and static checks only. Long-running editing methods are not required for validating the MMLU integration.

## Credits

- [Heretic codebase](https://github.com/p-e-w/heretic/)
