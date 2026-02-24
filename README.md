# Automated Discovery of Category-Specific Refusal Axes in LLMs via Knowledge-Graph Retrieval

This repository contains experiments on refusal directions and abliteration for steering model behavior.

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
LLM_editing/
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
from visualization.plots import create_harmfulness_heatmap, create_distribution_plots
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
