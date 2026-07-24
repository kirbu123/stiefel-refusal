# Refusal Pipeline

This document covers the retained RDO workflow on `main`:

1. build a DIM refusal direction with `pipeline.run_pipeline`;
2. train/evaluate an RDO steering method with `baselines.rdo_refusal`;
3. launch individual or grid experiments;
4. optionally run paper Angular / Spherical inference-only steering baselines
   (no DIM / no RDO training);
5. generate `k_proj` and `num_opt_layers` ablation plots.

All commands assume the repository is located at:

```text
/home/user1/buka2004/LLM-Attack-Defense
```

## Security first

Do not commit Hugging Face access tokens. At the time this document was written,
`scripts/run_rdo_refusal.sh` contained a hardcoded `HF_TOKEN`. Revoke that token,
remove the hardcoded export, and provide credentials through your shell:

```bash
export HF_TOKEN="<your-token>"
```

You can also authenticate once with:

```bash
hf auth login
```

## Environment installation

Python 3.12 and a CUDA-capable PyTorch installation are the currently used
environment.

```bash
cd /home/user1/buka2004/LLM-Attack-Defense
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install einops \
  "git+https://github.com/dsbowen/strong_reject.git@main"
```

The root requirements currently use the working RDO versions, including:

```text
torch==2.5.1
transformers==4.57.6
tokenizers==0.22.2
accelerate==1.12.0
nnsight==0.3.7
wandb==0.25.0
```

Do not install `geometry-of-refusal/requirements.txt` over this environment
without reviewing its pins: it specifies older Transformers/Accelerate versions
that may not support the current Qwen3 and `dtype=` code paths.

Confirm the active interpreter:

```bash
which python
python -c "import torch, transformers, nnsight; print(torch.__version__, transformers.__version__, nnsight.__version__)"
```

## Environment configuration

Create `geometry-of-refusal/.env`:

```dotenv
HUGGINGFACE_CACHE_DIR="/path/to/huggingface-cache"
SAVE_DIR="./../results/rdo_refusal"
DIM_DIR="dim"
WANDB_ENTITY="refusal-representations"
WANDB_PROJECT="refusal_directions"
```

The relative `SAVE_DIR` above is resolved from `geometry-of-refusal` and writes
DIM artifacts to the repository-level `results/rdo_refusal/dim/` directory.

Optionally create a repository-root `.env` for settings used by RDO:

```dotenv
HUGGINGFACE_CACHE_DIR="/path/to/huggingface-cache"
HF_TOKEN="<your-token>"
```

Useful shell settings:

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled       # optional: disable online W&B logging
export HF_HOME=/path/to/hf-home # optional
```

## Required data and artifact flow

The geometry pipeline reads:

```text
geometry-of-refusal/data/saladbench_splits/
├── harmful_train.json
├── harmful_val.json
├── harmless_train.json
├── harmless_val.json
└── harmless_test.json
```

The RDO stage separately reads the repository-root splits:

```text
data/saladbench_splits/harmful_train.json
data/saladbench_splits/harmless_train.json
```

Do not remove either copy unless the corresponding code is changed.

The two stages communicate through:

```text
results/rdo_refusal/dim/<model-id>/
├── direction.pt
├── direction_metadata.json
└── generate_directions/
    └── mean_diffs.pt
```

`<model-id>` is the final component of the model path. For example,
`tiiuae/Falcon3-7B-Base` uses `Falcon3-7B-Base`.

## Stage 1: build the DIM refusal direction

Run from the refusal-direction project directory:

```bash
cd /home/user1/buka2004/LLM-Attack-Defense/geometry-of-refusal/refusal_direction
source ../../.venv/bin/activate

python -m pipeline.run_pipeline \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Other examples:

```bash
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B
python -m pipeline.run_pipeline --model_path Rootkit7/Qwen3-8B-abliterated
python -m pipeline.run_pipeline --model_path CWRUSafetyLab/Qwen2.5-1.5B-Instruct-EASE
python -m pipeline.run_pipeline --model_path allenai/Olmo-3-7B-Instruct
python -m pipeline.run_pipeline --model_path tiiuae/Falcon3-7B-Base
```

The pipeline uses the exact `--model_path`; it does not remap Base checkpoints
to Instruct checkpoints.

### What Stage 1 does

1. loads the model and its family adapter;
2. loads harmful/harmless train and validation splits;
3. optionally keeps harmful prompts the model refuses and harmless prompts it
   answers;
4. computes layer/token mean-difference candidate directions;
5. selects a direction using refusal, steering, and KL behavior;
6. writes `direction.pt`, metadata, plots, completions, and evaluations.

Default settings are in:

```text
geometry-of-refusal/refusal_direction/pipeline/config.py
```

Important defaults:

```text
filter_train=True
filter_val=True
evaluation_datasets=("strongreject",)
jailbreak_eval_methodologies=("substring_matching", "strongreject")
max_new_tokens=512
completions_batch_size=128
```

Base or abliterated checkpoints may produce zero harmful refusals. If that is
intentional for the experiment, set `filter_train=False` and
`filter_val=False`; otherwise use an instruction/refusal-tuned checkpoint.

### Stage 1 model families

The current factory recognizes model paths containing:

- `qwen` (also catches DeepSeek Distill Qwen);
- `llama-3`;
- `gemma`;
- `olmo`;
- `falcon3`.

The factory also contains Llama-2 and Yi branches, but their adapter files are
currently absent, so those branches are not operational.

Compatibility behavior:

- `Rootkit7/Qwen3-8B-abliterated` uses the official `Qwen/Qwen3-8B`
  tokenizer while retaining Rootkit weights.
- `tiiuae/Falcon3-7B-Base` uses the
  `tiiuae/Falcon3-7B-Instruct` tokenizer while retaining Base weights.

## Stage 2: train and evaluate RDO

Stage 2 requires the three DIM files from Stage 1.

Run from the repository root:

```bash
cd /home/user1/buka2004/LLM-Attack-Defense
source .venv/bin/activate

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
DIRECTION_MODE="activation_additive_rot" \
NUM_OPT_LAYERS=1 \
K_PROJ=35 \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_rdo_refusal.sh
```

Common alternatives:

```bash
# Original RDO/RefusalCone baseline
MODEL_NAME="Qwen/Qwen3-8B" \
DIRECTION_MODE="baseline" \
NUM_OPT_LAYERS=0 \
./scripts/run_rdo_refusal.sh

# Low-rank Stiefel additive rotation
MODEL_NAME="allenai/Olmo-3-7B-Instruct" \
DIRECTION_MODE="shtiefel_additive_rot" \
NUM_OPT_LAYERS=5 \
K_PROJ=32 \
./scripts/run_rdo_refusal.sh

# Falcon3 Base weights with the compatible Instruct tokenizer
MODEL_NAME="tiiuae/Falcon3-7B-Base" \
DIRECTION_MODE="baseline" \
NUM_OPT_LAYERS=0 \
./scripts/run_rdo_refusal.sh
```

The wrapper accepts these environment overrides:

```text
MODEL_NAME
DIRECTION_MODE
NUM_OPT_LAYERS
K_PROJ
RESULT_ROOT
CUDA_VISIBLE_DEVICES
ABLATION_LAMBDA
ADDITION_LAMBDA
RETAIN_LAMBDA
REPETITION_LAMBDA
MMLU_SAMPLE_SIZE
MMLU_MAX_NEW_TOKENS
PPL_MAX_WINDOWS
ARC_SAMPLE_SIZE
GSM8K_SAMPLE_SIZE
GSM8K_MAX_NEW_TOKENS
GUARD_EVAL_BATCH_SIZE
GUARD_TRAIN_BATCH_SIZE
```

The current wrapper enables LlamaGuard/Qwen3Guard/WildGuard, MMLU, WikiText
perplexity, ARC-Easy, ARC-Challenge, and GSM8K evaluation. Edit the corresponding
`enable_*` variables in the wrapper to disable expensive evaluations.

### Direct Python launch

Use a direct launch when you need options not exposed as environment overrides:

```bash
python -m baselines.rdo_refusal \
  --train_direction \
  --model Qwen/Qwen3-8B \
  --direction_mode activation_additive_rot \
  --num_opt_layers 5 \
  --k_proj 35 \
  --init_mode diag_permutation \
  --orth_method svd \
  --optimizer AdamW \
  --lr 1e-5 \
  --retain_loss \
  --eval_llamaguard \
  --eval_guard_backend llamaguard qwen3guard wildguard \
  --eval_mmlu \
  --eval_ppl \
  --eval_arc_easy \
  --eval_arc_challenge \
  --eval_gsm8k \
  --result_root ./results/rdo_refusal/tensorboard
```

To evaluate/reuse an existing run instead of training a new one:

```bash
python -m baselines.rdo_refusal \
  --model Qwen/Qwen3-8B \
  --direction_mode activation_additive_rot \
  --result_path /path/to/existing/run \
  --eval_llamaguard
```

The existing run must contain the appropriate checkpoint files.

## Steering methods

`--direction_mode` supports:

### `baseline`

Original `RefusalCone`. It learns intervention vectors and applies
ablation/addition at selected layers. Use this as the direct RDO reference.

### `rotation`

Learns one orthogonal Cayley rotation and transforms the DIM refusal direction
as `r = M @ r0`. This mode requires a one-dimensional cone.

### `activation_rot`

Applies a learned full-dimensional orthogonal Cayley rotation directly to each
selected layer's activations.

### `activation_additive_rot`

Learns a low-rank activation subspace of width `k_proj`, rotates inside that
subspace, and preserves the orthogonal complement. This is the principal Cayley
low-rank method used by the grid and ablation scripts.

### `shtiefel_rot`

Learns full-dimensional per-layer activation maps and retracts them onto the
Stiefel/orthogonal manifold after optimizer updates.

### `shtiefel_proj_rot`

Low-rank Stiefel map using factors `QA` and `QB`, with rank controlled directly
by `k_proj`.

### `shtiefel_additive_rot`

Applies the projected Stiefel map only inside the learned subspace. This is the
principal Stiefel low-rank method used by the grid and ablation scripts.

### `angular_steering`

Adaptive angular steering built on per-layer Stiefel rotations.

### `householder_pseudo_rotation`

Uses a norm-preserving Householder map to move the original refusal direction
toward a learned target direction.

### Layer and projection controls

- `num_opt_layers=N`: optimize `N` selected layers.
- `num_opt_layers=0`: optimize all model layers, but only for `baseline`,
  `activation_additive_rot`, and `shtiefel_additive_rot`.
- `k_proj`: low-rank subspace width for projected/additive methods.
- `init_mode`: `random`, `diag_permutation`, `ones`, or `ab_orthogonal`.
- `orth_method`: `qr` or `svd` for Stiefel orthogonalization.
- `optimizer`: `Adam`, `AdamW`, or `SGD`.

Loss weights:

- `ablation_lambda`: suppress harmful/refusal behavior;
- `addition_lambda`: induce the desired steering behavior;
- `retain_lambda`: retain harmless/base-model behavior;
- `repetition_lambda`: repetition penalty term.

## Grid launches

Configure:

```text
scripts/grid_scripts/run_rdo_refusal_grid.sh
```

The grid forms the Cartesian product of:

```bash
model_values=(...)
direction_mode_values=(baseline activation_additive_rot shtiefel_additive_rot)
k_proj_values=(...)
n_of_layers_values=(...)
```

Then run:

```bash
cd /home/user1/buka2004/LLM-Attack-Defense
./scripts/grid_scripts/run_rdo_refusal_grid.sh
```

The grid intentionally allows only:

- `baseline`;
- `activation_additive_rot`;
- `shtiefel_additive_rot`.

Set `RESULT_ROOT` and GPU selection in the grid script. Note that the current
script assigns `CUDA_VISIBLE_DEVICES` internally, so exporting it before launch
does not override that assignment unless the script is changed.

## Paper Angular and Spherical steering

These are **inference-only** refusal baselines that reuse the same SaladBench
data, guard metrics, and experiment logging as RDO. They do **not** require
Stage 1 DIM artifacts and do not train an RDO direction.

Papers:

- [Angular Steering: Behavior Control via Rotation in Activation Space](https://arxiv.org/abs/2510.26243)
  (`paper_angular_steering`)
- [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://arxiv.org/pdf/2602.08169)
  (`paper_spherical_steering`)

### Angular Steering

Fixed attack/protect defaults (attack θ=180°, protect θ=0°). No θ / `k_proj` /
`num_opt_layers` sweep — grids iterate **models only**.

Single run:

```bash
cd /home/user1/buka2004/LLM-Attack-Defense
source .venv/bin/activate

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
RESULT_ROOT="./results/rdo_refusal/AAAI-results/paper_angular_steering" \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_angular_steering_refusal.sh
```

Grid (edit `model_values` / `RESULT_ROOT` / GPU in the script as needed):

```bash
./scripts/grid_scripts/run_angular_steering_refusal_grid.sh
```

Optional overrides: `ANGULAR_STRATEGY`, `ANGULAR_ADAPTIVE_MODE`,
`ANGULAR_ATTACK_DEGREE`, `ANGULAR_PROTECT_DEGREE`,
`ANGULAR_N_EXTRACT_SAMPLES`, `ANGULAR_EXTRACT_BATCH_SIZE`.

### Spherical Steering

Fixed κ / α / β defaults (`kappa=20`, `alpha=0.7`, `beta=0.1`). Grids iterate
**models only**.

Single run:

```bash
cd /home/user1/buka2004/LLM-Attack-Defense
source .venv/bin/activate

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
RESULT_ROOT="./results/rdo_refusal/AAAI-results/paper_spherical_steering" \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_spherical_steering_refusal.sh
```

Grid:

```bash
./scripts/grid_scripts/run_spherical_steering_refusal_grid.sh
```

Optional overrides: `SPHERICAL_KAPPA`, `SPHERICAL_ALPHA`, `SPHERICAL_BETA`,
`SPHERICAL_N_EXTRACT_SAMPLES`, `SPHERICAL_EXTRACT_BATCH_SIZE`.

Both wrappers enable the same LlamaGuard / Qwen3Guard / WildGuard, MMLU,
WikiText perplexity, ARC, and GSM8K evaluations as `run_rdo_refusal.sh`.
Artifacts (including `checkpoints/steering_prototype.pt`) are written under
`RESULT_ROOT`.

## Outputs

Training runs are written below `--result_root`/`RESULT_ROOT`. A run normally
contains:

```text
<run-dir>/
├── hparams.json
├── eval_metrics_<timestamp>.json
├── eval_metrics_<timestamp>.csv
├── checkpoints/
└── TensorBoard event files
```

Cached generated targets are stored under:

```text
results/rdo_refusal/rdo/<model-id>/
```

Inspect TensorBoard logs with:

```bash
tensorboard --logdir results/rdo_refusal
```

## Ablation plots

### `k_proj` ablation

This compares Cayley additive and Stiefel additive experiment families and adds
horizontal RDO reference lines from `rdo_exp_list.txt`:

```bash
python scripts/metrics/ablation_study_r.py \
  -a /path/to/activation_additive_rot \
  -s /path/to/shtiefel_additive_rot \
  --rdo-exp-list results/rdo_refusal/analysis/rdo_exp_list.txt \
  --output-dir results/rdo_refusal/analysis/<model>/ablation_study_k_proj
```

### `num_opt_layers` paired ablation

```bash
python scripts/metrics/ablation_study_nol.py \
  -a /path/to/activation_additive_rot \
  -s /path/to/shtiefel_additive_rot \
  --rdo-exp-list results/rdo_refusal/analysis/rdo_exp_list.txt \
  --output-dir results/rdo_refusal/analysis/<model>/ablation_study_nol
```

### Baseline-only `num_opt_layers` plots

Use `-b` to process only one baseline experiment family without mapped RDO
reference lines:

```bash
python scripts/metrics/ablation_study_nol.py \
  -b /path/to/baseline/experiment-family \
  --output-dir results/rdo_refusal/analysis/baseline/<model>
```

Repeated baseline runs with the same `num_opt_layers` are treated as replicates
and plotted using their mean and standard deviation.

The reference map format is:

```text
model-key: /absolute/or/relative/path/to/baseline/run
```

Current matching supports DeepSeek, Olmo, Qwen3, Qwen2.5 EASE, and Falcon3
entries.

Each metrics output includes:

```text
run_metrics.csv
aggregated_metrics.csv
run_diagnostics.json
summary.json
families/family_<id>/
├── family_parameters.json
├── tables/aggregated_metrics.csv
└── plots/*.png and *.pdf
```

## Evaluation backends and benchmarks

Guard backends:

- `llamaguard`;
- `qwen3guard`;
- `wildguard`.

Capability/locality evaluations:

- MMLU accuracy;
- WikiText-2 perplexity;
- ARC-Easy accuracy;
- ARC-Challenge accuracy;
- GSM8K exact match.

These models and datasets are downloaded from Hugging Face unless already
cached. LlamaGuard may require accepted model terms and an authenticated token.

## Troubleshooting

### `DIM direction files not found`

Run Stage 1 with exactly the same model ID before starting RDO. Confirm:

```bash
ls results/rdo_refusal/dim/<model-id>/
```

The directory must contain `direction.pt`, `direction_metadata.json`, and
`generate_directions/mean_diffs.pt`.

### `Unknown model family`

The geometry model factory selects an adapter by substring in `--model_path`.
Use a supported family name or add a dedicated adapter and factory branch.

### `tokenizer.chat_template is not set`

Base checkpoints often do not include a chat template. Falcon3 Base is handled
through its Instruct tokenizer. Other Base models need a compatible tokenizer
or a dedicated raw-prompt adapter.

### `0 harmful refusals detected`

The checkpoint is not refusing the harmful split, or the prompt template is
wrong. Use an Instruct checkpoint or deliberately disable train/validation
filtering for raw difference-in-means experiments.

### CUDA out of memory

Reduce:

- `completions_batch_size` in pipeline config;
- `GUARD_EVAL_BATCH_SIZE`;
- `GUARD_TRAIN_BATCH_SIZE`;
- MMLU/ARC/GSM8K sample sizes;
- `k_proj` or `num_opt_layers`.

### Hugging Face connection/authentication failures

Check:

```bash
echo "${HF_TOKEN:+HF_TOKEN is set}"
hf auth whoami
```

Transient `HEAD` request failures are normally retried. Gated models require
access approval.

### Metrics reports multiple parameter families

The experiment directory contains runs whose non-volatile `hparams.json` fields
differ. Point the command at a narrower family directory or inspect the
differing parameters. `result_root` and other storage metadata are excluded from
family identity.

### Metrics reports duplicate ablation settings

Paired rotation analysis rejects duplicate runs to avoid silently averaging
accidental duplicates. Baseline-only `-b` mode intentionally averages repeated
runs as replicates.

## Recommended workflow

```text
Install environment
  -> configure geometry-of-refusal/.env
  -> run pipeline.run_pipeline for each model
  -> verify DIM files
  -> run one RDO experiment
  -> inspect evaluations/checkpoints
  -> run the grid
  -> generate ablation plots
```
