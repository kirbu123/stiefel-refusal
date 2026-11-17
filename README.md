# Refusal Pipeline

This document covers the retained RDO workflow on `main`:

1. build a DIM refusal direction with `pipeline.run_pipeline`;
2. train/evaluate an RDO steering method with `baselines.rdo_refusal`;
3. launch individual or grid experiments;
4. optionally run paper Angular / Spherical inference-only steering baselines
   (no DIM / no RDO training);
5. generate `k_proj` and `num_opt_layers` ablation plots;
6. optionally dump best-layer activations and build PCA comparison figures.

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

## Alibaba PPU server: labs-pool

On `ali117197`, the training partition is **labs-pool**. Slurm calls the
accelerator GRES `gpu`, but assigns Alibaba PPUs, not NVIDIA GPUs. Physical
indices and MIG UUIDs must not be chosen manually. Corporate resources are not used.

Both launchers submit batch jobs, enter the installed SAIL 2.1/PyTorch 2.10 image,
and set up `.venv-ppu-sail21` **inside the allocated labs-pool job**. Vendor torch,
torchvision/torchaudio, NumPy, Triton and bitsandbytes are inherited from the image
and protected by constraints. Use `requirements-ppu.txt`, not the NVIDIA root
requirements, in this environment. PyArrow 23.0.1 supports vendor NumPy 1.26.

```bash
# From this repository root: automatic setup and batch submission.
./scripts/run_rdo_refusal.sh
./scripts/run_dim_pipeline.sh --model_path mistralai/Magistral-Small-2506 --batch_size 2

# Optional setup only, live resources, and small preflight checks.
./scripts/run_rdo_refusal.sh --setup-only
./scripts/run_rdo_refusal.sh --resources
./scripts/run_rdo_refusal.sh --check_environment
./scripts/run_dim_pipeline.sh --model_path mistralai/Magistral-Small-2506 --check_environment

# Real-backend smoke tests; no pretrained model weights downloaded.
./scripts/run_rdo_refusal.sh --smoke-test
./scripts/run_dim_pipeline.sh --model_path mistralai/Magistral-Small-2506 --smoke-test
```

Defaults: one **whole 96 GiB PPU**, 8 CPUs, 128 GiB host RAM, 24 hours, lowest
priority `pool-batch`. Whole cards avoid the earlier MIG fencing failure path and
provide headroom for Magistral 24B. Jobs can queue. Optional overrides:
`RDO_PPU_PROFILE=half|auto|whole`, `RDO_PPU_CPUS`, `RDO_PPU_RAM`, `RDO_PPU_TIME`,
`RDO_PPU_IMAGE`, `RDO_PPU_VENV`. Use a separate venv for a different image.
Short setup/check/smoke commands wait at most 30 seconds for allocation.
Slurm's assigned visibility mask is preserved, including with `apptainer --cleanenv`.

Console logs: `results/rdo_refusal/slurm/rdo-ppu-<JobID>.log` and
`dim-ppu-<JobID>.log`. Check status with `squeue -u "$USER"`.
The launcher handles Slurm's temporary batch-script path independently of `$0`.
The generic installation instructions below apply to other/NVIDIA hosts.

Magistral DIM support uses the native Tekken tokenizer, official system prompt,
Mistral decoder hooks and tokenizer-specific refusal IDs. `--batch_size` controls
both generation and direction scoring, default 2. If the existing short-completion
refusal filter finds no refusals for a reasoning model, `--no_filter` explicitly
uses raw data instead; this changes the experiment's data selection.

Direct host `python -m pipeline.run_pipeline` does not allocate PPUs; use the DIM
launcher above. The tmux/libtinfo version warning is unrelated to model support.
RDO's CLI learning rate and loss weights now reach the training function. All
existing experiment settings in the shell script, including cleanup, are retained.

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
EVAL_ACTIVATION_PCA
ACTIVATION_SAVE_GAP
KEEP_ACTIVATION_PCA_DUMPS
```

The current wrapper enables LlamaGuard/Qwen3Guard/WildGuard, MMLU, WikiText
perplexity, ARC-Easy, ARC-Challenge, and GSM8K evaluation. Edit the corresponding
`enable_*` variables in the wrapper to disable expensive evaluations.

### Activation PCA dumps during training

To record best-layer post-steering activations (attack mode) and write a per-run
3D PCA trajectory plot:

```bash
EVAL_ACTIVATION_PCA=true \
ACTIVATION_SAVE_GAP=100 \
KEEP_ACTIVATION_PCA_DUMPS=true \
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
DIRECTION_MODE="activation_additive_rot" \
NUM_OPT_LAYERS=1 \
K_PROJ=35 \
./scripts/run_rdo_refusal.sh
```

Equivalent CLI flags for a direct Python launch:

```text
--eval_activation_pca
--activation_save_gap N          # dump every N optimizer steps (plus first/final)
--keep_activation_pca_dumps      # keep checkpoints/tmp_best_layer_activations/
```

Without `--keep_activation_pca_dumps`, dumps are deleted after the per-run plot.
Kept dumps are required by the multi-experiment PCA comparison scripts below.
Snapshots are mean last-token residuals under **attack** steering on a fixed
harmful probe set.

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
  --eval_activation_pca \
  --activation_save_gap 100 \
  --keep_activation_pca_dumps \
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

Training keeps one detached best operator on CPU, replacing it only when the
training loss improves (equal losses keep the first best state). It does not
accumulate operator histories or write `cayley_params.pt` / `reft_params.pt`.
Evaluation loads the best checkpoint and ignores legacy operator-history files
even if they are present. Existing history files are not deleted automatically.
Scalar loss logs, vector/PCA diagnostics and explicitly requested `log_steps`
progress checkpoints retain their existing behavior.

On ALI, use the launcher so evaluation stays in the scheduler/container runtime:

```bash
env RDO_PLATFORM=ppu RDO_DEVICE=cuda CLEAR_CKPTS=false RDO_LLAMAGUARD_STRICT=1 \
  MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  DIRECTION_MODE=shtiefel_additive_rot \
  ./scripts/run_rdo_refusal.sh --result_path /absolute/path/to/existing/run
```

Use the saved model, method, rank, layer configuration and orthogonalization
setting from the run's `hparams.json`. `--result_path` loads the best checkpoint
and restarts evaluation without retraining. `CLEAR_CKPTS=false` preserves tensor
checkpoints after success; `RDO_LLAMAGUARD_STRICT=1` makes guard failures fail the
job with their original error. Both settings are forwarded into the PPU container.
The basic guard dataset is loaded directly through `datasets`, independently of
the installed Heretic version or the repository's local `heretic` package.

Offline CPU evaluation smoke test (requires PyTorch, datasets, tensorboard and
pydantic-settings; no model/dataset downloads during the test):

```bash
python -m unittest discover -s tests -p 'test_rdo_eval_resume_smoke.py' -v
```

The smoke test reloads constrained/unconstrained checkpoints, applies steering on
CPU tensors, and runs the actual evaluation orchestration and artifact writers
with local prompt fixtures and guard test doubles. It also exercises dataset,
generation and classifier failures to prevent secondary `UnboundLocalError`s.

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

### `paper_angular_steering` / `paper_spherical_steering`

Inference-only paper baselines (no DIM / no RDO training). See
[Paper Angular and Spherical steering](#paper-angular-and-spherical-steering).

### Layer and projection controls

- `num_opt_layers=N`: optimize `N` selected layers.
- `num_opt_layers=0`: optimize all model layers, but only for `baseline`,
  `activation_additive_rot`, `shtiefel_additive_rot`, `loreft`, and `direft`.
- `k_proj`: low-rank subspace width for projected/additive methods.
- `init_mode`: `random`, `diag_permutation`, `ones`, or `ab_orthogonal`.
- `orth_method`: `qr` or `svd` for Stiefel orthogonalization.
- `orthogonalize`: `true` (default) or `false`; controls training constraints for
  `activation_additive_rot` and `shtiefel_additive_rot` only.
- `optimizer`: `Adam`, `AdamW`, or `SGD`.

Both additive rotation methods keep their current orthogonal initialization for
either setting. With `orthogonalize=true`, forward projections and retraction
after every optimizer step behave as before. With `false`, training skips both
forward SVD projections (including their custom backward rules) and post-update
retractions. Cayley additive uses its raw learned basis, but its subspace map
remains orthogonal by the Cayley formula. Stiefel additive uses raw factors
`B = proj_A` and `Q = proj_B @ proj_A`, with no orthogonal projection.
Scaling, clipping, layer selection, and attack/protect transpose conventions are
unchanged; without constraints the protect map is not guaranteed to invert attack.

The launcher currently defaults to `true`. Override it using the environment or
an explicit CLI argument (CLI takes precedence):

```bash
ORTHOGONALIZE=false DIRECTION_MODE=shtiefel_additive_rot ./scripts/run_rdo_refusal.sh
./scripts/run_rdo_refusal.sh --orthogonalize false
```

Passing `false` for other methods is rejected before model loading. The setting
is recorded in run configuration and progress/active-layer checkpoint metadata;
dense checkpoints reload the actual saved map without re-projecting it. Older
artifacts remain compatible with the default `true` setting. CPU coverage:

```bash
.venv-reft/bin/python -m unittest discover -s tests -p 'test_orthogonalize_flag.py' -v
```

### Training diagnostics and timing

Training writes CSV logs directly into the experiment directory beside the
TensorBoard events. These files survive `clear_ckpts` cleanup:

- `operator_metrics.csv` (the two additive rotation methods): one row per active
  layer with `b_frobenius`, `q_frobenius`, `b_orthogonality_error`, and
  `q_orthogonality_error`. Errors are Frobenius norms of `B.T @ B - I` and
  `Q.T @ Q - I`. Measurements use the effective factors in float32 without
  gradients; unconstrained factors are not projected for measurement.
- `training_forward_timing.csv` (all trained direction modes): one row per actual
  model forward, with `steering_application_ms`, `model_forward_ms`, steering
  region count, upcoming optimizer step, microbatch iteration, phase, actual
  batch/sequence dimensions, and timing backend. Combined nnsight invocation
  batches produce one model timing row.

`TRAIN_DIAGNOSTICS_GAP=100` / `--train_diagnostics_gap 100` is the default. Forward
timing samples every accumulated microbatch leading to optimizer update 1 and
updates 100, 200, etc. Operator metrics describe initialization, those post-update
states, and the final updated state (without duplicate final rows). Set the gap
to `0` to disable diagnostics. CLI arguments override the environment setting:

```bash
TRAIN_DIAGNOSTICS_GAP=100 ./scripts/run_rdo_refusal.sh
./scripts/run_rdo_refusal.sh --train_diagnostics_gap 0
```

Steering timing covers application to activations, including scaling/clipping,
and excludes constructing B/Q. Model timing includes steering and runtime hook
overhead, ending at logits before traced loss/backward. Backward, optimizer work,
post-update retraction, scanning, guard validation, PCA probes, and final
evaluation are excluded. CPU uses `perf_counter`; CUDA-compatible devices use
stream events resolved after the forward, without per-layer synchronization.
Timing is sampled instrumentation, not an estimate of overhead-free throughput.
Training score forwards are identified by their phase; unsteered reference
forwards have zero steering regions. Timing definitions and cadence are saved in
`hparams.json`. Existing training CSVs are never overwritten.

Run offline CPU checks with:

```bash
python -m unittest discover -s tests -p 'test_training_diagnostics.py' -v
```

The real deferred-execution integration test requires `nnsight==0.3.7` and its
dependencies; it uses a tiny local model without downloading model weights.

### `loreft` / `direft`

These methods use [ReFT equations 2–3](https://proceedings.neurips.cc/paper_files/paper/2024/file/75008a0fba53bf13b0bb3b7bff986e0e-Paper-Conference.pdf)
as layer-input interventions in the existing RDO training pipeline. For column
activations `h`, each selected layer learns its own rank-`k_proj` residual:

- LoReFT: `delta(h) = R.T @ (W @ h + b - R @ h)`, with `R @ R.T = I`.
  Initialize a random orthonormal `R`, copy `W = R`, and set `b = 0`; the initial
  operator is identity. Orthogonality is enforced by a Householder parametrization.
- DiReFT: `delta(h) = W2.T @ (W1 @ h + b)`. Initialize `W1` with PyTorch linear-layer
  Kaiming uniform initialization, copy `W2 = W1`, and set `b = 0`. Both matrices
  remain unconstrained; this initialization is intentionally not identity.

Copies are independent trainable parameters. Attack applies `h + delta(h)`;
defence (`protect`) applies `h - delta(h)` using the same learned parameters.
Defence reverses the residual sign and is not generally the exact inverse of
attack. The training `add` path also reverses the residual sign at its requested
layer. No learned norm scaling or clipping is applied. `init_mode` and
`orth_method` do not override these ReFT-specific initialization/constraint rules.
Use `--train_direction` (the script default); refusal cones are not supported.

Interventions cover all token positions at the selected layer inputs, including
generation steps; this reuses the project's activation-intervention placement,
rather than reproducing the paper's prefix/suffix position search.

```bash
DIRECTION_MODE=loreft K_PROJ=35 NUM_OPT_LAYERS=1 ./scripts/run_rdo_refusal.sh
DIRECTION_MODE=direft K_PROJ=35 NUM_OPT_LAYERS=0 ./scripts/run_rdo_refusal.sh
```

ReFT checkpoints retain the bias and complete parameter state, rank, method,
and selected layers. Best-loss and final artifacts are `lowest_loss_reft.pt` and
`final_reft.pt` under `checkpoints/`; per-step operator histories are not saved.
Progress and active-layer artifacts also include the ReFT state. Existing
`--result_path` evaluation restores the saved rank and layers. The launcher's
default `clear_ckpts=true` setting deletes checkpoints after successful training
and evaluation; set `CLEAR_CKPTS=false` to retain artifacts for reuse.

Offline CPU smoke tests need only PyTorch, with no model or dataset downloads:

```bash
python3 -m venv .venv-reft
.venv-reft/bin/python -m pip install torch
.venv-reft/bin/python -m unittest discover -s tests -p 'test_reft_smoke.py' -v
```

Tests exercise exact formulas and initialization, gradients and orthogonality
during tiny frozen-model training, layer-input adapters, attack/protect modes,
checkpoint reload, and CPU launcher argument forwarding. They do not run a full
nnsight model or guard service.

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
│   ├── best_layer_activation_pca_3d.png   # if --eval_activation_pca
│   └── tmp_best_layer_activations/        # if --keep_activation_pca_dumps
│       └── step_*.pt
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

## Visualization scripts

### Multi-experiment activation PCA (shared start)

Fits one centered 3D PCA across dumps from several runs, aligns trajectories to a
shared start, and draws the DIM refusal direction. Requires
`checkpoints/tmp_best_layer_activations/` in each run.

```bash
python scripts/metrics/plot_activation_pca_compare.py \
  --exp-dirs \
    /path/to/run_activation_additive_rot \
    /path/to/run_baseline \
    /path/to/run_shtiefel_additive_rot \
  --dim-root results/rdo_refusal/dim
```

Default output:

```text
<first-exp>/checkpoints/activation_pca_compare_3d.png
```

### Multi-experiment activation PCA (origin vectors)

Uncentered PCA so plot origin is zero activation. Plots only start/end points as
vectors from the origin (no shared-start alignment), plus a compact refusal
direction cue.

```bash
python scripts/metrics/plot_activation_pca_vector_compare.py \
  --exp-dirs \
    /path/to/run_activation_additive_rot \
    /path/to/run_baseline \
    /path/to/run_shtiefel_additive_rot \
  --dim-root results/rdo_refusal/dim
```

Default output:

```text
<first-exp>/checkpoints/activation_pca_vector_compare_3d.png
```

Legend labels map `baseline` → RDO, `activation_additive_rot` → Cayley Steering,
`shtiefel_additive_rot` → Stiefel Rotation.

## Ablation plots

Batch driver over all models under the AAAI NOL / `k_proj` roots (and optional
baseline NOL trees):

```bash
python scripts/metrics/ablation_study.py \
  --rdo-exp-list results/rdo_refusal/AAAI-results/rdo_exp_list.yaml \
  --output-dir results/rdo_refusal/analysis
```

Add `-b results/rdo_refusal/AAAI-results/baseline/nol-ablations` to also plot
baseline-only NOL families under `analysis/baseline/<model>/`.

Horizontal reference lines (RDO / Angular / Spherical) are **not** stored inside
`analysis/`; they are loaded from experiment dirs listed in
`results/rdo_refusal/AAAI-results/rdo_exp_list.yaml`. Curve data for the
Cayley/Stiefel series is written under `analysis/` as CSVs.

### `k_proj` ablation

Compares Cayley additive and Stiefel additive experiment families and adds
horizontal reference lines from the YAML map:

```bash
python scripts/metrics/ablation_study_r.py \
  -a /path/to/activation_additive_rot \
  -s /path/to/shtiefel_additive_rot \
  --rdo-exp-list results/rdo_refusal/AAAI-results/rdo_exp_list.yaml \
  --output-dir results/rdo_refusal/analysis/<model>/ablation_study_k_proj
```

### `num_opt_layers` paired ablation

```bash
python scripts/metrics/ablation_study_nol.py \
  -a /path/to/activation_additive_rot \
  -s /path/to/shtiefel_additive_rot \
  --rdo-exp-list results/rdo_refusal/AAAI-results/rdo_exp_list.yaml \
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

The YAML reference map lists per-model experiment dirs for `rdo`,
`angular-steering`, and `spherical-steering` sections (see
`results/rdo_refusal/AAAI-results/rdo_exp_list.yaml`).

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
  -> run one RDO experiment (optionally EVAL_ACTIVATION_PCA=true)
  -> inspect evaluations/checkpoints
  -> run the grid
  -> generate ablation plots (ablation_study.py)
  -> optional: plot_activation_pca_compare / plot_activation_pca_vector_compare
```

### SafetyAnalisys: reference-based answer quality

Enable this optional **final** safety evaluator alongside the guards:

```bash
./scripts/run_rdo_refusal.sh \
  --eval_guard_backend llamaguard qwen3guard wildguard safetyanalisys \
  --safety_analisys_references /absolute/path/to/references.json \
  --safety_analisys_batch_size 1
```

The safety stage (`--eval_llamaguard`) must be enabled. Periodic training
validation does not run SafetyAnalisys. Without this backend, evaluation behavior
is unchanged apart from the additional, empty `split` column in the metrics CSV.

Both post-train evaluation and offline backfill default to
`./val_utils/safety_analisys_references.json`, relative to the current working
directory. Run from the repository root. Fill its empty fields, or pass an explicit
references path to use a separate file.
The template deliberately fails validation until filled. The version-1 JSON has
`harmful` and `harmless` objects, each containing keys `"1"` through `"4"`.
Every key contains the same positive number n of `{ "prompt": "...", "response":
"..." }` examples: 8n examples in total. **1 means worst quality; 4 means best
quality**, for both categories. Examples define what quality means in each
category; this score is distinct from the guards' harmfulness scale.

A separately loaded, frozen `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` scores
conditional likelihoods of the four continuations, including multi-token labels.
The selected integer is the most likely continuation; ties select the smaller
score. Probabilities are normalized over these four candidates only. No judge
reasoning is generated. All 4n examples for the relevant category are included;
context overflow fails explicitly without truncation. The judge requires memory
in addition to the evaluated model and uses the available Hugging Face cache.

The evaluator reuses safety generations for `initial`, `refined_attack`, and,
where supported, `refined_protect`. Four separate means are reported per phase:
SafetyAnalisys harmful train/test and harmless train/test. Existing source
conventions apply: `basic` is labeled `train`, `rdo` is labeled `test`; the latter
uses the configured `eval_split`, which may actually be `val`. Actual sources are
recorded in JSON metadata. Existing sample limits also apply.

Results appear under `safety_analisys` in the final metrics JSON, as `guard` rows
with backend `safetyanalisys` and a `split` column in CSV, and at
`eval/guard/safetyanalisys/{split}/{kind}/{phase}/mean_score` in TensorBoard.
The JSON records the model, prompt version, and references SHA-256. A separate
`safety_analisys_scores_*.json` stores each candidate's prompt, response, selected
score, and four probabilities. Research readers distinguish train/test as
`harmful_train`, `harmful_test`, `harmless_train`, and `harmless_test` groups.
Errors are recorded under `guard_errors.safetyanalisys`; setting
`RDO_LLAMAGUARD_STRICT=1` makes them fatal. Missing or invalid reference files
are rejected before training.

#### Backfill SafetyAnalisys from saved answers

Run from the repository root with one or more experiment directories:

```bash
python3 scripts/backfill_safety_metrics.py \
  --exp-dirs /absolute/path/to/exp1 /absolute/path/to/exp2 \
  --metrics safetyanalisys \
  --safety-analisys-references /absolute/path/to/references.json \
  --safety-analisys-batch-size 1
```

Currently `safetyanalisys` is the only supported metric and the default when
`--metrics` is omitted. `--safety-analisys-references` is optional and defaults
to the same file as post-train evaluation (see above).
Only missing or invalid groups (split/category/phase)
are scored. Training, response generation, and other evaluators are not run.
The script reuses one lazily loaded judge for the supplied experiments.

On Alibaba PPU, the same Python command automatically uses `scripts/launch_ppu.sh`
to submit a `backfill-ppu` job in `labs-pool` and run inside the SAIL container
with `.venv-ppu-sail21`, just like the training launcher. Detection uses
`RDO_PLATFORM=ppu` or `/dev/alixpu` when `RDO_PLATFORM=auto`; `RDO_DEVICE=cpu`
opts out. Existing `RDO_PPU_*` resource/image/venv overrides apply, and Slurm's
assigned device mask is preserved. Logs are written to
`results/rdo_refusal/slurm/backfill-ppu-<JobID>.log`. Already inside the container,
the script runs directly. `--dry-run` and `--help` stay local without allocation.

Use `--dry-run` to inspect response files and missing groups without loading
the judge or writing artifacts; references are not required for this mode.
For each experiment, the latest `eval_metrics_*.json` and its existing CSV are
updated. Each split uses the latest response batch at or before the metrics
filename timestamp; all phases must belong to that batch. Missing phases or
mismatched prompts cause an error rather than mixing response batches.

Existing metrics are preserved. Results include detailed per-answer scores,
JSON aggregates, CSV rows with `split`, and TensorBoard scalars at step 0.
Repeated runs skip completed groups and repair derived CSV/TensorBoard entries
without judging them again. A partial metric requires the same reference hash,
judge model, and prompt version. Fully completed experiments do not require
references or loading the judge. TensorBoard publication requires `tensorboard`.
Each file is replaced atomically; interrupted publication can be retried.
Errors are reported per experiment, processing continues, and any error gives
a nonzero exit status.

CPU smoke tests use a stub judge and temporary experiments:

```bash
python3 -m unittest discover -s tests -p 'test*safety*.py' -v
```

No models are downloaded by these tests. The TensorBoard event integration
check requires `tensorboard`; the existing likelihood check requires `torch`.
