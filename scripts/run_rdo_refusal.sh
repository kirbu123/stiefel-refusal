#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# variables
direction_mode="shtiefel_proj_rot" # "shtiefel_rot", "activation_rot", "baseline", "shtiefel_proj_rot"
freeze_step=20 # with --freeze_order_layers: freeze layers 0, freeze_step, 2*freeze_step, ...
llamaguard_data="rdo" # "rdo" (data/<splits>_splits/*_<eval_split>.json) or "basic" (SAVE_DIR/rdo/<model>/basic/targets/)
lr=1e-5
max_iters=100000
train_direction=1 # set empty to skip training: init steering from these parameters and run selected evals
result_path="" # e.g. results/rdo_refusal/tensorboard/<existing_run_dir> (leave empty to train; ignored when train_direction is empty)
eval_efficacy="" # set to 1 to pass --eval_efficacy (inference time and method overhead; no metric scoring)

export MAX_ITERS="${max_iters}"

# ---- GPU ----
# Bare launch stays on GPU 0. A preset list is kept, e.g.
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_rdo_refusal.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-20}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-32}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-false}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-100}"

# Dataset caps for eval split JSONs.
export RDO_MAX_HARMFUL_PER_CATEGORY="${RDO_MAX_HARMFUL_PER_CATEGORY:-200}" # 200
export RDO_MAX_HARMLESS_TOTAL="${RDO_MAX_HARMLESS_TOTAL:-200}" # 200

export HF_TOKEN="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"

# ---- Default command ----
cmd=(python -m baselines.rdo_refusal \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --direction_mode "${direction_mode}" \
  --llamaguard_data "${llamaguard_data}" \
  --lr "${lr}" \
  --eval_llamaguard \
  --eval_mmlu \
  --mmlu_store_predictions \
  --freeze_order_layers \
  --freeze_step "${freeze_step}" \
  --init_mode "diag_permutation" \
)

# --init_mode: "random" or "diag_permutation"

# Empty train_direction skips training and only initializes steering, then runs evals.
if [[ -n "${train_direction}" ]]; then
  cmd+=(--train_direction)
fi

# Reuse an existing run dir (skip training inside train_refusal_vector) if provided.
if [[ -n "${result_path}" ]]; then
  cmd+=(--result_path "${result_path}")
fi

if [[ -n "${eval_efficacy}" ]]; then
  cmd+=(--eval_efficacy)
fi

# Optional eval flags:
# cmd+=(--eval_mmlu)
# cmd+=(--mmlu_store_predictions)

"${cmd[@]}"

# ---- Optional eval flags ----
# Add these if you want metrics after training:
#   --eval_llamaguard --eval_mmlu
#
# LlamaGuard requires HF access if using a gated model:
# export HF_TOKEN="..."
# export LLAMAGUARD_MODEL_NAME="meta-llama/Meta-Llama-Guard-2-8B"
# export LLAMAGUARD_DEVICE="cuda"
# export LLAMAGUARD_DTYPE="bfloat16"
