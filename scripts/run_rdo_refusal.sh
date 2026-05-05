#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# variables
direction_mode="householder_pseudo_rotation" # "shtiefel_rot", "activation_rot", "baseline", "shtiefel_proj_rot", "angular_steering", "householder_pseudo_rotation"
num_opt_layers=0 # optimize this many middle layers (plus best layer during intervention)
llamaguard_data="basic" # "rdo" (data/<splits>_splits/*_<eval_split>.json) or "basic" (SAVE_DIR/rdo/<model>/basic/targets/)
lr=1e-5
max_iters=100000
result_path="/home/user1/buka2004/LLM-Attack-Defense/results/rdo_refusal/tensorboard/basic_rdo_DeepSeek-R1-Distill-Qwen-7B_householder_pseudo_rotation_nol=0_im=random_om=svd_c2fd4a3abbbc" # e.g. results/rdo_refusal/tensorboard/<existing_run_dir> (leave empty to train)
log_steps=1000
init_mode="random" # "random" or "diag_permutation"
orth_method="svd" # "qr" or "svd"

export MAX_ITERS="${max_iters}"

# ---- GPU (override: CUDA_VISIBLE_DEVICES=1 ./scripts/run_rdo_refusal.sh) ----
export CUDA_VISIBLE_DEVICES=1

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-20}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-32}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-false}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-100}"

# LlamaGuard eval caps (total counts, not per-category).
export MAX_HARMFUL="${MAX_HARMFUL:-500}"
export MAX_HARMLESS="${MAX_HARMLESS:-500}"

export HF_TOKEN="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"

# ---- Default command ----
cmd=(python -m baselines.rdo_refusal \
  --train_direction \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --direction_mode "${direction_mode}" \
  --llamaguard_data "${llamaguard_data}" \
  --lr "${lr}" \
  --eval_llamaguard \
  # --eval_mmlu \
  # --mmlu_store_predictions \
  --num_opt_layers "${num_opt_layers}" \
  --init_mode "${init_mode}" \
  --orth_method "${orth_method}" \
  --log_steps "${log_steps}" \
  # --retain_loss \
)

# --init_mode: "random" or "diag_permutation"

# Reuse an existing run dir (skip training) if provided.
if [[ -n "${result_path}" ]]; then
  cmd+=(--result_path "${result_path}")
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
