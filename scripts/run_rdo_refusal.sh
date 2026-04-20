#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# variables
direction_mode="baseline" # "shtiefel_rot", "activation_rot", "baseline"
lr=1e-5
max_iters=20
result_path="/home/buka2004/work/LLM-MOTIONS/LLM-Attack-Defense/results/rdo_refusal/tensorboard/basic_rdo_DeepSeek-R1-Distill-Qwen-1.5B_baseline" # e.g. results/rdo_refusal/tensorboard/<existing_run_dir> (leave empty to train)

export MAX_ITERS="${max_iters}"

# ---- GPU (override: CUDA_VISIBLE_DEVICES=1 ./scripts/run_rdo_refusal.sh) ----
export CUDA_VISIBLE_DEVICES=0

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-20}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-32}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-false}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-32}"

# Dataset caps for eval split JSONs.
export RDO_MAX_HARMFUL_PER_CATEGORY="${RDO_MAX_HARMFUL_PER_CATEGORY:-200}" # 200
export RDO_MAX_HARMLESS_TOTAL="${RDO_MAX_HARMLESS_TOTAL:-200}" # 200

export HF_TOKEN="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"

# ---- Default command ----
cmd=(python -m baselines.rdo_refusal \
  --train_direction \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --direction_mode "${direction_mode}" \
  --lr "${lr}" \
  --eval_llamaguard
)

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
