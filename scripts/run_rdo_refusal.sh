#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# ---- GPU (override: CUDA_VISIBLE_DEVICES=1 ./scripts/run_rdo_refusal.sh) ----
export CUDA_VISIBLE_DEVICES=0,1,2

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-20}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-16}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-false}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-1}"

# Dataset caps for eval split JSONs.
export RDO_MAX_HARMFUL_PER_CATEGORY="${RDO_MAX_HARMFUL_PER_CATEGORY:-2}"
export RDO_MAX_HARMLESS_TOTAL="${RDO_MAX_HARMLESS_TOTAL:-20}"

# ---- Default command ----
python -m baselines.rdo_refusal \
  --train_direction \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --direction_mode activation_rot \
  --lr 3e-4 \
  --eval_llamaguard \
  --eval_mmlu \
  --mmlu_store_predictions \
  

# ---- Optional eval flags ----
# Add these if you want metrics after training:
#   --eval_llamaguard --eval_mmlu
#
# LlamaGuard requires HF access if using a gated model:
# export HF_TOKEN="..."
# export LLAMAGUARD_MODEL_NAME="meta-llama/Meta-Llama-Guard-2-8B"
# export LLAMAGUARD_DEVICE="cuda"
# export LLAMAGUARD_DTYPE="bfloat16"
