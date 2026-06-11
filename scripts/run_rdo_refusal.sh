#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# variables
direction_mode="shtiefel_rot" # "shtiefel_rot", "activation_rot", "baseline", "shtiefel_proj_rot", "angular_steering", "householder_pseudo_rotation"
train_guard_val_gap=0 # 0 disables; otherwise run train guard validation every N dataloader iterations
num_opt_layers=1 # optimize this many middle layers (plus best layer during intervention)
llamaguard_data="basic" # "rdo" (data/<splits>_splits/*_<eval_split>.json) or "basic" (SAVE_DIR/rdo/<model>/basic/targets/)
eval_guard_backends=("llamaguard" "qwen3guard") # any subset of: "llamaguard" "qwen3guard" "wildguard"
lr=1e-5
optimizer="AdamW" # "Adam", "AdamW", "SGD"
max_iters=10000
result_path="" # e.g. results/rdo_refusal/tensorboard/<existing_run_dir> (leave empty to train)
log_steps=0
init_mode="diag_permutation" # "random", "diag_permutation", or "ab_orthogonal"
orth_method="qr" # "qr" or "svd"
proj_reduce_ratio=100 # used when direction_mode="shtiefel_proj_rot" (k = hidden_size / ratio)

# LlamaGuard eval config
eval_max_new_tokens=256 # inportant param for llama guard eval
eval_batch_size=8
MAX_HARMFUL=20 # 200
MAX_HARMLESS=20 # 200

# MMLU eval config
enable_mmlu_eval=false # true
mmlu_dataset="cais/mmlu"
mmlu_subset="all"
mmlu_split="test" # train|val|test
mmlu_mode="zero_shot" # zero_shot|few_shot
mmlu_answer_mode="generate" # generate|logits
mmlu_n_shots=5
mmlu_sample_size=300
mmlu_sample_seed=42
mmlu_max_new_tokens=8
mmlu_store_predictions=false

export MAX_ITERS="${max_iters}"

# ---- GPU (override: CUDA_VISIBLE_DEVICES=1 ./scripts/run_rdo_refusal.sh) ----
export CUDA_VISIBLE_DEVICES=6

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-$mmlu_sample_size}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-$mmlu_max_new_tokens}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-$mmlu_store_predictions}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-100}"
# Optional guard-generation micro-batch overrides (reduce to avoid CUDA OOM):
export GUARD_EVAL_BATCH_SIZE="${GUARD_EVAL_BATCH_SIZE:-$eval_batch_size}"
export GUARD_TRAIN_BATCH_SIZE="${GUARD_TRAIN_BATCH_SIZE:-$eval_batch_size}"

# LlamaGuard eval caps (total counts, not per-category).
export MAX_HARMFUL="${MAX_HARMFUL}" # 250
export MAX_HARMLESS="${MAX_HARMLESS}" # 250

# export HF_TOKEN="hf_fJyEXMwqeWZJvzLrBqCLDajXibFEDMGWW"
# export HF_TOKEN="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"
export HF_TOKEN="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"

# ---- Default command ----
cmd=(python -m baselines.rdo_refusal \
  --train_direction \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --direction_mode "${direction_mode}" \
  --llamaguard_data "${llamaguard_data}" \
  --lr "${lr}" \
  --optimizer "${optimizer}" \
  --eval_llamaguard \
  --num_opt_layers "${num_opt_layers}" \
  --proj_reduce_ratio "${proj_reduce_ratio}" \
  --init_mode "${init_mode}" \
  --orth_method "${orth_method}" \
  --log_steps "${log_steps}" \
  --train_guard_val_gap "${train_guard_val_gap}" \
  --eval_max_new_tokens "${eval_max_new_tokens}" \
  --eval_batch_size "${eval_batch_size}" \
  --mmlu_dataset "${mmlu_dataset}" \
  --mmlu_subset "${mmlu_subset}" \
  --mmlu_split "${mmlu_split}" \
  --mmlu_mode "${mmlu_mode}" \
  --mmlu_answer_mode "${mmlu_answer_mode}" \
  --mmlu_n_shots "${mmlu_n_shots}" \
  --mmlu_sample_size "${MMLU_SAMPLE_SIZE}" \
  --mmlu_sample_seed "${mmlu_sample_seed}" \
  --mmlu_max_new_tokens "${MMLU_MAX_NEW_TOKENS}" \
  --retain_loss \
  --retain_lambda 1.0 \
)

# --init_mode: "random", "diag_permutation", or "ab_orthogonal"

# Reuse an existing run dir (skip training) if provided.
if [[ -n "${result_path}" ]]; then
  cmd+=(--result_path "${result_path}")
fi

cmd+=(--eval_guard_backend "${eval_guard_backends[@]}")

if [[ "${enable_mmlu_eval}" == "true" ]]; then
  cmd+=(--eval_mmlu)
fi

if [[ "${MMLU_STORE_PREDICTIONS}" == "true" ]]; then
  cmd+=(--mmlu_store_predictions)
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
