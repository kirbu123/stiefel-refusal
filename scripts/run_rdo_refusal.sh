#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."
# variablesallenai/Olmo-3-7B-Instruct
model="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}" # override with MODEL_NAME
direction_mode="${DIRECTION_MODE:-activation_additive_rot}" # override with DIRECTION_MODE
train_guard_val_gap=0 # 0 disables; otherwise run train guard validation every N dataloader iterations
num_opt_layers="${NUM_OPT_LAYERS:-0}" # override with NUM_OPT_LAYERS; 0 means all layers for additive modes
llamaguard_data="basic" # "rdo" (data/<splits>_splits/*_<eval_split>.json) or "basic" (SAVE_DIR/rdo/<model>/basic/targets/)
eval_guard_backends=("llamaguard" "qwen3guard" "wildguard") # any subset of: "llamaguard" "qwen3guard" "wildguard"
lr=1e-5
optimizer="AdamW" # "Adam", "AdamW", "SGD"
max_iters=10000
result_path="" # e.g. results/rdo_refusal/tensorboard/<existing_run_dir> (leave empty to train)
result_root="${RESULT_ROOT:-./results/rdo_refusal/tensorboard}" # root for newly trained run directories
log_steps=0
clear_ckpts=true # delete checkpoint .pt files only after training and final validation succeed
eval_activation_pca="${EVAL_ACTIVATION_PCA:-false}" # true: dump best-layer acts + 3D PCA plot during train
activation_save_gap="${ACTIVATION_SAVE_GAP:-100}" # optimizer-step gap when eval_activation_pca=true
keep_activation_pca_dumps="${KEEP_ACTIVATION_PCA_DUMPS:-false}" # true: keep tmp_best_layer_activations after plot
init_mode="diag_permutation" # "random", "diag_permutation", or "ab_orthogonal"
orth_method="svd" # "qr" or "svd"
k_proj="${K_PROJ:-35}" # direct low-rank projection width; override with K_PROJ
ablation_lambda="${ABLATION_LAMBDA:-1.0}"
addition_lambda="${ADDITION_LAMBDA:-0.0}"
retain_lambda="${RETAIN_LAMBDA:-1.0}"
repetition_lambda="${REPETITION_LAMBDA:-0.0}"

# LlamaGuard eval config
eval_max_new_tokens=256 # inportant param for llama guard eval
eval_batch_size=8
MAX_HARMFUL=200 # 200
MAX_HARMLESS=200 # 200

# MMLU eval config
enable_mmlu_eval=true # true
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

# Locality/capability eval config (independent on/off switches)
enable_ppl_eval=true
enable_arc_easy_eval=true
enable_arc_challenge_eval=true
enable_gsm8k_eval=true
locality_store_predictions=false

# WikiText-2 PPL
ppl_dataset="Salesforce/wikitext"
ppl_subset="wikitext-2-raw-v1"
ppl_split="test"
ppl_max_length=512
ppl_stride=256
ppl_max_windows=100

# ARC-Easy / ARC-Challenge
arc_dataset="allenai/ai2_arc"
arc_split="validation"
arc_sample_size=100
arc_sample_seed=42

# GSM8K
gsm8k_dataset="openai/gsm8k"
gsm8k_subset="main"
gsm8k_split="test"
gsm8k_sample_size=100
gsm8k_sample_seed=42
gsm8k_max_new_tokens=512

export MAX_ITERS="${max_iters}"

# ---- GPU (override: CUDA_VISIBLE_DEVICES=1 ./scripts/run_rdo_refusal.sh) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# ---- Fast defaults (override by exporting before running) ----
# These avoid long runs when eval flags are enabled.
export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-$mmlu_sample_size}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-$mmlu_max_new_tokens}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-$mmlu_store_predictions}"
export PPL_MAX_WINDOWS="${PPL_MAX_WINDOWS:-$ppl_max_windows}"
export PPL_MAX_LENGTH="${PPL_MAX_LENGTH:-$ppl_max_length}"
export PPL_STRIDE="${PPL_STRIDE:-$ppl_stride}"
export ARC_SAMPLE_SIZE="${ARC_SAMPLE_SIZE:-$arc_sample_size}"
export GSM8K_SAMPLE_SIZE="${GSM8K_SAMPLE_SIZE:-$gsm8k_sample_size}"
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-$gsm8k_max_new_tokens}"

# LlamaGuard: we only need the first generated token to compute P(unsafe).
export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-100}"
# Optional guard-generation micro-batch overrides (reduce to avoid CUDA OOM):
export GUARD_EVAL_BATCH_SIZE="${GUARD_EVAL_BATCH_SIZE:-$eval_batch_size}"
export GUARD_TRAIN_BATCH_SIZE="${GUARD_TRAIN_BATCH_SIZE:-$eval_batch_size}"

# LlamaGuard eval caps (total counts, not per-category).
export MAX_HARMFUL="${MAX_HARMFUL}" # 250
export MAX_HARMLESS="${MAX_HARMLESS}" # 250

export HF_TOKEN="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"

# ---- Default command ----
cmd=(python -m baselines.rdo_refusal \
  --train_direction \
  --model "${model}" \
  --direction_mode "${direction_mode}" \
  --llamaguard_data "${llamaguard_data}" \
  --lr "${lr}" \
  --optimizer "${optimizer}" \
  --eval_llamaguard \
  --num_opt_layers "${num_opt_layers}" \
  --k_proj "${k_proj}" \
  --init_mode "${init_mode}" \
  --orth_method "${orth_method}" \
  --log_steps "${log_steps}" \
  --train_guard_val_gap "${train_guard_val_gap}" \
  --result_root "${result_root}" \
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
  --ppl_dataset "${ppl_dataset}" \
  --ppl_subset "${ppl_subset}" \
  --ppl_split "${ppl_split}" \
  --ppl_max_length "${PPL_MAX_LENGTH}" \
  --ppl_stride "${PPL_STRIDE}" \
  --ppl_max_windows "${PPL_MAX_WINDOWS}" \
  --arc_dataset "${arc_dataset}" \
  --arc_split "${arc_split}" \
  --arc_sample_size "${ARC_SAMPLE_SIZE}" \
  --arc_sample_seed "${arc_sample_seed}" \
  --gsm8k_dataset "${gsm8k_dataset}" \
  --gsm8k_subset "${gsm8k_subset}" \
  --gsm8k_split "${gsm8k_split}" \
  --gsm8k_sample_size "${GSM8K_SAMPLE_SIZE}" \
  --gsm8k_sample_seed "${gsm8k_sample_seed}" \
  --gsm8k_max_new_tokens "${GSM8K_MAX_NEW_TOKENS}" \
  --ablation_lambda "${ablation_lambda}" \
  --addition_lambda "${addition_lambda}" \
  --retain_lambda "${retain_lambda}" \
  --retain_loss \
  --repetition_lambda "${repetition_lambda}" \
  # --retain_lambda 0.0 \
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

if [[ "${enable_ppl_eval}" == "true" ]]; then
  cmd+=(--eval_ppl)
fi

if [[ "${enable_arc_easy_eval}" == "true" ]]; then
  cmd+=(--eval_arc_easy)
fi

if [[ "${enable_arc_challenge_eval}" == "true" ]]; then
  cmd+=(--eval_arc_challenge)
fi

if [[ "${enable_gsm8k_eval}" == "true" ]]; then
  cmd+=(--eval_gsm8k)
fi

if [[ "${MMLU_STORE_PREDICTIONS}" == "true" ]]; then
  cmd+=(--mmlu_store_predictions)
fi

if [[ "${locality_store_predictions}" == "true" ]]; then
  cmd+=(--locality_store_predictions)
fi

if [[ "${clear_ckpts}" == "true" ]]; then
  cmd+=(--clear_ckpts)
fi

if [[ "${eval_activation_pca}" == "true" ]]; then
  cmd+=(--eval_activation_pca --activation_save_gap "${activation_save_gap}")
  if [[ "${keep_activation_pca_dumps}" == "true" ]]; then
    cmd+=(--keep_activation_pca_dumps)
  fi
fi

# Optional eval flags:
# cmd+=(--eval_mmlu)
# cmd+=(--mmlu_store_predictions)

"${cmd[@]}"

# ---- Optional eval flags ----
# Add these if you want metrics after training:
#   --eval_llamaguard --eval_mmlu --eval_ppl
#   --eval_arc_easy --eval_arc_challenge --eval_gsm8k
#
# LlamaGuard requires HF access if using a gated model:
# export HF_TOKEN="..."
# export LLAMAGUARD_MODEL_NAME="meta-llama/Meta-Llama-Guard-2-8B"
# export LLAMAGUARD_DEVICE="cuda"
# export LLAMAGUARD_DTYPE="bfloat16"
