#!/bin/bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

model="${MODEL_NAME:-tiiuae/Falcon3-7B-Base}"
# Fixed paper-spherical defaults (not swept).
spherical_kappa="${SPHERICAL_KAPPA:-20}"
spherical_alpha="${SPHERICAL_ALPHA:-0.7}"
spherical_beta="${SPHERICAL_BETA:-0.1}"
spherical_n_extract_samples="${SPHERICAL_N_EXTRACT_SAMPLES:-512}"
spherical_extract_batch_size="${SPHERICAL_EXTRACT_BATCH_SIZE:-8}"

train_guard_val_gap=0
# Unused RDO knobs kept for identical dirname / hparams contract.
num_opt_layers="${NUM_OPT_LAYERS:-1}"
llamaguard_data="basic"
eval_guard_backends=("llamaguard" "qwen3guard" "wildguard")
lr=1e-5
optimizer="AdamW"
max_iters=10000
result_path=""
result_root="${RESULT_ROOT:-./results/rdo_refusal/tensorboard}"
log_steps=0
clear_ckpts=true
init_mode="diag_permutation"
orth_method="svd"
k_proj="${K_PROJ:-35}"
ablation_lambda="${ABLATION_LAMBDA:-1.0}"
addition_lambda="${ADDITION_LAMBDA:-0.0}"
retain_lambda="${RETAIN_LAMBDA:-1.0}"
repetition_lambda="${REPETITION_LAMBDA:-0.0}"

eval_max_new_tokens=256
eval_batch_size=8
MAX_HARMFUL=200
MAX_HARMLESS=200

enable_mmlu_eval=true
mmlu_dataset="cais/mmlu"
mmlu_subset="all"
mmlu_split="test"
mmlu_mode="zero_shot"
mmlu_answer_mode="generate"
mmlu_n_shots=5
mmlu_sample_size=300
mmlu_sample_seed=42
mmlu_max_new_tokens=8
mmlu_store_predictions=false

enable_ppl_eval=true
enable_arc_easy_eval=true
enable_arc_challenge_eval=true
enable_gsm8k_eval=true
locality_store_predictions=false

ppl_dataset="Salesforce/wikitext"
ppl_subset="wikitext-2-raw-v1"
ppl_split="test"
ppl_max_length=512
ppl_stride=256
ppl_max_windows=100

arc_dataset="allenai/ai2_arc"
arc_split="validation"
arc_sample_size=100
arc_sample_seed=42

gsm8k_dataset="openai/gsm8k"
gsm8k_subset="main"
gsm8k_split="test"
gsm8k_sample_size=100
gsm8k_sample_seed=42
gsm8k_max_new_tokens=512

export MAX_ITERS="${max_iters}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MMLU_SAMPLE_SIZE="${MMLU_SAMPLE_SIZE:-$mmlu_sample_size}"
export MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-$mmlu_max_new_tokens}"
export MMLU_STORE_PREDICTIONS="${MMLU_STORE_PREDICTIONS:-$mmlu_store_predictions}"
export PPL_MAX_WINDOWS="${PPL_MAX_WINDOWS:-$ppl_max_windows}"
export PPL_MAX_LENGTH="${PPL_MAX_LENGTH:-$ppl_max_length}"
export PPL_STRIDE="${PPL_STRIDE:-$ppl_stride}"
export ARC_SAMPLE_SIZE="${ARC_SAMPLE_SIZE:-$arc_sample_size}"
export GSM8K_SAMPLE_SIZE="${GSM8K_SAMPLE_SIZE:-$gsm8k_sample_size}"
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-$gsm8k_max_new_tokens}"

export LLAMAGUARD_MAX_NEW_TOKENS="${LLAMAGUARD_MAX_NEW_TOKENS:-100}"
export GUARD_EVAL_BATCH_SIZE="${GUARD_EVAL_BATCH_SIZE:-$eval_batch_size}"
export GUARD_TRAIN_BATCH_SIZE="${GUARD_TRAIN_BATCH_SIZE:-$eval_batch_size}"
export MAX_HARMFUL="${MAX_HARMFUL}"
export MAX_HARMLESS="${MAX_HARMLESS}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

cmd=(python -m baselines.spherical_steering_refusal \
  --train_direction \
  --model "${model}" \
  --direction_mode paper_spherical_steering \
  --spherical_kappa "${spherical_kappa}" \
  --spherical_alpha "${spherical_alpha}" \
  --spherical_beta "${spherical_beta}" \
  --spherical_n_extract_samples "${spherical_n_extract_samples}" \
  --spherical_extract_batch_size "${spherical_extract_batch_size}" \
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
  --repetition_lambda "${repetition_lambda}"
)

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

echo "=== paper_spherical_steering model=${model} kappa=${spherical_kappa} alpha=${spherical_alpha} beta=${spherical_beta} ==="
"${cmd[@]}"
