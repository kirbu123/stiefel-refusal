#!/bin/bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")/../.."

# GPU used for every sequential run.
# Override with: CUDA_VISIBLE_DEVICES=1 ./scripts/grid_scripts/run_rdo_refusal_grid.sh
CUDA_VISIBLE_DEVICES=0
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

# Root directory where this grid stores TensorBoard experiment run directories.
# Override with: RESULT_ROOT=/path/to/tensorboard ./scripts/grid_scripts/run_rdo_refusal_grid.sh

RESULT_ROOT="/home/user1/buka2004/LLM-Attack-Defense/results/rdo_refusal/AAAI-results/baseline/nol-ablations/falcon-7b-base"

result_root="${RESULT_ROOT:-./results/rdo_refusal/tensorboard}"
if [[ -z "${result_root}" ]]; then
  echo "Invalid empty result root" >&2
  exit 2
fi

# Cartesian-product grid: every model and mode runs with every k_proj and nol value.
# "Qwen/Qwen3-8B" "tiiuae/Falcon3-7B-Base" "huihui-ai/Huihui-Qwen3.5-9B-abliterated" "CWRUSafetyLab/Qwen2.5-1.5B-Instruct-EASE" "allenai/Olmo-3-7B-Instruct" "allenai/OLMo-2-0425-1B-Instruct" "Rootkit7/Qwen3-8B-abliterated" "allenai/Olmo-3-1025-7B" "Qwen/Qwen3-8B-Base" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_values=(
  "tiiuae/Falcon3-7B-Base"
)
direction_mode_values=(baseline) # baseline activation_additive_rot shtiefel_additive_rot
k_proj_values=(35)
n_of_layers_values=(1 3 4 5 7 12 15 20)

# Loss weights (defaults match scripts/run_rdo_refusal.sh).
ADDITION_LAMBDA=0.2

ablation_lambda="${ABLATION_LAMBDA:-1.0}"
addition_lambda="${ADDITION_LAMBDA:-0.0}"
retain_lambda="${RETAIN_LAMBDA:-1.0}"
repetition_lambda="${REPETITION_LAMBDA:-0.0}"

for model_name in "${model_values[@]}"; do
  if [[ -z "${model_name}" ]]; then
    echo "Invalid empty model value" >&2
    exit 2
  fi

  for direction_mode in "${direction_mode_values[@]}"; do
    case "${direction_mode}" in
      baseline|activation_additive_rot|shtiefel_additive_rot) ;;
      *)
        echo "Invalid direction_mode value: ${direction_mode}" >&2
        exit 2
        ;;
    esac

    for k_proj in "${k_proj_values[@]}"; do
      if ! [[ "${k_proj}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid k_proj value: ${k_proj} (expected a positive integer)" >&2
        exit 2
      fi

      for n_of_layers in "${n_of_layers_values[@]}"; do
        if ! [[ "${n_of_layers}" =~ ^[0-9]+$ ]]; then
          echo "Invalid nol value: ${n_of_layers} (expected a non-negative integer)" >&2
          exit 2
        fi

        echo "=== Starting RDO run with model=${model_name}, mode=${direction_mode}, k_proj=${k_proj}, nol=${n_of_layers} on GPU ${cuda_visible_devices} ==="
        CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
          RESULT_ROOT="${result_root}" \
          ABLATION_LAMBDA="${ablation_lambda}" \
          ADDITION_LAMBDA="${addition_lambda}" \
          RETAIN_LAMBDA="${retain_lambda}" \
          REPETITION_LAMBDA="${repetition_lambda}" \
          MODEL_NAME="${model_name}" \
          DIRECTION_MODE="${direction_mode}" \
          K_PROJ="${k_proj}" \
          NUM_OPT_LAYERS="${n_of_layers}" \
          ./scripts/run_rdo_refusal.sh
        echo "=== Completed RDO run with model=${model_name}, mode=${direction_mode}, k_proj=${k_proj}, nol=${n_of_layers} ==="
      done
    done
  done
done
