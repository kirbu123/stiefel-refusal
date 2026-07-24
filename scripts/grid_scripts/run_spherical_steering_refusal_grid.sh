#!/bin/bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")/../.."

# GPU used for every sequential run.
# Override with: CUDA_VISIBLE_DEVICES=1 ./scripts/grid_scripts/run_spherical_steering_refusal_grid.sh
CUDA_VISIBLE_DEVICES=3
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

# Root directory where this grid stores experiment run directories.
# Override with: RESULT_ROOT=/path/to/runs ./scripts/grid_scripts/run_spherical_steering_refusal_grid.sh
RESULT_ROOT="/home/user1/buka2004/LLM-Attack-Defense/results/rdo_refusal/AAAI-results/paper_spherical_steering"

result_root="${RESULT_ROOT:-./results/rdo_refusal/tensorboard}"
if [[ -z "${result_root}" ]]; then
  echo "Invalid empty result root" >&2
  exit 2
fi

# Models only — paper spherical uses fixed attack/protect defaults (no κ/α/β sweep).
model_values=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "tiiuae/Falcon3-7B-Base"
  "Qwen/Qwen3-8B"
  "CWRUSafetyLab/Qwen2.5-1.5B-Instruct-EASE"
  "allenai/Olmo-3-7B-Instruct"
)

for model_name in "${model_values[@]}"; do
  if [[ -z "${model_name}" ]]; then
    echo "Invalid empty model value" >&2
    exit 2
  fi

  echo "=== Starting paper_spherical_steering run with model=${model_name} on GPU ${cuda_visible_devices} ==="
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    RESULT_ROOT="${result_root}" \
    MODEL_NAME="${model_name}" \
    ./scripts/run_spherical_steering_refusal.sh
  echo "=== Completed paper_spherical_steering run with model=${model_name} ==="
done
