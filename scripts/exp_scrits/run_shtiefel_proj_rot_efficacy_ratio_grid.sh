#!/bin/bash
set -euo pipefail

# Run from repo root. scripts/exp_scrits -> ../..
cd "$(dirname "$0")/../.."

model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
optimized_layers=1
# Override from the environment, e.g. CUDA_VISIBLE_DEVICES=0 ./scripts/exp_scrits/run_shtiefel_proj_rot_efficacy_ratio_grid.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=3
fi

for r in 100 10 1000 200 500 250 20 50 5 1; do
  echo "[grid] shtiefel_proj_rot optimized_layers=${optimized_layers} proj_reduce_ratio=${r}"
  python -m baselines.rdo_refusal \
    --model "${model}" \
    --direction_mode shtiefel_proj_rot \
    --init_mode diag_permutation \
    --proj_reduce_ratio "${r}" \
    --optimized_layers "${optimized_layers}" \
    --eval_efficacy
done
