#!/bin/bash
set -euo pipefail

# Run from repo root. scripts/exp_scrits -> ../..
cd "$(dirname "$0")/../.."

model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
proj_reduce_ratio=10 # factor rank k = hidden_size // proj_reduce_ratio
# Override from the environment, e.g. CUDA_VISIBLE_DEVICES=0 ./scripts/exp_scrits/run_shtiefel_proj_rot_efficacy_grid.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=3
fi

for n in 1 2 3 4 5 7 10 12 15 20; do
  echo "[grid] shtiefel_proj_rot optimized_layers=${n}"
  python -m baselines.rdo_refusal \
    --model "${model}" \
    --direction_mode shtiefel_proj_rot \
    --init_mode diag_permutation \
    --proj_reduce_ratio "${proj_reduce_ratio}" \
    --optimized_layers "${n}" \
    --eval_efficacy
done
