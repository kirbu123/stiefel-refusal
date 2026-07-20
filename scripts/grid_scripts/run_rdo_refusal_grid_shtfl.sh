#!/bin/bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")/../.."

# GPU used for every sequential run.
# Override with: CUDA_VISIBLE_DEVICES=1 ./scripts/grid_scripts/run_rdo_refusal_grid.sh
CUDA_VISIBLE_DEVICES=3
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-2}"

# Cartesian-product grid: every k_proj runs with every nol value.
k_proj_values=(20 35 50 100 175)
n_of_layers_values=(0)

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

    echo "=== Starting RDO run with k_proj=${k_proj}, nol=${n_of_layers} on GPU ${cuda_visible_devices} ==="
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
      K_PROJ="${k_proj}" \
      NUM_OPT_LAYERS="${n_of_layers}" \
      ./scripts/run_rdo_refusal_shtfl.sh
    echo "=== Completed RDO run with k_proj=${k_proj}, nol=${n_of_layers} ==="
  done
done
