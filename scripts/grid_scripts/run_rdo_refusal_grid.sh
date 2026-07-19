#!/bin/bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")/../.."

# GPU used for every sequential run.
# Override with: CUDA_VISIBLE_DEVICES=1 ./scripts/grid_scripts/run_rdo_refusal_grid.sh
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-6}"

# Positional arguments override the default k_proj grid.

k_proj_values=(10 175 300 225)

for k_proj in "${k_proj_values[@]}"; do
  if ! [[ "${k_proj}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid k_proj value: ${k_proj} (expected a positive integer)" >&2
    exit 2
  fi

  echo "=== Starting RDO run with k_proj=${k_proj} on GPU ${cuda_visible_devices} ==="
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    K_PROJ="${k_proj}" \
    ./scripts/run_rdo_refusal.sh
  echo "=== Completed RDO run with k_proj=${k_proj} ==="
done
