#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/../.."

if [ -n "${MODEL_NAME:-}" ]; then
  python -m cli.run_config --method graph_average --config configs/blocking/graph_average.toml --model "$MODEL_NAME"
else
  python -m cli.run_config --method graph_average --config configs/blocking/graph_average.toml
fi
