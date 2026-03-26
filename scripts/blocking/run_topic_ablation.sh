#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/../.."

if [ -n "${MODEL_NAME:-}" ]; then
  python -m cli.run_config --method topic_ablation --config configs/blocking/topic_ablation.toml --model "$MODEL_NAME"
else
  python -m cli.run_config --method topic_ablation --config configs/blocking/topic_ablation.toml
fi
