#!/bin/bash
export WANDB_ENTITY="andrey"

set -euo pipefail

CATEGORY_DATASET_SOURCE="${CATEGORY_DATASET_SOURCE:-combined}"
CATEGORY_FILTER="${CATEGORY_FILTER:-Physical harm}"

usage() {
  echo "Usage: $0 [--category-dataset-source combined|jailbreakbench] [--category <name>]" >&2
}

validate_category_dataset_source() {
  case "$1" in
    combined|jailbreakbench) ;;
    *)
      echo "Error: invalid category dataset source '$1'." >&2
      echo "Valid values: combined, jailbreakbench" >&2
      exit 1
      ;;
  esac
}

validate_category() {
  case "$1" in
    "Harassment/Discrimination"|"Malware/Hacking"|"Physical harm"|"Economic harm"|"Fraud/Deception"|"Disinformation"|"Sexual/Adult content"|"Privacy"|"Expert advice"|"Government decision-making") ;;
    *)
      echo "Error: invalid category '$1'." >&2
      echo "Valid values: Harassment/Discrimination, Malware/Hacking, Physical harm, Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content, Privacy, Expert advice, Government decision-making" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --category-dataset-source)
      if [[ $# -lt 2 ]]; then
        echo "Error: --category-dataset-source requires a value." >&2
        usage
        exit 1
      fi
      CATEGORY_DATASET_SOURCE="$2"
      shift 2
      ;;
    --category)
      if [[ $# -lt 2 ]]; then
        echo "Error: --category requires a value." >&2
        usage
        exit 1
      fi
      CATEGORY_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage
      exit 1
      ;;
  esac
done

validate_category_dataset_source "$CATEGORY_DATASET_SOURCE"
validate_category "$CATEGORY_FILTER"

export CATEGORY_DATASET_SOURCE
export CATEGORY_FILTER

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/../.."

if [ -n "${MODEL_NAME:-}" ]; then
  python -m cli.run_config --method graph_grpo --config configs/blocking/graph_grpo.toml --model "$MODEL_NAME"
else
  python -m cli.run_config --method graph_grpo --config configs/blocking/graph_grpo.toml
fi
