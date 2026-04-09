#!/bin/bash

set -euo pipefail

CATEGORY_DATASET_SOURCE="${CATEGORY_DATASET_SOURCE:-jailbreakbench}"
CATEGORY_FILTER="${CATEGORY_FILTER:-Physical harm}"
CLI_CATEGORY_DATASET_SOURCE_SET=0
CLI_CATEGORY_SET=0

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
      CLI_CATEGORY_DATASET_SOURCE_SET=1
      shift 2
      ;;
    --category)
      if [[ $# -lt 2 ]]; then
        echo "Error: --category requires a value." >&2
        usage
        exit 1
      fi
      CATEGORY_FILTER="$2"
      CLI_CATEGORY_SET=1
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

if [[ -n "$CATEGORY_FILTER" ]]; then
  validate_category "$CATEGORY_FILTER"
  if [[ "$CLI_CATEGORY_SET" -eq 1 && "$CLI_CATEGORY_DATASET_SOURCE_SET" -eq 0 ]]; then
    echo "Error: --category requires --category-dataset-source jailbreakbench." >&2
    exit 1
  fi
  if [[ "$CATEGORY_DATASET_SOURCE" != "jailbreakbench" ]]; then
    echo "Error: --category requires --category-dataset-source jailbreakbench." >&2
    exit 1
  fi
fi

export CATEGORY_DATASET_SOURCE
export CATEGORY_FILTER

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=3

# ---- Model ----
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ---- Graph file (tags for averaged refusal direction) ----
export GRAPH_FILE="graphs/physical harm_wordnet_graph_actions_terms.txt"

# ---- Grid-search hyperparameters (comma-separated lists) ----
export GRID_MAX_WEIGHT="2.5,3.0"
export GRID_MAX_WEIGHT_POSITION="0.7"
export GRID_MIN_WEIGHT="0.0,1.0"
export GRID_MIN_WEIGHT_DISTANCE="0.3"

# ---- Evaluation backend ----
# Options: local_llm_judge | llamaguard
export EVALUATION_BACKEND="llamaguard"

# ---- Judge / Classifier APIs (used by local_llm_judge) ----
export JUDGE_API_URL="http://localhost:31181/v1/chat/completions"
export CLASSIFIER_API_URL="http://localhost:31180/v1/chat/completions"

# ---- LlamaGuard settings (used by llamaguard) ----
export LLAMAGUARD_MODEL_NAME="meta-llama/Meta-Llama-Guard-2-8B"
export LLAMAGUARD_DEVICE="cuda"
export LLAMAGUARD_DTYPE="bfloat16"
# export HF_TOKEN="<your_hf_token>"

# ---- Locality evaluation ----
export EVALUATE_LOCALITY="true"

cd "$(dirname "$0")/.."
python -m baselines.graph_average
