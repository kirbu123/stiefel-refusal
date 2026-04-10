#!/bin/bash
export WANDB_ENTITY="andrey"

set -euo pipefail

CATEGORY_MODE="${CATEGORY_MODE:-single}"
CATEGORY_DATASET_SOURCE="${CATEGORY_DATASET_SOURCE:-combined}"
CATEGORY_FILTER="${CATEGORY_FILTER:-Physical harm}"

usage() {
  echo "Usage: $0 [--category-dataset-source combined|jailbreakbench] [--category <name>] [--all-categories]" >&2
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
      CATEGORY_MODE="single"
      CATEGORY_FILTER="$2"
      shift 2
      ;;
    --all-categories)
      CATEGORY_MODE="all"
      CATEGORY_FILTER=""
      shift
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
if [[ "$CATEGORY_MODE" == "single" ]]; then
  validate_category "$CATEGORY_FILTER"
fi

export CATEGORY_MODE
export CATEGORY_DATASET_SOURCE
export CATEGORY_FILTER

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=3

# ---- Fast smoke debug ----
# When true, graph_grpo uses only a small subset of category questions and a smaller rollout noise scale.
export DEBUG=false
export DEBUG_N_QUESTIONS=4
export DEBUG_NOISE_SCALE=0.02

# ---- Model ----
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export BATCH_SIZE=32

# ---- Graph file (tags for refusal directions) ----
export GRAPH_FILE="graphs/physical harm_wordnet_graph_actions_terms.txt"
export ALL_CATEGORIES_HARMFUL_PROMPT_COUNT="${ALL_CATEGORIES_HARMFUL_PROMPT_COUNT:-128}"
export ALL_CATEGORIES_HARMFUL_PROMPT_SEED="${ALL_CATEGORIES_HARMFUL_PROMPT_SEED:-42}"

# ---- GRPO-IS training hyperparameters ----
export GRPO_N_GROUPS=8
export GRPO_N_EPOCHS=100
export GRPO_LEARNING_RATE=5e-3
export GRPO_NOISE_SCALE=100

# ---- GRPO-IS specific: shared intervention scale for all sampled W rollout policies ----
export GRPO_REF_ALPHA=1.0
export IS_CLIP_RATIO=5.0
export GRPO_CLIP_RATIO=0.2
export GRPO_LOSS_AGG_MODE="token-mean"
export GRPO_KL_LOSS_COEF=0.01

# ---- Abliteration parameters ----
export ABLITERATION_MAX_WEIGHT=2.0
export ABLITERATION_MAX_WEIGHT_POSITION=0.7
export ABLITERATION_MIN_WEIGHT=0.1
export ABLITERATION_MIN_WEIGHT_DISTANCE=0.3

# ---- Weight initialization ----
export WEIGHTS_MODE="scalar"
export WEIGHTS_INIT_TYPE="average"

# ---- Optimizer selection ----
export OPTIMIZER_METHOD="grpo"

# ---- Optuna weight search ----
export OPTUNA_SAMPLER="cmaes"
export OPTUNA_N_TRIALS=50
export OPTUNA_SAMPLER_SEED=42
export OPTUNA_WEIGHT_MIN=-2.0
export OPTUNA_WEIGHT_MAX=2.0

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
export HF_TOKEN="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"

# ---- Locality evaluation ----
export EVALUATE_LOCALITY="true"

# ---- Academic benchmarks ----
export ACADEMIC_BENCHMARKS_ENABLED="${ACADEMIC_BENCHMARKS_ENABLED:-tinyhellaswag,arc,winogrande,gsm8k,truthfulqa}"
export ACADEMIC_BENCHMARKS_SAMPLE_SEED="${ACADEMIC_BENCHMARKS_SAMPLE_SEED:-42}"
export ACADEMIC_BENCHMARKS_STORE_PREDICTIONS="${ACADEMIC_BENCHMARKS_STORE_PREDICTIONS:-true}"
export TINYHELLASWAG_SAMPLE_SIZE="${TINYHELLASWAG_SAMPLE_SIZE:-100}"
export ARC_SAMPLE_SIZE="${ARC_SAMPLE_SIZE:-100}"
export WINOGRANDE_SAMPLE_SIZE="${WINOGRANDE_SAMPLE_SIZE:-100}"
export GSM8K_SAMPLE_SIZE="${GSM8K_SAMPLE_SIZE:-100}"
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
export TRUTHFULQA_SAMPLE_SIZE="${TRUTHFULQA_SAMPLE_SIZE:-100}"

cd "$(dirname "$0")/.."
python -m baselines.graph_grpo
