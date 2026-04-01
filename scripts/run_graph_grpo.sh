#!/bin/bash

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

# ---- GRPO-IS training hyperparameters ----
export GRPO_N_GROUPS=4
export GRPO_N_EPOCHS=10
export GRPO_LEARNING_RATE=1e-3
export GRPO_NOISE_SCALE=0.1

# ---- GRPO-IS specific: shared intervention scale for all sampled W rollout policies ----
export GRPO_REF_ALPHA=1.0
export IS_CLIP_RATIO=5.0
export GRPO_CLIP_RATIO=0.2
export GRPO_LOSS_AGG_MODE="token-mean"

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

# ---- Optuna scalar search ----
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

cd "$(dirname "$0")/.."
python -m baselines.graph_grpo
