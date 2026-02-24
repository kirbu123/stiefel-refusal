#!/bin/bash

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=0

# ---- Model ----
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ---- Graph file (tags for refusal directions) ----
export GRAPH_FILE="graph_build/Physical harm_graph.txt"

# ---- GRPO training hyperparameters ----
export GRPO_N_GROUPS=4
export GRPO_N_EPOCHS=50
export GRPO_LEARNING_RATE=1e-3
export GRPO_NOISE_SCALE=1.0
export GRPO_BETA=0.1
export GRPO_GRADIENT_SCALE=1e6

# ---- Abliteration parameters ----
export ABLITERATION_MAX_WEIGHT=2.0
export ABLITERATION_MAX_WEIGHT_POSITION=0.7
export ABLITERATION_MIN_WEIGHT=0.1
export ABLITERATION_MIN_WEIGHT_DISTANCE=0.3

# ---- Weight initialization ----
export WEIGHTS_INIT_TYPE="zero"

# ---- Evaluation backend ----
# Options: local_llm_judge | llamaguard
export EVALUATION_BACKEND="local_llm_judge"

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
python -m baselines.graph_grpo_old
