#!/bin/bash

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=0

# ---- Model ----
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ---- Grid-search hyperparameters (comma-separated lists) ----
export GRID_MAX_WEIGHT="2.5,3.0"
export GRID_MAX_WEIGHT_POSITION="0.7"
export GRID_MIN_WEIGHT="0.0,1.0"
export GRID_MIN_WEIGHT_DISTANCE="0.3"

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
python -m baselines.topic_ablation
