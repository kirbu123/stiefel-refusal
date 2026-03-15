export HF_TOKEN="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"

# LlamaGuard
# python evaluate_answers.py --evaluator llamaguard --cuda-device 1

# DeepInfra API 
export DEEPINFRA_API_KEY="pNj4rZVd3yHXUPkvia2UySyN19mNADFT"
python evaluate_answers.py --evaluator deepinfra --model deepseek-ai/DeepSeek-V3.2

# python evaluate_answers.py
