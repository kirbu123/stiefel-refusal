"""
Full-sequence log-probability computation.

Heretic's get_logprobs() only returns log-probs for 1 token.
This module computes per-token log-probs for entire (prompt, response) sequences.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_sequence_log_probs(
    model,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log-probabilities for (prompt, response) pairs under current model state.

    Args:
        model: Heretic Model (with model.model = HF model, model.tokenizer, model.get_chat)
        prompts: List of prompt strings
        responses: List of response strings (same length as prompts)
        batch_size: Batch size for forward passes

    Returns:
        log_probs: (total_responses, max_response_len) - per-token log-probs, 0 for padding
        response_mask: (total_responses, max_response_len) - 1 for valid tokens, 0 for padding
    """
    tokenizer = model.tokenizer
    hf_model = model.model
    device = hf_model.device

    all_log_probs = []
    all_masks = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]

        batch_log_probs = []
        batch_masks = []

        for prompt, response in zip(batch_prompts, batch_responses):
            chat = model.get_chat(prompt)
            prompt_text = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            full_text = prompt_text + response

            prompt_enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            full_enc = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=hf_model.config.max_position_embeddings,
            )

            prompt_len = prompt_enc["input_ids"].shape[1]
            full_ids = full_enc["input_ids"][0]
            response_len = full_ids.shape[0] - prompt_len

            if response_len <= 0:
                batch_log_probs.append(torch.zeros(1, device=device))
                batch_masks.append(torch.zeros(1, device=device))
                continue

            input_ids = full_ids.unsqueeze(0).to(device)
            outputs = hf_model(input_ids=input_ids)
            logits = outputs.logits

            # logits[t] predicts token at position t+1
            # Response tokens are at positions prompt_len, prompt_len+1, ..., prompt_len+resp_len-1
            # We need logits at prompt_len-1, prompt_len, ..., prompt_len+resp_len-2
            response_logits = logits[0, prompt_len - 1 : prompt_len + response_len - 1]
            response_tokens = full_ids[prompt_len : prompt_len + response_len].to(device)

            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
            mask = torch.ones(response_len, device=device, dtype=torch.float32)

            batch_log_probs.append(token_log_probs)
            batch_masks.append(mask)

        max_len = max(p.shape[0] for p in batch_log_probs)
        padded_log_probs = []
        padded_masks = []
        for lp, m in zip(batch_log_probs, batch_masks):
            pad_len = max_len - lp.shape[0]
            padded_log_probs.append(F.pad(lp, (0, pad_len), value=0.0))
            padded_masks.append(F.pad(m, (0, pad_len), value=0.0))
        all_log_probs.append(torch.stack(padded_log_probs))
        all_masks.append(torch.stack(padded_masks))

    log_probs = torch.cat(all_log_probs, dim=0)
    response_mask = torch.cat(all_masks, dim=0)
    return log_probs, response_mask
