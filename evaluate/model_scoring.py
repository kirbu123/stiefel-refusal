"""
Shared model-scoring helpers for benchmark evaluation.
"""

from __future__ import annotations

from typing import Any, Iterable

def batchify(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_chat_prompts(model: Any, prompts: list[str]) -> list[str]:
    chats = [model.get_chat(prompt) for prompt in prompts]
    return model.tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_responses(
    model: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    responses: list[str] = []
    batch_size = max(1, int(getattr(model.settings, "batch_size", 1)) or 1)

    for batch in batchify(prompts, batch_size):
        inputs, outputs = model.generate(batch, max_new_tokens=max_new_tokens)
        decoded = model.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        responses.extend(decoded)

    return responses


def score_completion_logprob(model: Any, prompt_text: str, completion_text: str) -> float:
    import torch
    import torch.nn.functional as F

    prompt_inputs = model.tokenizer(
        prompt_text,
        return_tensors="pt",
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    full_inputs = model.tokenizer(
        prompt_text + completion_text,
        return_tensors="pt",
        return_token_type_ids=False,
        add_special_tokens=False,
    ).to(model.model.device)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    candidate_ids = full_inputs["input_ids"][:, prompt_len:]
    if candidate_ids.shape[1] == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model.model(**full_inputs)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    total = 0.0
    start_position = prompt_len - 1
    for step, token_id in enumerate(candidate_ids[0]):
        total += log_probs[0, start_position + step, token_id.item()].item()
    return total


def choice_variants(choice_text: str) -> tuple[str, ...]:
    normalized = str(choice_text)
    variants = [normalized]
    if normalized and not normalized[0].isspace():
        variants.append(f" {normalized}")
    return tuple(dict.fromkeys(variants))


def score_text_choice_variants(
    model: Any,
    prompt_text: str,
    choice_text: str,
    *,
    variants: Iterable[str] | None = None,
) -> float:
    candidate_variants = tuple(variants or choice_variants(choice_text))
    return max(
        score_completion_logprob(model, prompt_text, variant)
        for variant in candidate_variants
    )
