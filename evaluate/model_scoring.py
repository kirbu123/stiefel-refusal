"""
Shared model-scoring helpers for benchmark evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CompletionScore:
    completion_text: str
    logprob: float
    normalized_logprob: float
    token_count: int


@dataclass(frozen=True)
class ChoiceScore:
    best_logprob: float
    best_logprob_variant: str
    best_logprob_token_count: int
    best_normalized_logprob: float
    best_normalized_variant: str
    best_normalized_token_count: int


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


def get_next_token_logprobs(
    model: Any,
    prompts: list[str],
    *,
    token_ids: Iterable[int] | None = None,
    prompts_are_chat_formatted: bool = False,
):
    import torch
    import torch.nn.functional as F

    prompt_texts = prompts
    if not prompts_are_chat_formatted:
        prompt_texts = build_chat_prompts(model, prompts)

    batch_size = max(1, int(getattr(model.settings, "batch_size", 1)) or 1)
    gathered_batches = []
    token_id_list = list(token_ids) if token_ids is not None else None

    if not prompt_texts:
        width = len(token_id_list) if token_id_list is not None else 0
        return torch.empty((0, width), dtype=torch.float32)

    for batch in batchify(prompt_texts, batch_size):
        inputs = model.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(model.model.device)

        attention_mask = inputs.get("attention_mask")
        with torch.no_grad():
            outputs = model.model(**inputs)
            logits = outputs.logits

        if attention_mask is None:
            last_positions = torch.full(
                (logits.shape[0],),
                logits.shape[1] - 1,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            reversed_mask = attention_mask.to(dtype=torch.long).flip(dims=[1])
            last_offsets = torch.argmax(reversed_mask, dim=1)
            last_positions = attention_mask.shape[1] - last_offsets - 1

        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[batch_indices, last_positions, :]
        batch_logprobs = F.log_softmax(next_token_logits, dim=-1)

        if token_id_list is not None:
            gather_indices = torch.tensor(
                token_id_list,
                device=batch_logprobs.device,
                dtype=torch.long,
            )
            batch_logprobs = batch_logprobs.index_select(dim=1, index=gather_indices)

        gathered_batches.append(batch_logprobs.to(torch.float32).cpu())

    return torch.cat(gathered_batches, dim=0)


def score_completion(model: Any, prompt_text: str, completion_text: str) -> CompletionScore:
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
        return CompletionScore(
            completion_text=completion_text,
            logprob=float("-inf"),
            normalized_logprob=float("-inf"),
            token_count=0,
        )

    with torch.no_grad():
        outputs = model.model(**full_inputs)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    total = 0.0
    start_position = prompt_len - 1
    for step, token_id in enumerate(candidate_ids[0]):
        total += log_probs[0, start_position + step, token_id.item()].item()
    token_count = int(candidate_ids.shape[1])
    return CompletionScore(
        completion_text=completion_text,
        logprob=total,
        normalized_logprob=(total / token_count) if token_count else float("-inf"),
        token_count=token_count,
    )


def score_completion_logprob(model: Any, prompt_text: str, completion_text: str) -> float:
    return score_completion(model, prompt_text, completion_text).logprob


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
    return score_text_choice_variants_detailed(
        model,
        prompt_text,
        choice_text,
        variants=variants,
    ).best_logprob


def score_text_choice_variants_detailed(
    model: Any,
    prompt_text: str,
    choice_text: str,
    *,
    variants: Iterable[str] | None = None,
) -> ChoiceScore:
    candidate_variants = tuple(variants or choice_variants(choice_text))

    scored_variants = [
        score_completion(model, prompt_text, variant)
        for variant in candidate_variants
    ]
    best_logprob = max(scored_variants, key=lambda item: item.logprob)
    best_normalized = max(
        scored_variants,
        key=lambda item: item.normalized_logprob,
    )
    return ChoiceScore(
        best_logprob=best_logprob.logprob,
        best_logprob_variant=best_logprob.completion_text,
        best_logprob_token_count=best_logprob.token_count,
        best_normalized_logprob=best_normalized.normalized_logprob,
        best_normalized_variant=best_normalized.completion_text,
        best_normalized_token_count=best_normalized.token_count,
    )
