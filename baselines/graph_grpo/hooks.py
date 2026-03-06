"""
Differentiable abliteration via forward hooks.

Makes the model modification differentiable w.r.t. direction weights
so gradients flow: loss -> log_probs -> hook output -> direction -> LearnableDirectionWeights.
"""

from contextlib import suppress
from typing import Any, List

import torch

from heretic.model import Model


def _create_abliteration_hook(
    direction: torch.Tensor,
    weight: float,
    device: torch.device,
):
    """Create a forward hook that projects output onto direction and subtracts."""
    direction = direction.to(device)

    def hook(module, input, output):
        # output shape: (..., hidden_size), direction: (hidden_size,)
        # projection: (output · d) * d
        dots = (output * direction).sum(dim=-1, keepdim=True)
        projection = dots * direction
        return output - weight * projection

    return hook


def _get_modules_for_hooks(model: Model) -> List[tuple]:
    """
    Get (module, layer_index) for each abliterable component.
    Mirrors heretic's get_layer_matrices structure.
    """
    modules_list = []
    layers = model.get_layers()
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        # attn.o_proj
        try:
            modules_list.append((layer.self_attn.o_proj, layer_index))
        except Exception:
            pass
        # mlp.down_proj (dense)
        with suppress(Exception):
            modules_list.append((layer.mlp.down_proj, layer_index))
        # MoE experts
        with suppress(Exception):
            for expert in layer.mlp.experts:
                modules_list.append((expert.down_proj, layer_index))
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                modules_list.append((expert.w2, layer_index))
        with suppress(Exception):
            modules_list.append((layer.mlp.experts.down_proj, layer_index))
        with suppress(Exception):
            modules_list.append((layer.shared_mlp.output_linear, layer_index))
        with suppress(Exception):
            for expert in layer.moe.experts:
                modules_list.append((expert.output_linear, layer_index))
    return modules_list


def register_abliteration_hooks(
    model: Model,
    combined_direction: torch.Tensor,
    alpha: float,
    n_layers: int,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> List:
    """
    Register forward hooks for differentiable abliteration.

    Args:
        model: Heretic model
        combined_direction: (n_layers+1, hidden_size) - direction per layer
        alpha: scalar coefficient scaling the abliteration
        n_layers: number of transformer layers
        max_weight, max_weight_position, min_weight, min_weight_distance: abliteration params

    Returns:
        List of hook handles for removal.
    """
    max_weight_pos_abs = max_weight_position * (n_layers - 1)
    min_weight_dist_abs = min_weight_distance * (n_layers - 1)
    device = next(model.model.parameters()).device

    handles = []
    modules_list = _get_modules_for_hooks(model)

    for module, layer_index in modules_list:
        distance = abs(layer_index - max_weight_pos_abs)
        if distance > min_weight_dist_abs:
            continue
        layer_weight = max_weight + (distance / min_weight_dist_abs) * (min_weight - max_weight)
        effective_weight = alpha * layer_weight
        if effective_weight < 1e-8:
            continue

        # combined_direction[layer_index+1] for layer l (index 0 is embeddings)
        direction = combined_direction[layer_index + 1]
        hook_fn = _create_abliteration_hook(direction, effective_weight, device)
        handle = module.register_forward_hook(hook_fn)
        handles.append(handle)

    return handles


def remove_hooks(handles: List) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
