"""
Utilities for model and learnable weights.
"""

import torch
import torch.nn.functional as F
from typing import List
from heretic.model import Model, AbliterationParameters


class LearnableDirectionWeights(torch.nn.Module):
    """
    Learnable coefficients for weighted sum of refusal directions.
    One coefficient matrix (n_layers+1, hidden_size) per direction, multiplied element-wise then summed.
    """
    def __init__(
        self, 
        n_directions: int, 
        n_layers: int, 
        hidden_size: int, 
        init_scale: float = 0.1,
        init_type: str = "zero",
        topic_idx: int = 0
    ):
        """
        Args:
            n_directions: Number of refusal directions
            n_layers: Number of layers (excluding embeddings)
            hidden_size: Hidden size
            init_scale: Init scale (for random init only)
            init_type: "zero", "topic", or "average"
            topic_idx: Direction index for "topic" init
        """
        super().__init__()
        self.n_directions = n_directions
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        if init_type == "zero":
            init_weights = torch.zeros(n_directions, n_layers + 1, hidden_size)
        elif init_type == "topic":
            init_weights = torch.zeros(n_directions, n_layers + 1, hidden_size)
            init_weights[topic_idx] = torch.ones(n_layers + 1, hidden_size)
        elif init_type == "average":
            init_weights = torch.ones(n_directions, n_layers + 1, hidden_size) / n_directions
        else:
            init_weights = torch.randn(n_directions, n_layers + 1, hidden_size) * init_scale
        
        self.weights = torch.nn.Parameter(init_weights)
    
    def forward(self, refusal_directions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of refusal directions.
        
        Args:
            refusal_directions: List of tensors (n_layers+1, hidden_size)
        
        Returns:
            Weighted sum (n_layers+1, hidden_size)
        """
        directions_tensor = torch.stack(refusal_directions, dim=0)
        weighted_directions = self.weights * directions_tensor
        combined_direction = weighted_directions.sum(dim=0)
        combined_direction = F.normalize(combined_direction, p=2, dim=1)
        
        return combined_direction


def apply_abliteration_with_hyperparams(
    model: Model,
    refusal_directions: torch.Tensor,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
    n_layers: int
) -> None:
    """
    Apply abliteration with given hyperparameters.
    
    Args:
        model: Model to modify
        refusal_directions: Refusal directions
        max_weight: Max shift weight
        max_weight_position: Position of max weight (fraction of layers, 0-1)
        min_weight: Min weight
        min_weight_distance: Distance for weight decay (fraction of layers)
        n_layers: Number of layers
    """
    max_weight_pos_abs = max_weight_position * (n_layers - 1)
    min_weight_dist_abs = min_weight_distance * (n_layers - 1)
    
    parameters = {}
    for component in model.get_abliterable_components():
        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_pos_abs,
            min_weight=min_weight,
            min_weight_distance=min_weight_dist_abs
        )
    
    with torch.no_grad():
        model.abliterate(refusal_directions, direction_index=None, parameters=parameters)
