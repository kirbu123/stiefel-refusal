"""Keep a single detached best operator instead of a per-update history."""

import math
from pathlib import Path

import torch


class BestOperatorCheckpoint:
    def __init__(self, direction_mode):
        self.direction_mode = direction_mode
        self.loss = float("inf")
        self.index = None
        self.artifact = None

    def update(self, loss, operation, index):
        score = float(loss)
        # Keep the first minimum, matching argmin for finite training losses.
        if not math.isfinite(score) or (self.index is not None and score >= self.loss):
            return False
        if self.direction_mode == "baseline":
            artifact = None
        elif self.direction_mode in {"loreft", "direft"}:
            # ReFT exports a detached, independently copied CPU state dict.
            artifact = operation.export_checkpoint()
        else:
            matrix = (operation.checkpoint_matrix() if hasattr(operation, "checkpoint_matrix")
                      else operation.cayley_param)
            artifact = matrix.detach().to(device="cpu", copy=True)
        self.loss, self.index, self.artifact = score, index, artifact
        return True

    def require_best(self):
        if self.index is None:
            raise RuntimeError("No finite optimizer-step loss; cannot select a best checkpoint")

    def save(self, directory):
        self.require_best()
        if self.direction_mode == "baseline":
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = ("lowest_loss_reft.pt" if self.direction_mode in {"loreft", "direft"}
                else "lowest_loss_cayley_param.pt")
        torch.save(self.artifact, directory / name)
