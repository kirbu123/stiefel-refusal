"""CLI entry for upstream paper angular-steering with RDO eval/logging.

Usage (from repo root):
  python -m baselines.angular_steering_refusal --model tiiuae/Falcon3-7B-Base --train_direction ...

This injects ``--direction_mode paper_angular_steering`` (and ``--train_direction`` if
missing) then runs the shared ``baselines.rdo_refusal`` pipeline: SaladBench plane
extract (no DIM), fixed attack/protect angles, identical experiment logging.
"""

from __future__ import annotations

import os
import sys


def _ensure_flag(flag: str, value: str | None = None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def _inject_defaults() -> None:
    # rdo_refusal reads MAX_ITERS at import time.
    os.environ.setdefault("MAX_ITERS", "10000")
    os.environ.setdefault("SAVE_DIR", "./results/rdo_refusal")
    os.environ.setdefault("DIM_DIR", "dim")

    # Force paper mode even if a different --direction_mode was passed earlier.
    if "--direction_mode" in sys.argv:
        idx = sys.argv.index("--direction_mode")
        if idx + 1 < len(sys.argv):
            sys.argv[idx + 1] = "paper_angular_steering"
        else:
            sys.argv.append("paper_angular_steering")
    else:
        _ensure_flag("--direction_mode", "paper_angular_steering")

    _ensure_flag("--train_direction")
    # Unused RDO knobs kept for identical dirname / hparams contract.
    _ensure_flag("--num_opt_layers", "1")
    _ensure_flag("--k_proj", "35")
    _ensure_flag("--init_mode", "diag_permutation")
    _ensure_flag("--orth_method", "svd")


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        # Avoid injecting --train_direction so --help only prints argparse text.
        os.environ.setdefault("MAX_ITERS", "10000")
        _ensure_flag("--direction_mode", "paper_angular_steering")
    else:
        _inject_defaults()
    # Importing rdo_refusal parses argv and runs the selected train_* entrypoints.
    import baselines.rdo_refusal  # noqa: F401


if __name__ == "__main__":
    main()
