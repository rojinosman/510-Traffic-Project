from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Mapping from LISA traffic light state labels -> whether vehicles are allowed to move.
# If your dataset uses different strings, update these sets.
MOVE_LABELS = {"go", "goforward", "goleft"}
HOLD_LABELS = {"stop", "stopleft", "warning", "warningleft"}


@dataclass(frozen=True)
class MoveHoldRates:
    wasted_green_rate: float
    illegal_go_rate: float
    throughput_factor: float  # simple proxy: 1 - wasted_green_rate

    n_move_true: int
    n_hold_true: int


def _norm(label: str) -> str:
    return label.strip().replace(" ", "").lower()


def move_mask_from_labels(labels: list[str]) -> np.ndarray:
    """Return boolean mask of shape (K,) indicating which class indices mean MOVE.

    Any class that isn't explicitly in MOVE_LABELS is treated as HOLD for safety.
    """
    m = np.zeros(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        if _norm(lab) in MOVE_LABELS:
            m[i] = True
    return m


def compute_move_hold_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> MoveHoldRates:
    """Compute proxy rates that connect classification errors to throughput/queue impacts.

    - wasted_green_rate = P(pred=HOLD | true=MOVE)
      Interpretable as fraction of 'move-allowed' moments that the system would incorrectly hold,
      wasting green opportunity -> lowers throughput, increases queue.

    - illegal_go_rate = P(pred=MOVE | true=HOLD)
      Safety-critical mistake; included for completeness.

    throughput_factor is a simple proxy for throughput relative to perfect perception:
        throughput_factor ≈ 1 - wasted_green_rate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    move_mask = move_mask_from_labels(labels)
    true_move = move_mask[y_true]
    pred_move = move_mask[y_pred]

    n_move = int(true_move.sum())
    n_hold = int((~true_move).sum())

    # Avoid division by zero in degenerate cases.
    wasted = float(((true_move) & (~pred_move)).sum()) / (n_move if n_move else 1)
    illegal = float(((~true_move) & (pred_move)).sum()) / (n_hold if n_hold else 1)

    return MoveHoldRates(
        wasted_green_rate=wasted,
        illegal_go_rate=illegal,
        throughput_factor=1.0 - wasted,
        n_move_true=n_move,
        n_hold_true=n_hold,
    )
