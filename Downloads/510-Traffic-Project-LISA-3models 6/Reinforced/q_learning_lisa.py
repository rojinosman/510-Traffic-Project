from __future__ import annotations

import argparse

# Allow running this file directly (e.g., `python Reinforced/q_learning_lisa.py ...`)
# without needing to set PYTHONPATH.
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from common.data_utils import load_npz, split_data
from common.traffic_metrics import compute_move_hold_rates


class QBandit:
    """Tabular Q-learning on discretized features.

    We keep the *algorithm family* from the original project (Q-learning), but to use
    the LISA image dataset we treat classification as a one-step RL problem:

      state  = discretized feature vector from a cropped traffic light image
      action = predicted class (traffic light state/tag)
      reward = 1 if correct else 0

    This is a contextual bandit (gamma=0), trained with epsilon-greedy exploration.
    """

    def __init__(
        self,
        n_actions: int,
        n_bins: int = 6,
        alpha: float = 0.2,
        gamma: float = 0.0,
        eps: float = 0.2,
        eps_min: float = 0.02,
        eps_decay: float = 0.995,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.bin_edges: list[np.ndarray] | None = None

    def fit_binner(self, X: np.ndarray) -> None:
        edges = []
        qs = np.linspace(0, 1, self.n_bins + 1)
        for j in range(X.shape[1]):
            col = X[:, j]
            # quantiles; jitter for constant columns
            e = np.quantile(col + 1e-12 * self.rng.standard_normal(size=col.shape), qs)
            e = np.unique(e)
            if len(e) < 3:
                e = np.array([col.min() - 1e-6, col.max() + 1e-6], dtype=np.float32)
            edges.append(e.astype(np.float32))
        self.bin_edges = edges

    def _state(self, x: np.ndarray) -> tuple[int, ...]:
        assert self.bin_edges is not None
        idxs = []
        for j, e in enumerate(self.bin_edges):
            b = int(np.digitize(x[j], e[1:-1], right=False))
            idxs.append(b)
        return tuple(idxs)

    def act(self, s: tuple[int, ...]) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 2, shuffle: bool = True) -> None:
        if self.bin_edges is None:
            self.fit_binner(X)

        idx = np.arange(len(X))
        for _ep in range(epochs):
            if shuffle:
                self.rng.shuffle(idx)
            for i in idx:
                s = self._state(X[i])
                a = self.act(s)
                r = 1.0 if a == int(y[i]) else 0.0
                # one-step update (gamma=0)
                self.Q[s][a] += self.alpha * (r - self.Q[s][a])
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.bin_edges is not None
        yhat = np.zeros(len(X), dtype=np.int64)
        for i in range(len(X)):
            s = self._state(X[i])
            yhat[i] = int(np.argmax(self.Q[s]))
        return yhat


def main() -> None:
    ap = argparse.ArgumentParser(description="Reinforcement model: Q-learning bandit on LISA features")
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3, help="Number of passes over the training set")
    ap.add_argument("--bins", type=int, default=6, help="Quantile bins per feature for discretization")
    args = ap.parse_args()

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    agent = QBandit(n_actions=len(labels), n_bins=args.bins, seed=args.seed)
    agent.train(Xtr_s, ytr, epochs=args.epochs)
    pred = agent.predict(Xte_s)

    rates = compute_move_hold_rates(yte, pred, labels)

    print("=== Q-learning (contextual bandit) on LISA ===")
    print(f"Classes: {len(labels)} | Train/Val/Test: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"Macro-F1 (test): {f1_score(yte, pred, average='macro'):.4f}")
    print(f"MOVE/HOLD proxy: wasted_green={rates.wasted_green_rate:.3f} | illegal_go={rates.illegal_go_rate:.3f} | throughput_factor≈{rates.throughput_factor:.3f}")
    print()
    print(classification_report(yte, pred, target_names=labels, digits=3))


if __name__ == "__main__":
    main()
