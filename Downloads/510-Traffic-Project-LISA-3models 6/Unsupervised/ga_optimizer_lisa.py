from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

from common.data_utils import load_npz, split_data
from common.traffic_metrics import compute_move_hold_rates


@dataclass(frozen=True)
class GABest:
    best_log10: float
    best_val_f1: float


def _eval_log10_var_smoothing(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    log10_vs: float,
) -> float:
    """Train NB with var_smoothing=10**log10_vs and return macro-F1 on validation."""
    model = GaussianNB(var_smoothing=10.0 ** log10_vs)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    return float(f1_score(yva, pred, average="macro"))


def ga_tune_var_smoothing(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    seed: int = 0,
    pop: int = 18,
    gens: int = 10,
    lo: float = -12.0,
    hi: float = -2.0,
    elite_frac: float = 0.25,
    mut_sigma: float = 0.35,
) -> Tuple[float, float]:
    """
    Simple GA that searches log10(var_smoothing) in [lo, hi] to maximize validation macro-F1.
    Returns (best_log10, best_val_f1).
    """
    rng = random.Random(seed)

    # Initialize population uniformly in [lo, hi]
    population = [rng.uniform(lo, hi) for _ in range(pop)]
    elite_k = max(1, int(math.ceil(pop * elite_frac)))

    best_log10 = population[0]
    best_f1 = -1.0

    for _gen in range(gens):
        scores = [(_eval_log10_var_smoothing(Xtr, ytr, Xva, yva, g), g) for g in population]
        scores.sort(reverse=True, key=lambda t: t[0])  # higher is better
        if scores[0][0] > best_f1:
            best_f1, best_log10 = scores[0][0], scores[0][1]

        elites = [g for (_s, g) in scores[:elite_k]]

        # Reproduce: keep elites + fill rest by mutated copies of elite parents
        new_pop = elites[:]
        while len(new_pop) < pop:
            parent = rng.choice(elites)
            child = parent + rng.gauss(0.0, mut_sigma)
            child = max(lo, min(hi, child))
            new_pop.append(child)

        population = new_pop

    return float(best_log10), float(best_f1)


def main() -> None:
    ap = argparse.ArgumentParser(description="GA-tune GaussianNB var_smoothing on LISA features.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=18)
    ap.add_argument("--gens", type=int, default=10)
    args = ap.parse_args()

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    best_log10, best_val_f1 = ga_tune_var_smoothing(Xtr, ytr, Xva, yva, seed=args.seed, pop=args.pop, gens=args.gens)

    model = GaussianNB(var_smoothing=10.0 ** best_log10)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    f1 = float(f1_score(yte, pred, average="macro"))
    rates = compute_move_hold_rates(yte, pred, labels)

    print("=== GA-tuned GaussianNB on LISA ===")
    print(f"Best val macro-F1: {best_val_f1:.4f} at log10(var_smoothing)={best_log10:.2f}")
    print(f"Macro-F1 (test):   {f1:.4f}")
    print(
        f"MOVE/HOLD proxy: wasted_green={rates.wasted_green_rate:.3f} | illegal_go={rates.illegal_go_rate:.3f} | "
        f"throughput_factor≈{rates.throughput_factor:.3f}"
    )


if __name__ == "__main__":
    main()

