from __future__ import annotations

import argparse

# Allow running this file directly (e.g., `python Supervised/naive_bayes_lisa.py ...`)
# without needing to set PYTHONPATH.
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from common.data_utils import load_npz, split_data
from common.traffic_metrics import compute_move_hold_rates


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Supervised model: Gaussian Naive Bayes on LISA traffic-light crops"
    )
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = GaussianNB()
    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)

    rates = compute_move_hold_rates(yte, pred, labels)

    print("=== GaussianNB on LISA ===")
    print(f"Classes: {len(labels)} | Train/Val/Test: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"Macro-F1 (test): {f1_score(yte, pred, average='macro'):.4f}")
    print(f\"MOVE/HOLD proxy: wasted_green={rates.wasted_green_rate:.3f} | illegal_go={rates.illegal_go_rate:.3f} | throughput_factor≈{rates.throughput_factor:.3f}\")
    print()
    print(classification_report(yte, pred, target_names=labels, digits=3))


if __name__ == "__main__":
    main()
