from __future__ import annotations

import argparse

# Allow running this file directly (e.g., `python NeuralNetwork/mlp_lisa.py --features lisa_features.npz`)
# without needing to set PYTHONPATH.
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common.data_utils import load_npz, split_data
from common.traffic_metrics import compute_move_hold_rates


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Neural network model: MLPClassifier on LISA traffic-light crops"
    )
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hidden",
        type=str,
        default="256,128",
        help="Comma-separated hidden layer sizes (e.g., 256,128)",
    )
    ap.add_argument("--max-iter", type=int, default=40, help="Max MLP training iterations")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    if not hidden:
        raise ValueError("--hidden must contain at least one integer, e.g. 128 or 256,128")

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # scikit-learn MLPClassifier is a feed-forward neural network (multilayer perceptron).
    # early_stopping=True automatically holds out a validation subset from the training split.
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=args.batch_size,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=False,
    )
    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)

    rates = compute_move_hold_rates(yte, pred, labels)

    print("=== Neural Network (MLPClassifier) on LISA ===")
    print(f"Hidden layers: {hidden} | Max iter: {args.max_iter} | Batch: {args.batch_size}")
    print(f"Classes: {len(labels)} | Train/Val/Test: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"Macro-F1 (test): {f1_score(yte, pred, average='macro'):.4f}")
    print(f\"MOVE/HOLD proxy: wasted_green={rates.wasted_green_rate:.3f} | illegal_go={rates.illegal_go_rate:.3f} | throughput_factor≈{rates.throughput_factor:.3f}\")
    print()
    print(classification_report(yte, pred, target_names=labels, digits=3))


if __name__ == "__main__":
    main()
