from __future__ import annotations

import argparse

from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common.data_utils import load_npz, split_data
from common.traffic_metrics import compute_move_hold_rates
from Reinforced.q_learning_lisa import QBandit
from Unsupervised.ga_optimizer_lisa import ga_tune_var_smoothing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare NB, Q-learning, GA-tuned NB, and an MLP neural network on the same LISA dataset."
    )
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz (from build_lisa_features.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl-epochs", type=int, default=3)
    ap.add_argument("--rl-bins", type=int, default=6)
    ap.add_argument("--ga-pop", type=int, default=18)
    ap.add_argument("--ga-gens", type=int, default=10)
    ap.add_argument("--nn-hidden", type=str, default="256,128", help="MLP hidden layers (comma-separated)")
    ap.add_argument("--nn-max-iter", type=int, default=40, help="MLP max training iterations")
    ap.add_argument("--nn-batch", type=int, default=256, help="MLP batch size")
    args = ap.parse_args()

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # 1) Supervised: GaussianNB
    nb = GaussianNB()
    nb.fit(Xtr_s, ytr)
    pred_nb = nb.predict(Xte_s)
    f1_nb = float(f1_score(yte, pred_nb, average="macro"))
    rates_nb = compute_move_hold_rates(yte, pred_nb, labels)

    # 2) Reinforced: tabular Q-learning (contextual bandit)
    qb = QBandit(n_actions=len(labels), n_bins=args.rl_bins, seed=args.seed)
    qb.train(Xtr_s, ytr, epochs=args.rl_epochs)
    pred_q = qb.predict(Xte_s)
    f1_q = float(f1_score(yte, pred_q, average="macro"))
    rates_q = compute_move_hold_rates(yte, pred_q, labels)

    # 3) GA tuning var_smoothing on validation (same dataset)
    best_log10, best_val_f1 = ga_tune_var_smoothing(
        Xtr_s, ytr, Xva_s, yva, seed=args.seed, pop=args.ga_pop, gens=args.ga_gens
    )
    tuned = GaussianNB(var_smoothing=10.0 ** best_log10)
    tuned.fit(Xtr_s, ytr)
    pred_ga = tuned.predict(Xte_s)
    f1_ga = float(f1_score(yte, pred_ga, average="macro"))
    rates_ga = compute_move_hold_rates(yte, pred_ga, labels)

    # 4) Neural network: MLPClassifier
    hidden = tuple(int(x) for x in args.nn_hidden.split(",") if x.strip())
    if not hidden:
        raise ValueError("--nn-hidden must contain at least one integer, e.g. 128 or 256,128")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=args.nn_batch,
        learning_rate_init=1e-3,
        max_iter=args.nn_max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=False,
    )
    mlp.fit(Xtr_s, ytr)
    pred_mlp = mlp.predict(Xte_s)
    f1_mlp = float(f1_score(yte, pred_mlp, average="macro"))
    rates_mlp = compute_move_hold_rates(yte, pred_mlp, labels)

    def _fmt_rates(r) -> str:
        return (
            f"wasted_green={r.wasted_green_rate:.3f} | illegal_go={r.illegal_go_rate:.3f} | "
            f"throughput_factor≈{r.throughput_factor:.3f}"
        )

    print("\n=== LISA dataset: same features, same split, same metric (Macro-F1 on TEST) ===")
    print(f"Classes: {len(labels)}")
    print(f"Train/Val/Test: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print("MOVE/HOLD proxy: MOVE={go, goLeft, goForward}; HOLD={stop, stopLeft, warning, warningLeft}")
    print()
    print(f"  GaussianNB:              {f1_nb:.4f}  | {_fmt_rates(rates_nb)}")
    print(f"  Q-learning bandit:       {f1_q:.4f}  (epochs={args.rl_epochs}, bins={args.rl_bins}) | {_fmt_rates(rates_q)}")
    print(f"  GA-tuned GaussianNB:     {f1_ga:.4f}  (best val F1={best_val_f1:.4f}, log10(vs)={best_log10:.2f}) | {_fmt_rates(rates_ga)}")
    print(f"  Neural net (MLP):        {f1_mlp:.4f}  (hidden={hidden}, max_iter={args.nn_max_iter}) | {_fmt_rates(rates_mlp)}")


if __name__ == "__main__":
    main()
