from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_npz(npz_path: str | Path):
    d = np.load(Path(npz_path), allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    labels = d["labels"].tolist()
    groups = d.get("groups", None)
    if groups is not None and (hasattr(groups, "shape") and groups.shape == ()):
        groups = None
    return X, y, labels, groups


def split_data(X: np.ndarray, y: np.ndarray, groups=None, seed: int = 0):
    """Train/Val/Test split: 70/15/15. Uses group split when groups are provided."""
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr_idx, tmp_idx = next(gss.split(X, y, groups))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xtmp, ytmp = X[tmp_idx], y[tmp_idx]
        gtmp = groups[tmp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
        va_idx, te_idx = next(gss2.split(Xtmp, ytmp, gtmp))
        Xva, yva = Xtmp[va_idx], ytmp[va_idx]
        Xte, yte = Xtmp[te_idx], ytmp[te_idx]
    else:
        Xtr, Xtmp, ytr, ytmp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )
        Xva, Xte, yva, yte = train_test_split(
            Xtmp, ytmp, test_size=0.50, random_state=seed + 1, stratify=ytmp
        )

    return Xtr, ytr, Xva, yva, Xte, yte
