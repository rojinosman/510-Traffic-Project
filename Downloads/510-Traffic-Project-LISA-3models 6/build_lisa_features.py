
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from common.lisa_loader import load_annotations, load_crop, extract_features_rgb, LISAFeatures


def build_features(dataset_root: Path, prefer: str = "BULB", pad: int = 0, size: int = 16, hist_bins: int = 8,
                   max_samples: int | None = None, seed: int = 0) -> LISAFeatures:
    ann = load_annotations(dataset_root, prefer=prefer)

    # Optional subsample (useful for quick runs)
    if max_samples is not None and len(ann) > max_samples:
        ann = ann.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    labels = sorted(ann["label"].unique().tolist())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    y = ann["label"].map(label_to_id).to_numpy(dtype=np.int64)

    # Group split by origin_track if present; else by origin_file; else None
    if "origin_track" in ann.columns and ann["origin_track"].notna().any():
        groups = ann["origin_track"].to_numpy()
    elif "origin_file" in ann.columns and ann["origin_file"].notna().any():
        groups = ann["origin_file"].to_numpy()
    else:
        groups = None

    feats = []
    for i, (_, row) in enumerate(ann.iterrows()):
        if (i+1) % 2000 == 0:
            print(f"  processed {i+1}/{len(ann)}")
        crop = load_crop(row, dataset_root, pad=pad)
        f = extract_features_rgb(crop, size=(size, size), hist_bins=hist_bins)
        feats.append(f)

    X = np.stack(feats, axis=0).astype(np.float32)
    return LISAFeatures(X=X, y=y, labels=labels, groups=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to extracted Kaggle dataset root")
    ap.add_argument("--out", default="lisa_features.npz", help="Output NPZ path")
    ap.add_argument("--prefer", default="BULB", choices=["BULB", "BOX"])
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--hist-bins", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    feats = build_features(
        dataset_root=data_root,
        prefer=args.prefer,
        pad=args.pad,
        size=args.size,
        hist_bins=args.hist_bins,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    np.savez_compressed(
        out_path,
        X=feats.X,
        y=feats.y,
        labels=np.array(feats.labels, dtype=object),
        groups=np.array(feats.groups, dtype=object) if feats.groups is not None else None,
        meta=np.array(
            {"data_root": str(data_root), "prefer": args.prefer, "pad": args.pad, "size": args.size, "hist_bins": args.hist_bins},
            dtype=object,
        ),
    )
    print(f"Saved: {out_path}  (X={feats.X.shape}, y={feats.y.shape}, classes={len(feats.labels)})")


if __name__ == "__main__":
    main()
