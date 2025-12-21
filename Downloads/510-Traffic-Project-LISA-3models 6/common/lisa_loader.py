
"""
Common loader for the Kaggle "LISA Traffic Light Dataset" (mbornoe/lisa-traffic-light-dataset).

Expected dataset layout:
- A root directory containing image sequences and annotation CSVs.
  In Kaggle mirrors, annotation files are typically named:
    - frameAnnotationsBULB.csv (tight box around the lit lamp)
    - frameAnnotationsBOX.csv  (expanded box around the whole traffic light)
  and can live in many nested subfolders (e.g., daySequence1, nightSequence1, etc.)

Annotation CSV format (semicolon-separated) is commonly:
Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;
Origin file;Origin frame number;Origin track;Origin track frame number
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception as e:
    raise ImportError("Pillow (PIL) is required for image loading. Install with: pip install pillow") from e


ANNOT_COLS = [
    "filename",
    "label",
    "x1",
    "y1",
    "x2",
    "y2",
    "origin_file",
    "origin_frame",
    "origin_track",
    "origin_track_frame",
]


def _read_one_csv(csv_path: Path) -> pd.DataFrame:
    # Some mirrors use ';' delimiter; others sometimes use ','.
    # We try ';' first, then ','.
    for sep in [";", ","]:
        try:
            df = pd.read_csv(csv_path, sep=sep, header=None, engine="python")
            if df.shape[1] >= 6:
                break
        except Exception:
            df = None
    if df is None or df.shape[1] < 6:
        raise ValueError(f"Could not parse annotation CSV: {csv_path}")

    # Keep first 10 columns if present; pad missing columns with NaN.
    if df.shape[1] < len(ANNOT_COLS):
        for _ in range(len(ANNOT_COLS) - df.shape[1]):
            df[df.shape[1]] = np.nan
    df = df.iloc[:, : len(ANNOT_COLS)]
    df.columns = ANNOT_COLS

    # Coerce numeric bbox columns
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize filename to POSIX-like path for joining
    df["filename"] = df["filename"].astype(str).str.strip().str.replace("\\\\", "/", regex=False)
    df["label"] = df["label"].astype(str).str.strip()
    df["origin_track"] = df["origin_track"].astype(str).str.strip()

    # Add pointers to where the CSV lived (helps resolving image paths)
    df["__csv_dir__"] = str(csv_path.parent)
    df["__csv_path__"] = str(csv_path)
    return df


def load_annotations(dataset_root: str | Path, prefer: str = "BULB") -> pd.DataFrame:
    """
    Load all annotation rows from the dataset root.

    prefer:
      - "BULB" (default): look for frameAnnotationsBULB.csv first, else fall back to BOX.
      - "BOX": look for frameAnnotationsBOX.csv first, else fall back to BULB.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    patterns = []
    if prefer.upper() == "BULB":
        patterns = ["**/frameAnnotationsBULB*.csv", "**/frameAnnotationsBOX*.csv"]
    else:
        patterns = ["**/frameAnnotationsBOX*.csv", "**/frameAnnotationsBULB*.csv"]

    csvs: List[Path] = []
    for pat in patterns:
        found = list(root.glob(pat))
        if found:
            csvs = found
            break

    if not csvs:
        raise FileNotFoundError(
            f"No annotation CSVs found under {root}. "
            f"Expected something like **/frameAnnotationsBULB.csv or **/frameAnnotationsBOX.csv."
        )

    dfs = [_read_one_csv(p) for p in sorted(csvs)]
    ann = pd.concat(dfs, ignore_index=True)

    # Drop invalid boxes
    ann = ann.dropna(subset=["x1", "y1", "x2", "y2"])
    ann = ann[(ann["x2"] > ann["x1"]) & (ann["y2"] > ann["y1"])]

    return ann


def resolve_image_path(row: pd.Series, dataset_root: str | Path) -> Path:
    """
    Resolve the on-disk image path for one annotation row.

    Strategy:
      1) Try dataset_root / row.filename
      2) Try csv_dir / row.filename (some mirrors store paths relative to annotation CSV)
      3) If filename is already absolute and exists, use it
    """
    root = Path(dataset_root).expanduser().resolve()
    fname = str(row["filename"])
    p = Path(fname)

    # Absolute path
    if p.is_absolute() and p.exists():
        return p

    # Root-relative
    p1 = root / fname
    if p1.exists():
        return p1

    # CSV-dir-relative
    csv_dir = Path(str(row.get("__csv_dir__", "")))
    if csv_dir.exists():
        p2 = csv_dir / fname
        if p2.exists():
            return p2

    # Last resort: search by basename (slow; avoid in big runs)
    base = Path(fname).name
    matches = list(root.glob(f"**/{base}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve image path for filename={fname} (from {row.get('__csv_path__')})")


def load_crop(row: pd.Series, dataset_root: str | Path, pad: int = 0) -> np.ndarray:
    """
    Load the image, crop by bbox, and return RGB uint8 array.
    """
    img_path = resolve_image_path(row, dataset_root)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        x1 = max(0, int(row["x1"]) - pad)
        y1 = max(0, int(row["y1"]) - pad)
        x2 = min(W, int(row["x2"]) + pad)
        y2 = min(H, int(row["y2"]) + pad)
        crop = im.crop((x1, y1, x2, y2))
        return np.array(crop, dtype=np.uint8)


def extract_features_rgb(crop_rgb: np.ndarray, size: Tuple[int, int] = (16, 16), hist_bins: int = 8) -> np.ndarray:
    """
    Simple, fast features that work with classical ML:
      - resized RGB pixels (flattened)
      - per-channel mean and std
      - per-channel histogram
    """
    from PIL import Image

    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError("crop_rgb must be HxWx3 RGB array")

    im = Image.fromarray(crop_rgb, mode="RGB").resize(size, resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1]
    flat = arr.reshape(-1)

    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))

    # Histograms per channel in [0,1]
    h = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=hist_bins, range=(0.0, 1.0), density=True)
        h.append(hist.astype(np.float32))
    hist = np.concatenate(h, axis=0)

    return np.concatenate([flat, mean.astype(np.float32), std.astype(np.float32), hist], axis=0)


@dataclass
class LISAFeatures:
    X: np.ndarray
    y: np.ndarray
    labels: List[str]
    groups: Optional[np.ndarray] = None  # for grouped splitting (e.g., by origin_track)

