from __future__ import annotations

"""
Tkinter GUI for the best-performing model (MLP) on the LISA feature pipeline.

What it does:
- Upload an image (best results if it's a cropped traffic-light image)
- Optional manual crop via x1,y1,x2,y2
- Predict class label + confidence
- Visualize per-class probabilities as progress bars
- Show a simple MOVE vs HOLD explanation (helpful for throughput/queue discussion)

Run:
  python mlp_gui.py --features /path/to/lisa_features.npz

Before running the GUI, train and save the model once:
  python train_mlp_model.py --features /path/to/lisa_features.npz
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Ensure we can import common/ when run as a script
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.lisa_loader import extract_features_rgb


MOVE_LABELS = {"go", "goLeft", "goForward"}
HOLD_LABELS = {"stop", "stopLeft", "warning", "warningLeft"}


@dataclass
class FeatureMeta:
    size: int = 16
    hist_bins: int = 8


def load_feature_meta(npz_path: Path) -> FeatureMeta:
    """
    Reads meta saved by build_lisa_features.py (size, hist_bins).
    Falls back to defaults if missing.
    """
    try:
        d = np.load(npz_path, allow_pickle=True)
        meta_obj = d.get("meta", None)
        if meta_obj is None:
            return FeatureMeta()
        meta = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
        return FeatureMeta(size=int(meta.get("size", 16)), hist_bins=int(meta.get("hist_bins", 8)))
    except Exception:
        return FeatureMeta()


def load_model_artifacts(model_dir: Path):
    model_path = model_dir / "mlp_model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "model_meta.json"

    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Missing model artifacts in {model_dir}.\n"
            f"Expected: mlp_model.joblib, scaler.joblib, model_meta.json\n\n"
            f"Run:\n  python train_mlp_model.py --features /path/to/lisa_features.npz"
        )

    mlp = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())
    labels = meta.get("labels", None)
    if not labels:
        raise ValueError("model_meta.json missing 'labels'. Re-train with train_mlp_model.py")
    return mlp, scaler, list(labels)


class App(tk.Tk):
    def __init__(self, features_npz: Path, model_dir: Path):
        super().__init__()
        self.title("LISA Traffic Light Classifier (Best Model: MLP)")
        self.geometry("1060x740")

        self.features_npz = features_npz
        self.feature_meta = load_feature_meta(features_npz)

        self.mlp, self.scaler, self.labels = load_model_artifacts(model_dir)

        self.image_path: Optional[Path] = None
        self.image_pil: Optional[Image.Image] = None
        self.preview_tk: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        right = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # Image preview
        self.preview_label = ttk.Label(left, text="Upload an image to begin")
        self.preview_label.pack(pady=8)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=6)

        ttk.Button(btn_row, text="Upload Image", command=self.on_upload).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Predict", command=self.on_predict).pack(side=tk.LEFT, padx=4)

        # Crop inputs
        crop_box = ttk.LabelFrame(left, text="Optional Crop (pixels)")
        crop_box.pack(fill=tk.X, pady=10)

        self.x1 = tk.StringVar(value="")
        self.y1 = tk.StringVar(value="")
        self.x2 = tk.StringVar(value="")
        self.y2 = tk.StringVar(value="")

        def entry(lbl, var):
            col = ttk.Frame(crop_box)
            col.pack(side=tk.LEFT, padx=6, pady=6)
            ttk.Label(col, text=lbl).pack()
            ttk.Entry(col, width=7, textvariable=var).pack()

        entry("x1", self.x1)
        entry("y1", self.y1)
        entry("x2", self.x2)
        entry("y2", self.y2)

        ttk.Label(
            crop_box,
            text=(
                "Leave blank to use the full image.\n"
                "Tip: upload a cropped traffic-light image for best results."
            ),
        ).pack(side=tk.LEFT, padx=10)

        # Settings info
        info = ttk.LabelFrame(left, text="Feature Settings (from lisa_features.npz)")
        info.pack(fill=tk.X, pady=10)
        ttk.Label(info, text=f"Resize size: {self.feature_meta.size} x {self.feature_meta.size}").pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, text=f"Histogram bins: {self.feature_meta.hist_bins}").pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, text=f"Classes: {len(self.labels)}").pack(anchor="w", padx=8, pady=2)

        # Right: prediction
        out = ttk.LabelFrame(right, text="Prediction")
        out.pack(fill=tk.BOTH, expand=True)

        self.pred_label = ttk.Label(out, text="—", font=("Helvetica", 18, "bold"))
        self.pred_label.pack(padx=10, pady=(14, 4))

        self.pred_conf = ttk.Label(out, text="Upload an image and click Predict.")
        self.pred_conf.pack(padx=10, pady=(0, 10))

        self.explain = ttk.Label(out, text="")
        self.explain.pack(padx=10, pady=(0, 12))

        prob_frame = ttk.LabelFrame(out, text="Class probabilities")
        prob_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.rows: list[tuple[str, ttk.Progressbar, ttk.Label]] = []
        for name in self.labels:
            r = ttk.Frame(prob_frame)
            r.pack(fill=tk.X, padx=8, pady=4)
            ttk.Label(r, text=name, width=12).pack(side=tk.LEFT)
            bar = ttk.Progressbar(r, length=200, maximum=1.0)
            bar.pack(side=tk.LEFT, padx=6)
            val = ttk.Label(r, text="0.000", width=6)
            val.pack(side=tk.LEFT)
            self.rows.append((name, bar, val))

    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        self.image_path = Path(path)
        try:
            with Image.open(self.image_path) as im:
                self.image_pil = im.convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")
            return

        self._update_preview()
        self._reset_output()

    def _reset_output(self):
        self.pred_label.config(text="—")
        self.pred_conf.config(text="")
        self.explain.config(text="")
        for _, bar, val in self.rows:
            bar["value"] = 0.0
            val.config(text="0.000")

    def _update_preview(self):
        assert self.image_pil is not None
        im = self.image_pil.copy()

        # Fit preview into left pane
        max_w, max_h = 680, 520
        w, h = im.size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))

        self.preview_tk = ImageTk.PhotoImage(im)
        self.preview_label.config(image=self.preview_tk, text="")

    def _parse_crop(self) -> Optional[Tuple[int, int, int, int]]:
        vals = (self.x1.get().strip(), self.y1.get().strip(), self.x2.get().strip(), self.y2.get().strip())
        if all(v == "" for v in vals):
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in vals)
        except ValueError:
            raise ValueError("Crop values must be integers or left blank.")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Crop must satisfy x2>x1 and y2>y1.")
        return x1, y1, x2, y2

    def on_predict(self):
        if self.image_pil is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return

        try:
            crop = self._parse_crop()
        except Exception as e:
            messagebox.showerror("Crop error", str(e))
            return

        im = self.image_pil
        if crop is not None:
            W, H = im.size
            x1, y1, x2, y2 = crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            im = im.crop((x1, y1, x2, y2))

        crop_rgb = np.array(im, dtype=np.uint8)

        # Feature extraction must match build_lisa_features.py settings
        feat = extract_features_rgb(
            crop_rgb,
            size=(self.feature_meta.size, self.feature_meta.size),
            hist_bins=self.feature_meta.hist_bins,
        ).astype(np.float32)

        X = feat.reshape(1, -1)
        Xs = self.scaler.transform(X)

        proba = self.mlp.predict_proba(Xs)[0]
        pred_i = int(np.argmax(proba))
        pred_name = self.labels[pred_i]
        conf = float(proba[pred_i])

        self.pred_label.config(text=pred_name)
        self.pred_conf.config(text=f"Confidence (top prob): {conf:.3f}")

        # Basic explanation for throughput/queue framing
        if pred_name in MOVE_LABELS:
            explain = "Meaning: MOVE (vehicles should be allowed to proceed) → supports throughput"
        else:
            explain = "Meaning: HOLD (vehicles should stop/yield) → prevents unsafe movement"
        self.explain.config(text=explain)

        for name, bar, val in self.rows:
            i = self.labels.index(name)
            p = float(proba[i])
            bar["value"] = p
            val.config(text=f"{p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz (for feature meta)")
    ap.add_argument("--model-dir", default=".", help="Directory containing mlp_model.joblib, scaler.joblib, model_meta.json")
    args = ap.parse_args()

    # Quick check: Tk availability
    try:
        import tkinter  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Tkinter is not available in this Python environment.\n"
            "If you're using conda/miniforge, try:\n"
            "  conda install tk\n"
        ) from e

    app = App(Path(args.features).expanduser().resolve(), Path(args.model_dir).expanduser().resolve())
    app.mainloop()


if __name__ == "__main__":
    main()
