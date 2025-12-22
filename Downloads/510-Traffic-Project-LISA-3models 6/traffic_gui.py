# multi_intersection_gui_greenstyle_second_best.py
# ------------------------------------------------------------
# Uses the "2nd best" model (tie) instead of the Neural Net:
#   ✅ Gaussian Naive Bayes (Macro-F1 ≈ 0.3932, same as GA-NB on your run)
#
# Notes:
# - The traffic optimization (who gets green) is still the queue-based controller.
# - The ML model is used as an "overlay/perception" that predicts MOVE/HOLD from
#   sampled LISA features and shows match/mismatch dots near each signal.
#
# Run examples:
#   1) NO ML overlay:
#      python multi_intersection_gui_greenstyle_second_best.py --no-ml
#
#   2) "Second best" ML overlay (GaussianNB)  ✅ (only needs the .npz)
#      python multi_intersection_gui_greenstyle_second_best.py --features lisa_features.npz
#
#   3) GA-tuned NB overlay (still "2nd best tie") (optional):
#      python multi_intersection_gui_greenstyle_second_best.py --features lisa_features.npz --ml ga_gnb --ga-log10vs -4.42
#
#   4) MLP overlay (best model) (optional; requires artifacts):
#      python multi_intersection_gui_greenstyle_second_best.py --features lisa_features.npz --ml mlp --model-dir model_artifacts
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB

try:
    import joblib
except Exception:
    joblib = None


MOVE_LABELS = {"go", "goLeft", "goForward"}
TRUTH_TO_LABELPOOL = {
    "GREEN": ["go", "goLeft", "goForward"],
    "YELLOW": ["warning", "warningLeft"],
    "RED": ["stop", "stopLeft"],
}


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


@dataclass
class SignalState:
    phase: str
    t: float


class QueueResponsiveController:
    def __init__(self, min_green=3.0, max_green=10.0, yellow=1.2, all_red=0.6):
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow = float(yellow)
        self.all_red = float(all_red)
        self.state = SignalState("EW_GREEN", 0.0)

    def reset(self):
        self.state = SignalState("EW_GREEN", 0.0)

    def step(self, dt: float, q_ns: int, q_ew: int) -> SignalState:
        s = self.state
        s.t += dt

        def go(phase: str):
            self.state = SignalState(phase, 0.0)

        if s.phase in ("NS_GREEN", "EW_GREEN"):
            is_ns = (s.phase == "NS_GREEN")
            cur_q = q_ns if is_ns else q_ew
            other_q = q_ew if is_ns else q_ns

            if s.t < self.min_green:
                return self.state

            if s.t >= self.max_green:
                go("NS_YELLOW" if is_ns else "EW_YELLOW")
                return self.state

            if other_q > cur_q + 2:
                go("NS_YELLOW" if is_ns else "EW_YELLOW")
                return self.state

            return self.state

        if s.phase in ("NS_YELLOW", "EW_YELLOW"):
            if s.t >= self.yellow:
                go("ALL_RED")
            return self.state

        if s.phase == "ALL_RED":
            if s.t >= self.all_red:
                go("NS_GREEN" if q_ns >= q_ew else "EW_GREEN")
            return self.state

        return self.state

    @staticmethod
    def truth_buckets(phase: str) -> Tuple[str, str]:
        if phase == "NS_GREEN":
            return "GREEN", "RED"
        if phase == "NS_YELLOW":
            return "YELLOW", "RED"
        if phase == "EW_GREEN":
            return "RED", "GREEN"
        if phase == "EW_YELLOW":
            return "RED", "YELLOW"
        return "RED", "RED"


class BasePerception:
    name: str = "ML"

    def predict_bucket(self, truth_bucket: str) -> Tuple[str, float]:
        raise NotImplementedError


class GaussianNBPerception(BasePerception):
    """
    SECOND-BEST (tie): GaussianNB (supervised baseline)
    Fits on startup using the same split scheme as the project scripts.
    """
    name = "GaussianNB (2nd-best tie)"

    def __init__(self, features_npz: Path, seed: int = 0, var_smoothing: float = 1e-9):
        self.rng = random.Random(seed)
        X, y, labels, groups = load_npz(features_npz)

        # For sampling "truth-like" frames later
        y_names = np.array([labels[i] for i in y])
        self.idx_by_name: Dict[str, np.ndarray] = {name: np.where(y_names == name)[0] for name in labels}
        self.labels = labels

        Xtr, ytr, _, _, _, _ = split_data(X, y, groups=groups, seed=seed)
        self.model = GaussianNB(var_smoothing=float(var_smoothing))
        self.model.fit(Xtr, ytr)

        self.X = X  # fallback pool if needed

    def _sample_X(self, truth_bucket: str) -> np.ndarray:
        pool = TRUTH_TO_LABELPOOL[truth_bucket]
        candidates = [n for n in pool if self.idx_by_name.get(n, np.array([])).size > 0]
        if not candidates:
            idx = self.rng.randrange(0, self.X.shape[0])
            return self.X[idx:idx + 1]
        name = self.rng.choice(candidates)
        idx = int(self.rng.choice(self.idx_by_name[name]))
        return self.X[idx:idx + 1]

    def predict_bucket(self, truth_bucket: str) -> Tuple[str, float]:
        X = self._sample_X(truth_bucket)
        proba = self.model.predict_proba(X)[0]
        pred_i = int(np.argmax(proba))
        conf = float(proba[pred_i])
        pred_label = self.labels[pred_i] if pred_i < len(self.labels) else str(pred_i)
        pred_bucket = "GREEN" if pred_label in MOVE_LABELS else "RED"
        return pred_bucket, conf


class MLPPerception(BasePerception):
    """BEST model (MLP). Requires model artifacts directory."""
    name = "MLP (best)"

    def __init__(self, features_npz: Path, model_dir: Path, seed: int = 0):
        if joblib is None:
            raise RuntimeError("joblib not available; install joblib or use GaussianNB overlay.")

        self.rng = random.Random(seed)
        d = np.load(features_npz, allow_pickle=True)
        self.X = d["X"].astype(np.float32)
        self.y = d["y"].astype(int)
        labels_npz = list(d["labels"])
        y_names = np.array([labels_npz[i] for i in self.y])

        self.idx_by_name: Dict[str, np.ndarray] = {}
        for name in labels_npz:
            self.idx_by_name[name] = np.where(y_names == name)[0]

        meta = json.loads((model_dir / "model_meta.json").read_text())
        self.labels = list(meta["labels"])
        self.mlp = joblib.load(model_dir / "mlp_model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")

    def _sample_X(self, truth_bucket: str) -> np.ndarray:
        pool = TRUTH_TO_LABELPOOL[truth_bucket]
        candidates = [n for n in pool if self.idx_by_name.get(n, np.array([])).size > 0]
        if not candidates:
            idx = self.rng.randrange(0, self.X.shape[0])
            return self.X[idx:idx + 1]
        name = self.rng.choice(candidates)
        idx = int(self.rng.choice(self.idx_by_name[name]))
        return self.X[idx:idx + 1]

    def predict_bucket(self, truth_bucket: str) -> Tuple[str, float]:
        X = self._sample_X(truth_bucket)
        Xs = self.scaler.transform(X)
        proba = self.mlp.predict_proba(Xs)[0]
        pred_i = int(np.argmax(proba))
        conf = float(proba[pred_i])
        pred_label = self.labels[pred_i] if pred_i < len(self.labels) else str(pred_i)
        pred_bucket = "GREEN" if pred_label in MOVE_LABELS else "RED"
        return pred_bucket, conf


@dataclass
class CarEW:
    x: float
    v: float


@dataclass
class CarNS:
    y: float
    v: float


@dataclass
class Node:
    x: float
    ctrl: QueueResponsiveController
    truth_ns: str = "RED"
    truth_ew: str = "GREEN"

    ns_down: List[CarNS] = None
    ns_up: List[CarNS] = None

    ml_ns_bucket: str = "RED"
    ml_ew_bucket: str = "RED"
    _t: float = 0.0
    _interval: float = 0.25
    _votes_ns: List[str] = None
    _votes_ew: List[str] = None
    _win: int = 7

    def __post_init__(self):
        self.ns_down = []
        self.ns_up = []
        self._votes_ns, self._votes_ew = [], []

    @staticmethod
    def _majority(votes: List[str], default="RED") -> str:
        if not votes:
            return default
        return max(set(votes), key=votes.count)


class MultiSim:
    def __init__(self, W: int, H: int, n: int, seed: int = 0):
        self.W, self.H = W, H
        self.rng = random.Random(seed)

        self.n = max(2, min(8, int(n)))
        self.margin = 140
        self.spacing = (self.W - 2 * self.margin) / (self.n - 1)

        self.road_thick = 120
        self.lane_offset = 22
        self.stop_offset = 50
        self.box = 108

        self.y_mid = self.H * 0.52
        self.y_east = self.y_mid - self.lane_offset
        self.y_west = self.y_mid + self.lane_offset

        self.x_ns_down_offset = +self.lane_offset
        self.x_ns_up_offset = -self.lane_offset

        self.car_gap = 34
        self.v_ew = 150.0
        self.v_ns = 140.0

        self.throughput = 0
        self.time = 0.0

        self.nodes: List[Node] = []
        for i in range(self.n):
            x = self.margin + i * self.spacing
            self.nodes.append(Node(x=x, ctrl=QueueResponsiveController()))

        self.cars_ew: List[CarEW] = []

        self.spawn_pad = 60
        self.exit_pad = 220

    def reset(self):
        self.throughput = 0
        self.time = 0.0
        self.cars_ew.clear()
        for node in self.nodes:
            node.ctrl.reset()
            node.truth_ns, node.truth_ew = "RED", "GREEN"
            node.ns_down.clear()
            node.ns_up.clear()
            node._votes_ns.clear()
            node._votes_ew.clear()
            node._t = 0.0
            node.ml_ns_bucket = "RED"
            node.ml_ew_bucket = "RED"

    def add_initial(self, ew_each_side: int, ns_each_intersection: int):
        for i in range(ew_each_side):
            self.cars_ew.append(CarEW(x=self.spawn_pad - i * self.car_gap, v=self.v_ew))
            self.cars_ew.append(CarEW(x=self.W - self.spawn_pad + i * self.car_gap, v=-self.v_ew))

        for node in self.nodes:
            for k in range(ns_each_intersection):
                node.ns_down.append(CarNS(y=self.spawn_pad - k * self.car_gap, v=self.v_ns))
                node.ns_up.append(CarNS(y=self.H - self.spawn_pad + k * self.car_gap, v=-self.v_ns))

    def stopline_ew(self, node: Node, car: CarEW) -> float:
        return node.x - self.stop_offset if car.v > 0 else node.x + self.stop_offset

    def next_node_idx_ew(self, car: CarEW) -> Optional[int]:
        if car.v > 0:
            for i, node in enumerate(self.nodes):
                if node.x - self.stop_offset >= car.x - 1e-6:
                    return i
            return None
        else:
            for i in range(self.n - 1, -1, -1):
                node = self.nodes[i]
                if node.x + self.stop_offset <= car.x + 1e-6:
                    return i
            return None

    def ew_waiting_counts(self) -> List[int]:
        counts = [0 for _ in range(self.n)]
        for car in self.cars_ew:
            idx = self.next_node_idx_ew(car)
            if idx is None:
                continue
            sx = self.stopline_ew(self.nodes[idx], car)
            if abs(car.x - sx) <= 2.5:
                counts[idx] += 1
        return counts

    def ns_queue_count(self, node: Node) -> int:
        q = 0
        top_sl = self.y_mid - self.stop_offset
        bot_sl = self.y_mid + self.stop_offset

        for c in node.ns_down:
            if c.y <= top_sl + 1e-6:
                q += 1
        for c in node.ns_up:
            if c.y >= bot_sl - 1e-6:
                q += 1
        return q

    def spawn(self, lam_ew: float, lam_ns: float, dt: float):
        if self.rng.random() < lam_ew * dt:
            self.cars_ew.append(CarEW(x=self.spawn_pad, v=self.v_ew))
        if self.rng.random() < lam_ew * dt:
            self.cars_ew.append(CarEW(x=self.W - self.spawn_pad, v=-self.v_ew))

        for node in self.nodes:
            if self.rng.random() < lam_ns * dt:
                if not node.ns_down or node.ns_down[-1].y > self.spawn_pad + self.car_gap:
                    node.ns_down.append(CarNS(y=self.spawn_pad, v=self.v_ns))
            if self.rng.random() < lam_ns * dt:
                if not node.ns_up or node.ns_up[-1].y < self.H - self.spawn_pad - self.car_gap:
                    node.ns_up.append(CarNS(y=self.H - self.spawn_pad, v=-self.v_ns))

    def step(self, dt: float, perception: Optional[BasePerception] = None):
        self.time += dt
        ew_wait = self.ew_waiting_counts()

        for i, node in enumerate(self.nodes):
            q_ns = self.ns_queue_count(node)
            q_ew = ew_wait[i]
            st = node.ctrl.step(dt, q_ns=q_ns, q_ew=q_ew)
            node.truth_ns, node.truth_ew = node.ctrl.truth_buckets(st.phase)

            if perception is not None:
                node._t += dt
                if node._t >= node._interval:
                    node._t = 0.0
                    b_ns, _ = perception.predict_bucket(node.truth_ns)
                    b_ew, _ = perception.predict_bucket(node.truth_ew)
                    node._votes_ns.append(b_ns)
                    node._votes_ew.append(b_ew)
                    node._votes_ns[:] = node._votes_ns[-node._win:]
                    node._votes_ew[:] = node._votes_ew[-node._win:]
                    node.ml_ns_bucket = node._majority(node._votes_ns)
                    node.ml_ew_bucket = node._majority(node._votes_ew)

        east = [c for c in self.cars_ew if c.v > 0]
        west = [c for c in self.cars_ew if c.v < 0]
        east.sort(key=lambda c: c.x, reverse=True)
        west.sort(key=lambda c: c.x)

        def update_ew(cars_dir: List[CarEW]):
            for j, car in enumerate(cars_dir):
                proposed = car.x + car.v * dt
                idx = self.next_node_idx_ew(car)
                if idx is not None:
                    node = self.nodes[idx]
                    sx = self.stopline_ew(node, car)
                    if node.truth_ew != "GREEN":
                        if car.v > 0:
                            proposed = min(proposed, sx)
                        else:
                            proposed = max(proposed, sx)

                if j > 0:
                    lead = cars_dir[j - 1]
                    if car.v > 0:
                        proposed = min(proposed, lead.x - self.car_gap)
                    else:
                        proposed = max(proposed, lead.x + self.car_gap)

                car.x = proposed

        update_ew(east)
        update_ew(west)

        top_sl = self.y_mid - self.stop_offset
        bot_sl = self.y_mid + self.stop_offset

        for node in self.nodes:
            node.ns_down.sort(key=lambda c: c.y, reverse=True)
            for j, car in enumerate(node.ns_down):
                proposed = car.y + car.v * dt
                if node.truth_ns != "GREEN":
                    proposed = min(proposed, top_sl)
                if j > 0:
                    lead = node.ns_down[j - 1]
                    proposed = min(proposed, lead.y - self.car_gap)
                car.y = proposed

            node.ns_up.sort(key=lambda c: c.y)
            for j, car in enumerate(node.ns_up):
                proposed = car.y + car.v * dt
                if node.truth_ns != "GREEN":
                    proposed = max(proposed, bot_sl)
                if j > 0:
                    lead = node.ns_up[j - 1]
                    proposed = max(proposed, lead.y + self.car_gap)
                car.y = proposed

            alive_down = []
            for car in node.ns_down:
                if car.y > self.H + self.exit_pad:
                    self.throughput += 1
                else:
                    alive_down.append(car)
            node.ns_down = alive_down

            alive_up = []
            for car in node.ns_up:
                if car.y < -self.exit_pad:
                    self.throughput += 1
                else:
                    alive_up.append(car)
            node.ns_up = alive_up

        alive_ew: List[CarEW] = []
        for car in self.cars_ew:
            if car.x < -self.exit_pad or car.x > self.W + self.exit_pad:
                self.throughput += 1
            else:
                alive_ew.append(car)
        self.cars_ew = alive_ew

    def total_queue_ns(self) -> int:
        return sum(self.ns_queue_count(n) for n in self.nodes)

    def total_queue_ew(self) -> int:
        return sum(self.ew_waiting_counts())


class App(tk.Tk):
    def __init__(self, n: int, perception: Optional[BasePerception], seed: int = 0):
        super().__init__()
        self.title("Green-Style Multi-Intersection Traffic Demo (2nd-best ML overlay)")

        self.update_idletasks()
        sw = max(1200, self.winfo_screenwidth())
        sh = max(800, self.winfo_screenheight())

        self.right_w = 360
        self.W = int(min(1200, sw - self.right_w - 80))
        self.H = int(min(740, sh - 140))

        win_w = self.W + self.right_w + 60
        win_h = self.H + 80
        self.geometry(f"{win_w}x{win_h}")

        self.sim = MultiSim(self.W, self.H, n=n, seed=seed)
        self.perception = perception

        self.running = False
        self.last = time.time()

        self.arrival_ew_per_min = tk.StringVar(value="30")
        self.arrival_ns_per_min = tk.StringVar(value="8")
        self.initial_ew_each_side = tk.StringVar(value="6")
        self.initial_ns_each_int = tk.StringVar(value="3")
        self.n_var = tk.StringVar(value=str(n))
        self.sim_speed = tk.DoubleVar(value=1.0)

        self._build()
        self._tick()

    def _build(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        pw = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw)
        right = ttk.Frame(pw, width=self.right_w)
        right.pack_propagate(False)

        pw.add(left, weight=5)
        pw.add(right, weight=0)

        self.canvas = tk.Canvas(left, width=self.W, height=self.H, bg="#1f3a24", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.LabelFrame(right, text="Controls", padding=10)
        ctrl.pack(fill=tk.X, pady=(0, 10))

        def row(parent: ttk.Frame, label: str, var: tk.StringVar):
            r = ttk.Frame(parent)
            r.pack(fill=tk.X, pady=6)
            ttk.Label(r, text=label).grid(row=0, column=0, sticky="w")
            e = ttk.Entry(r, textvariable=var)
            e.grid(row=0, column=1, sticky="ew", padx=(10, 0))
            r.columnconfigure(1, weight=1)
            return e

        row(ctrl, "# intersections (2..8)", self.n_var)
        row(ctrl, "EW arrival (cars/min)", self.arrival_ew_per_min)
        row(ctrl, "NS arrival (cars/min/int)", self.arrival_ns_per_min)
        row(ctrl, "Initial EW cars/side", self.initial_ew_each_side)
        row(ctrl, "Initial NS cars/int", self.initial_ns_each_int)

        sr = ttk.Frame(ctrl)
        sr.pack(fill=tk.X, pady=8)
        ttk.Label(sr, text="Sim speed").grid(row=0, column=0, sticky="w")
        ttk.Scale(sr, from_=0.25, to=3.0, variable=self.sim_speed, orient=tk.HORIZONTAL).grid(
            row=0, column=1, sticky="ew", padx=(10, 0)
        )
        sr.columnconfigure(1, weight=1)

        br = ttk.Frame(ctrl)
        br.pack(fill=tk.X, pady=(12, 0))
        for c in range(2):
            br.columnconfigure(c, weight=1)

        ttk.Button(br, text="Start", command=self.on_start).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(br, text="Pause", command=self.on_pause).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(br, text="Reset", command=self.on_reset).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(br, text="Apply N", command=self.on_apply_n).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        stats = ttk.LabelFrame(right, text="Stats", padding=10)
        stats.pack(fill=tk.X)
        self.lbl_stats = ttk.Label(stats, text="—", justify="left")
        self.lbl_stats.pack(anchor="w")

    def _lam(self, per_min: str) -> float:
        try:
            v = float(per_min)
            if v < 0:
                raise ValueError
            return v / 60.0
        except Exception:
            return 0.0

    def on_apply_n(self):
        try:
            n = int(self.n_var.get())
            if n < 2 or n > 8:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "N must be an integer in [2..8].")
            return
        self.sim = MultiSim(self.W, self.H, n=n, seed=0)
        self.running = False
        self._render()

    def on_start(self):
        try:
            ew_init = int(self.initial_ew_each_side.get())
            ns_init = int(self.initial_ns_each_int.get())
            if ew_init < 0 or ns_init < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Initial values must be non-negative integers.")
            return

        self.sim.reset()
        self.sim.add_initial(ew_each_side=ew_init, ns_each_intersection=ns_init)
        self.running = True
        self.last = time.time()

    def on_pause(self):
        self.running = False

    def on_reset(self):
        self.running = False
        self.sim.reset()
        self._render()

    # ---------- Drawing ----------
    def _capsule(self, x: float, y: float, rx: float, ry: float, fill: str, outline: str):
        self.canvas.create_oval(x - rx, y - ry, x + rx, y + ry, fill=fill, outline=outline, width=2)

    def _car_ew(self, x: float, y: float, v: float):
        fill = "#56a7ff"
        outline = "#0b2a55"
        self._capsule(x, y, 16, 10, fill, outline)
        fx = x + (12 if v > 0 else -12)
        self.canvas.create_oval(fx - 3, y - 3, fx + 3, y + 3, fill="#e9f2ff", outline="")

    def _car_ns(self, x: float, y: float, v: float):
        fill = "#7fc2ff"
        outline = "#0b2a55"
        self._capsule(x, y, 10, 16, fill, outline)
        fy = y + (12 if v > 0 else -12)
        self.canvas.create_oval(x - 3, fy - 3, x + 3, fy + 3, fill="#e9f2ff", outline="")

    def _lamp(self, x, y, on, col):
        r = 6
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=(col if on else "#233"), outline="#000", width=2)

    def _signal_vertical(self, x: float, y: float, bucket: str, label: str, ml_dot: Optional[bool]):
        c = self.canvas
        w, h = 18, 52
        c.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2, fill="#101010", outline="#333", width=2)
        self._lamp(x, y - 16, bucket == "RED", "#ff4b4b")
        self._lamp(x, y, bucket == "YELLOW", "#ffd24b")
        self._lamp(x, y + 16, bucket == "GREEN", "#35ff7a")
        c.create_text(x, y + h/2 + 12, text=label, fill="#ffffff", font=("Menlo", 10))
        if ml_dot is not None:
            col = "#35ff7a" if ml_dot else "#ff4b4b"
            c.create_oval(x + w/2 + 4, y - h/2 + 4, x + w/2 + 12, y - h/2 + 12, fill=col, outline="")

    def _signal_horizontal(self, x: float, y: float, bucket: str, label: str, ml_dot: Optional[bool]):
        c = self.canvas
        w, h = 52, 18
        c.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2, fill="#101010", outline="#333", width=2)
        self._lamp(x - 16, y, bucket == "RED", "#ff4b4b")
        self._lamp(x, y, bucket == "YELLOW", "#ffd24b")
        self._lamp(x + 16, y, bucket == "GREEN", "#35ff7a")
        c.create_text(x, y + h/2 + 12, text=label, fill="#ffffff", font=("Menlo", 10))
        if ml_dot is not None:
            col = "#35ff7a" if ml_dot else "#ff4b4b"
            c.create_oval(x + w/2 + 4, y - h/2 - 12, x + w/2 + 12, y - h/2 - 4, fill=col, outline="")

    def _draw_world(self):
        c = self.canvas
        c.delete("all")

        c.create_rectangle(0, 0, self.sim.W, self.sim.H, fill="#1f3a24", outline="")
        for gx in range(0, self.sim.W, 220):
            c.create_rectangle(gx, 0, gx + 110, self.sim.H, fill="#1b331f", outline="")

        road = "#2a2f3a"
        road_edge = "#222733"
        mark = "#cfd6dd"

        ymid = self.sim.y_mid
        half = self.sim.road_thick / 2

        c.create_rectangle(0, ymid - half, self.sim.W, ymid + half, fill=road, outline=road_edge, width=4)

        x = 0
        while x < self.sim.W:
            c.create_line(x, ymid, min(self.sim.W, x + 16), ymid, fill=mark, width=3)
            x += 32

        for node in self.sim.nodes:
            cx = node.x
            b = self.sim.box / 2

            c.create_rectangle(cx - b, 0, cx + b, self.sim.H, fill=road, outline=road_edge, width=4)

            sxL = cx - self.sim.stop_offset
            sxR = cx + self.sim.stop_offset
            c.create_line(sxL, ymid - half, sxL, ymid + half, fill=mark, width=4)
            c.create_line(sxR, ymid - half, sxR, ymid + half, fill=mark, width=4)

            syT = ymid - self.sim.stop_offset
            syB = ymid + self.sim.stop_offset
            c.create_line(cx - half, syT, cx + half, syT, fill=mark, width=4)
            c.create_line(cx - half, syB, cx + half, syB, fill=mark, width=4)

            ml_ns = None
            ml_ew = None
            if self.perception is not None:
                if node.truth_ns == "YELLOW":
                    ml_ns = True
                else:
                    ml_ns = (node.ml_ns_bucket == ("GREEN" if node.truth_ns == "GREEN" else "RED"))
                if node.truth_ew == "YELLOW":
                    ml_ew = True
                else:
                    ml_ew = (node.ml_ew_bucket == ("GREEN" if node.truth_ew == "GREEN" else "RED"))

            self._signal_vertical(cx, ymid - b - 36, node.truth_ns, "NS", ml_ns)
            self._signal_horizontal(cx - b - 44, ymid, node.truth_ew, "EW", ml_ew)

            q = self.sim.ns_queue_count(node)
            c.create_text(cx, 18, text=f"NSQ={q}", fill="#e9f2ea", font=("Menlo", 11))

            x_down = cx + self.sim.x_ns_down_offset
            x_up = cx + self.sim.x_ns_up_offset
            for car in node.ns_down:
                self._car_ns(x_down, car.y, car.v)
            for car in node.ns_up:
                self._car_ns(x_up, car.y, car.v)

        for car in self.sim.cars_ew:
            y = self.sim.y_east if car.v > 0 else self.sim.y_west
            self._car_ew(car.x, y, car.v)

        c.create_rectangle(0, 0, 620, 102, fill="#0e1a11", outline="")
        model_name = self.perception.name if self.perception is not None else "None"
        c.create_text(16, 18, text=f"Throughput: {self.sim.throughput}", fill="#e9f2ea", anchor="w", font=("Menlo", 18))
        c.create_text(
            16, 46,
            text=f"Total Queue NS/EW: {self.sim.total_queue_ns()} / {self.sim.total_queue_ew()}",
            fill="#e9f2ea", anchor="w", font=("Menlo", 16),
        )
        c.create_text(16, 76, text=f"ML overlay: {model_name}", fill="#e9f2ea", anchor="w", font=("Menlo", 14))

    def _render(self):
        self._draw_world()
        phases = " | ".join(f"{i+1}:{n.ctrl.state.phase}" for i, n in enumerate(self.sim.nodes))
        model_name = self.perception.name if self.perception is not None else "None"
        self.lbl_stats.config(
            text=(
                f"ML overlay: {model_name}\n"
                f"Intersections: {self.sim.n}\n"
                f"Time: {self.sim.time:.1f}s\n"
                f"Throughput: {self.sim.throughput}\n"
                f"Total queue NS: {self.sim.total_queue_ns()}\n"
                f"Total queue EW: {self.sim.total_queue_ew()}\n"
                f"Phases: {phases}"
            )
        )

    def _tick(self):
        now = time.time()
        dt_real = min(now - self.last, 0.05)
        self.last = now
        dt = dt_real * float(self.sim_speed.get())

        if self.running:
            lam_ew = self._lam(self.arrival_ew_per_min.get())
            lam_ns = self._lam(self.arrival_ns_per_min.get())
            self.sim.spawn(lam_ew=lam_ew, lam_ns=lam_ns, dt=dt)
            self.sim.step(dt, perception=self.perception)

        self._render()
        self.after(16, self._tick)


def build_perception(ml: str, features: Path, model_dir: Optional[Path], ga_log10vs: float) -> BasePerception:
    ml = ml.lower()
    if ml == "gnb":
        return GaussianNBPerception(features, seed=0, var_smoothing=1e-9)
    if ml == "ga_gnb":
        # var_smoothing = 10^(log10vs)  (your output had log10(vs)=-4.42)
        vs = 10.0 ** float(ga_log10vs)
        return GaussianNBPerception(features, seed=0, var_smoothing=vs)
    if ml == "mlp":
        if model_dir is None:
            raise SystemExit("MLP requires --model-dir model_artifacts")
        return MLPPerception(features, model_dir, seed=0)
    raise SystemExit("Unknown --ml. Use: gnb, ga_gnb, mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intersections", type=int, default=6)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--features", default=None)
    ap.add_argument("--ml", default="gnb", choices=["gnb", "ga_gnb", "mlp"])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--ga-log10vs", type=float, default=-4.42)
    args = ap.parse_args()

    if args.no_ml:
        App(n=args.intersections, perception=None, seed=0).mainloop()
        return

    if args.features is None:
        raise SystemExit("Provide --features lisa_features.npz, or use --no-ml.")

    features = Path(args.features).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None

    perception = build_perception(args.ml, features, model_dir, args.ga_log10vs)
    App(n=args.intersections, perception=perception, seed=0).mainloop()


if __name__ == "__main__":
    main()
