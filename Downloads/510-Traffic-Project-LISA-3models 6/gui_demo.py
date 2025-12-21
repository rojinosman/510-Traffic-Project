from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pygame
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common.data_utils import load_npz, split_data
from Reinforced.q_learning_lisa import QBandit
from Unsupervised.ga_optimizer_lisa import ga_tune_var_smoothing


# ----------------------------
# Label mapping for the demo
# ----------------------------
MOVE_LABELS = {"go", "goLeft", "goForward"}
HOLD_LABELS = {"stop", "stopLeft", "warning", "warningLeft"}

# For sim truth phases, we use only 3 “truth” buckets:
# GREEN -> sample from MOVE labels
# YELLOW -> sample from warning labels
# RED -> sample from stop labels
TRUTH_TO_LABELPOOL = {
    "GREEN": {"go", "goLeft", "goForward"},
    "YELLOW": {"warning", "warningLeft"},
    "RED": {"stop", "stopLeft"},
}


def label_to_move_hold(label_name: str) -> str:
    return "MOVE" if label_name in MOVE_LABELS else "HOLD"


@dataclass
class Rates:
    wasted_green: int = 0
    illegal_go: int = 0
    total_true_move: int = 0
    total_true_hold: int = 0

    def wasted_green_rate(self) -> float:
        return self.wasted_green / self.total_true_move if self.total_true_move else 0.0

    def illegal_go_rate(self) -> float:
        return self.illegal_go / self.total_true_hold if self.total_true_hold else 0.0

    def throughput_factor(self) -> float:
        # Proxy from earlier: 1 - wasted_green_rate
        return 1.0 - self.wasted_green_rate()


# ----------------------------
# Simple traffic simulation
# ----------------------------
@dataclass
class Car:
    x: float
    y: float
    vx: float
    vy: float
    lane: str  # "N","S","E","W"
    length: int = 16
    width: int = 10

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x - self.width / 2), int(self.y - self.length / 2), self.width, self.length)


class IntersectionSim:
    """
    A lightweight demo (not a calibrated traffic simulator).
    Cars spawn on 4 approaches and move straight through if their direction is predicted "GREEN".
    """

    def __init__(self, w: int = 900, h: int = 700):
        self.w, self.h = w, h
        self.cx, self.cy = w // 2, h // 2
        self.road_half = 90
        self.stop_line = 70
        self.speed = 120.0  # px/sec
        self.spawn_prob_per_sec = 0.9  # per approach
        self.cars: List[Car] = []
        self.throughput = 0
        self.queues = {"N": 0, "S": 0, "E": 0, "W": 0}

    def reset(self):
        self.cars.clear()
        self.throughput = 0

    def spawn(self, dt: float):
        # Spawn cars probabilistically for each approach
        for lane in ["N", "S", "E", "W"]:
            if random.random() < self.spawn_prob_per_sec * dt:
                if lane == "N":
                    self.cars.append(Car(self.cx - 30, 40, 0, self.speed, lane))
                elif lane == "S":
                    self.cars.append(Car(self.cx + 30, self.h - 40, 0, -self.speed, lane))
                elif lane == "E":
                    self.cars.append(Car(self.w - 40, self.cy - 30, -self.speed, 0, lane, length=10, width=16))
                elif lane == "W":
                    self.cars.append(Car(40, self.cy + 30, self.speed, 0, lane, length=10, width=16))

    def _stop_line_pos(self, lane: str) -> float:
        # coordinate where cars should stop before intersection
        if lane == "N":
            return self.cy - self.stop_line
        if lane == "S":
            return self.cy + self.stop_line
        if lane == "E":
            return self.cx + self.stop_line
        if lane == "W":
            return self.cx - self.stop_line
        raise ValueError(lane)

    def step(self, dt: float, allow_ns: bool, allow_ew: bool):
        """
        allow_ns: predicted green for N/S directions
        allow_ew: predicted green for E/W directions
        """
        self.spawn(dt)

        # Move cars
        remaining: List[Car] = []
        for car in self.cars:
            allow = allow_ns if car.lane in ("N", "S") else allow_ew

            # Determine if car is approaching stop line and must stop
            if car.lane == "N" and not allow and car.y < self._stop_line_pos("N"):
                # Still far away, can roll forward
                pass
            elif car.lane == "N" and not allow and car.y >= self._stop_line_pos("N") - 5:
                # at stop line -> stop
                car.vx, car.vy = 0, 0
            elif car.lane == "S" and not allow and car.y <= self._stop_line_pos("S") + 5:
                car.vx, car.vy = 0, 0
            elif car.lane == "E" and not allow and car.x <= self._stop_line_pos("E") + 5:
                car.vx, car.vy = 0, 0
            elif car.lane == "W" and not allow and car.x >= self._stop_line_pos("W") - 5:
                car.vx, car.vy = 0, 0
            else:
                # If allowed, ensure moving speed
                if car.lane == "N":
                    car.vx, car.vy = 0, self.speed
                elif car.lane == "S":
                    car.vx, car.vy = 0, -self.speed
                elif car.lane == "E":
                    car.vx, car.vy = -self.speed, 0
                elif car.lane == "W":
                    car.vx, car.vy = self.speed, 0

            car.x += car.vx * dt
            car.y += car.vy * dt

            # Count throughput if car exits screen
            if car.x < -50 or car.x > self.w + 50 or car.y < -50 or car.y > self.h + 50:
                self.throughput += 1
            else:
                remaining.append(car)

        self.cars = remaining

    def draw(self, screen: pygame.Surface):
        # Roads
        screen.fill((25, 25, 25))
        road_color = (45, 45, 45)
        pygame.draw.rect(screen, road_color, (self.cx - self.road_half, 0, 2 * self.road_half, self.h))
        pygame.draw.rect(screen, road_color, (0, self.cy - self.road_half, self.w, 2 * self.road_half))

        # Lane markings
        line_color = (200, 200, 200)
        pygame.draw.line(screen, line_color, (self.cx, 0), (self.cx, self.h), 2)
        pygame.draw.line(screen, line_color, (0, self.cy), (self.w, self.cy), 2)

        # Stop lines
        stop_color = (255, 255, 255)
        pygame.draw.line(screen, stop_color, (self.cx - self.road_half, self.cy - self.stop_line), (self.cx + self.road_half, self.cy - self.stop_line), 3)  # N
        pygame.draw.line(screen, stop_color, (self.cx - self.road_half, self.cy + self.stop_line), (self.cx + self.road_half, self.cy + self.stop_line), 3)  # S
        pygame.draw.line(screen, stop_color, (self.cx + self.stop_line, self.cy - self.road_half), (self.cx + self.stop_line, self.cy + self.road_half), 3)  # E
        pygame.draw.line(screen, stop_color, (self.cx - self.stop_line, self.cy - self.road_half), (self.cx - self.stop_line, self.cy + self.road_half), 3)  # W

        # Cars
        for car in self.cars:
            pygame.draw.rect(screen, (80, 170, 255), car.rect())


# ----------------------------
# Model wrapper for predictions
# ----------------------------
class Algo:
    def name(self) -> str:
        raise NotImplementedError

    def predict_label_index(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NBAlgo(Algo):
    def __init__(self, model: GaussianNB):
        self.model = model

    def name(self) -> str:
        return "GaussianNB"

    def predict_label_index(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class GA_NBAlgo(Algo):
    def __init__(self, model: GaussianNB, best_log10: float, best_val_f1: float):
        self.model = model
        self.best_log10 = best_log10
        self.best_val_f1 = best_val_f1

    def name(self) -> str:
        return f"GA-tuned NB (log10(vs)={self.best_log10:.2f})"

    def predict_label_index(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class QAlgo(Algo):
    def __init__(self, qb: QBandit, epochs: int, bins: int):
        self.qb = qb
        self.epochs = epochs
        self.bins = bins

    def name(self) -> str:
        return f"Q-learning bandit (epochs={self.epochs}, bins={self.bins})"

    def predict_label_index(self, X: np.ndarray) -> np.ndarray:
        return self.qb.predict(X)


class MLPAlgo(Algo):
    def __init__(self, mlp: MLPClassifier, hidden: Tuple[int, ...], max_iter: int):
        self.mlp = mlp
        self.hidden = hidden
        self.max_iter = max_iter

    def name(self) -> str:
        return f"MLP {self.hidden} (max_iter={self.max_iter})"

    def predict_label_index(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict(X)


def build_algorithms(
    Xtr_s: np.ndarray, ytr: np.ndarray,
    Xva_s: np.ndarray, yva: np.ndarray,
    seed: int,
    rl_epochs: int, rl_bins: int,
    ga_pop: int, ga_gens: int,
    nn_hidden: Tuple[int, ...], nn_max_iter: int, nn_batch: int,
) -> List[Algo]:
    # NB
    nb = GaussianNB()
    nb.fit(Xtr_s, ytr)

    # Q-bandit
    qb = QBandit(n_actions=int(np.max(ytr)) + 1, n_bins=rl_bins, seed=seed)
    qb.train(Xtr_s, ytr, epochs=rl_epochs)

    # GA-tuned NB
    best_log10, best_val_f1 = ga_tune_var_smoothing(Xtr_s, ytr, Xva_s, yva, seed=seed, pop=ga_pop, gens=ga_gens)
    ga_nb = GaussianNB(var_smoothing=10.0 ** best_log10)
    ga_nb.fit(Xtr_s, ytr)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=nn_hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=nn_batch,
        learning_rate_init=1e-3,
        max_iter=nn_max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(Xtr_s, ytr)

    return [
        NBAlgo(nb),
        QAlgo(qb, rl_epochs, rl_bins),
        GA_NBAlgo(ga_nb, best_log10, best_val_f1),
        MLPAlgo(mlp, nn_hidden, nn_max_iter),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to lisa_features.npz")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rl-epochs", type=int, default=3)
    ap.add_argument("--rl-bins", type=int, default=6)

    ap.add_argument("--ga-pop", type=int, default=18)
    ap.add_argument("--ga-gens", type=int, default=10)

    ap.add_argument("--nn-hidden", type=str, default="256,128")
    ap.add_argument("--nn-max-iter", type=int, default=40)
    ap.add_argument("--nn-batch", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X, y, labels, groups = load_npz(args.features)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, groups=groups, seed=args.seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # Build pools of test indices by label name bucket
    label_names = list(labels)
    name_to_idx = {n: i for i, n in enumerate(label_names)}

    # For selecting realistic examples, keep indices by label name
    indices_by_name: Dict[str, np.ndarray] = {}
    yte_names = np.array([label_names[i] for i in yte])
    for name in label_names:
        indices_by_name[name] = np.where(yte_names == name)[0]

    # Build algos (takes a bit the first time)
    hidden = tuple(int(x) for x in args.nn_hidden.split(",") if x.strip())
    algos = build_algorithms(
        Xtr_s, ytr, Xva_s, yva,
        seed=args.seed,
        rl_epochs=args.rl_epochs, rl_bins=args.rl_bins,
        ga_pop=args.ga_pop, ga_gens=args.ga_gens,
        nn_hidden=hidden, nn_max_iter=args.nn_max_iter, nn_batch=args.nn_batch,
    )
    algo_i = 0

    # Signal program (truth)
    # Cycle: NS green 10s -> yellow 2s -> all-red 1s -> EW green 10s -> yellow 2s -> all-red 1s
    program = [
        ("NS_GREEN", 10.0),
        ("NS_YELLOW", 2.0),
        ("ALL_RED", 1.0),
        ("EW_GREEN", 10.0),
        ("EW_YELLOW", 2.0),
        ("ALL_RED", 1.0),
    ]
    prog_idx = 0
    prog_t = 0.0

    sim = IntersectionSim()
    rates = Rates()

    pygame.init()
    screen = pygame.display.set_mode((sim.w, sim.h))
    pygame.display.set_caption("LISA Traffic Demo: models controlling movement via predicted light state")
    font = pygame.font.SysFont("Menlo", 18)
    clock = pygame.time.Clock()

    paused = False
    last_switch = 0.0

    def truth_for_direction(state: str, direction: str) -> str:
        # direction is "NS" or "EW"
        if state.startswith("NS_"):
            return "GREEN" if direction == "NS" and "GREEN" in state else ("YELLOW" if direction == "NS" else "RED")
        if state.startswith("EW_"):
            return "GREEN" if direction == "EW" and "GREEN" in state else ("YELLOW" if direction == "EW" else "RED")
        # ALL_RED
        return "RED"

    def sample_feature_for_truth(truth_bucket: str) -> np.ndarray:
        pool_names = TRUTH_TO_LABELPOOL[truth_bucket]
        # pick a label name that has samples
        candidates = [n for n in pool_names if indices_by_name.get(n, np.array([])).size > 0]
        chosen_name = random.choice(candidates)
        j = int(random.choice(indices_by_name[chosen_name]))
        return Xte_s[j : j + 1]  # shape (1, d)

    def draw_light(x: int, y: int, truth: str, pred: str):
        # truth and pred are in {"GREEN","YELLOW","RED"}
        # draw two circles: left=truth, right=pred
        def col(s: str):
            if s == "GREEN":
                return (50, 220, 80)
            if s == "YELLOW":
                return (240, 220, 70)
            return (240, 70, 70)

        pygame.draw.circle(screen, col(truth), (x - 12, y), 10)
        pygame.draw.circle(screen, col(pred), (x + 12, y), 10)
        pygame.draw.circle(screen, (0, 0, 0), (x - 12, y), 10, 2)
        pygame.draw.circle(screen, (0, 0, 0), (x + 12, y), 10, 2)

    while True:
        dt = clock.tick(60) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_r:
                    sim.reset()
                    rates = Rates()
                if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    algo_i = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}[ev.key]
                    last_switch = time.time()

        if not paused:
            # advance truth signal program
            prog_t += dt
            if prog_t >= program[prog_idx][1]:
                prog_t = 0.0
                prog_idx = (prog_idx + 1) % len(program)
            truth_state = program[prog_idx][0]

            algo = algos[algo_i]

            # For each direction, sample a realistic LISA example matching the truth bucket
            truth_ns = truth_for_direction(truth_state, "NS")  # GREEN/YELLOW/RED
            truth_ew = truth_for_direction(truth_state, "EW")

            X_ns = sample_feature_for_truth(truth_ns)
            X_ew = sample_feature_for_truth(truth_ew)

            pred_ns_idx = int(algo.predict_label_index(X_ns)[0])
            pred_ew_idx = int(algo.predict_label_index(X_ew)[0])

            pred_ns_name = label_names[pred_ns_idx]
            pred_ew_name = label_names[pred_ew_idx]

            pred_ns_bucket = "GREEN" if label_to_move_hold(pred_ns_name) == "MOVE" else "RED"
            pred_ew_bucket = "GREEN" if label_to_move_hold(pred_ew_name) == "MOVE" else "RED"

            # Update rates (MOVE/HOLD level)
            # Truth MOVE is GREEN for that direction; HOLD is RED/YELLOW (we treat YELLOW as HOLD)
            def upd(truth_bucket: str, pred_bucket: str):
                nonlocal rates
                true_move = (truth_bucket == "GREEN")
                pred_move = (pred_bucket == "GREEN")
                if true_move:
                    rates.total_true_move += 1
                    if not pred_move:
                        rates.wasted_green += 1
                else:
                    rates.total_true_hold += 1
                    if pred_move:
                        rates.illegal_go += 1

            upd(truth_ns, pred_ns_bucket)
            upd(truth_ew, pred_ew_bucket)

            # Cars move based on predicted “green” for each direction
            allow_ns = (pred_ns_bucket == "GREEN")
            allow_ew = (pred_ew_bucket == "GREEN")
            sim.step(dt, allow_ns=allow_ns, allow_ew=allow_ew)

        # Draw scene
        sim.draw(screen)

        # Draw lights near corners of intersection:
        truth_state = program[prog_idx][0]
        truth_ns = truth_for_direction(truth_state, "NS")
        truth_ew = truth_for_direction(truth_state, "EW")

        # We don’t store pred buckets across pause; show last known by reusing the current algo quickly
        algo = algos[algo_i]
        X_ns = sample_feature_for_truth(truth_ns)
        X_ew = sample_feature_for_truth(truth_ew)
        pred_ns_idx = int(algo.predict_label_index(X_ns)[0])
        pred_ew_idx = int(algo.predict_label_index(X_ew)[0])
        pred_ns_name = label_names[pred_ns_idx]
        pred_ew_name = label_names[pred_ew_idx]
        pred_ns_bucket = "GREEN" if label_to_move_hold(pred_ns_name) == "MOVE" else "RED"
        pred_ew_bucket = "GREEN" if label_to_move_hold(pred_ew_name) == "MOVE" else "RED"

        draw_light(sim.cx, sim.cy - 120, truth_ns, pred_ns_bucket)  # NS light
        draw_light(sim.cx + 120, sim.cy, truth_ew, pred_ew_bucket)  # EW light

        # HUD
        hud_lines = [
            "Keys: 1=NB  2=Q  3=GA-NB  4=MLP | space=pause | r=reset | esc=quit",
            f"Algorithm: {algos[algo_i].name()}",
            f"Truth phase: {truth_state}",
            f"Throughput (cars exited): {sim.throughput}",
            f"Wasted green rate: {rates.wasted_green_rate():.3f}   Illegal go rate: {rates.illegal_go_rate():.3f}",
            f"Throughput factor≈ {rates.throughput_factor():.3f}",
            f"Note: LEFT circle=TRUTH, RIGHT circle=PRED (for each signal).",
        ]
        y0 = 10
        for line in hud_lines:
            surf = font.render(line, True, (235, 235, 235))
            screen.blit(surf, (10, y0))
            y0 += 22

        # If illegal_go is happening a lot, flash a warning
        if rates.illegal_go_rate() > 0.05:
            warn = font.render("WARNING: illegal_go is high (unsafe 'go on red')", True, (255, 90, 90))
            screen.blit(warn, (10, y0 + 10))

        pygame.display.flip()


if __name__ == "__main__":
    main()
