import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Simple, lightweight simulation scaffold so the GA can run without the original trafficSim.
# It models four inbound roads and two signals. Each signal alternates four phases whose
# durations are configurable via cycle_config.


@dataclass
class Road:
    name: str
    vehicles: List[int] = field(default_factory=list)


@dataclass
class TrafficSignal:
    cycle_length: int
    phase_durations: List[int]
    controlled_roads: List[int]
    phase_index: int = 0
    phase_elapsed: int = 0

    def step(self):
        self.phase_elapsed += 1
        if self.phase_elapsed >= self.phase_durations[self.phase_index]:
            self.phase_index = (self.phase_index + 1) % len(self.phase_durations)
            self.phase_elapsed = 0

    def is_green(self, road_idx: int) -> bool:
        # For this simplified model, phases alternate which controlled road is green.
        if road_idx not in self.controlled_roads:
            return False
        local = self.controlled_roads.index(road_idx)
        return self.phase_index % len(self.controlled_roads) == local


class Simulation:
    def __init__(self):
        self.roads: List[Road] = [Road(f"road{i}") for i in range(4)]
        self.signals: List[TrafficSignal] = []
        self.arrived_vehicles: List[int] = []
        self.t = 0

    def create_signal(self, cycle_length: int, phase_durations: List[int], controlled_roads: List[int]) -> TrafficSignal:
        sig = TrafficSignal(cycle_length=cycle_length, phase_durations=phase_durations, controlled_roads=controlled_roads)
        self.signals.append(sig)
        return sig

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def step(self):
        # Spawn vehicles (Poisson-like) on each road
        for road in self.roads:
            if random.random() < 0.3:
                road.vehicles.append(1)

        # Advance signals
        for signal in self.signals:
            signal.step()

        # Move vehicles through green lights
        service_rate = 2  # vehicles per step if green
        for sig in self.signals:
            for ridx in sig.controlled_roads:
                road = self.roads[ridx]
                if sig.is_green(ridx):
                    moved = min(service_rate, len(road.vehicles))
                    for _ in range(moved):
                        road.vehicles.pop(0)
                        self.arrived_vehicles.append(1)

        self.t += 1


def _apply_signal_config(signals: List[TrafficSignal], cycle_config: Dict) -> None:
    """
    Apply GA-generated cycle configuration to the list of signals.
    Expects keys 'signal0', 'signal1' mapped to cycle_length and phase_durations.
    """
    for idx, key in enumerate(["signal0", "signal1"]):
        if key not in cycle_config or idx >= len(signals):
            continue
        cfg = cycle_config[key]
        signals[idx].cycle_length = int(cfg.get("cycle_length", signals[idx].cycle_length))
        signals[idx].phase_durations = list(cfg.get("phase_durations", signals[idx].phase_durations))
        signals[idx].phase_index = 0
        signals[idx].phase_elapsed = 0


def build_sim(cycle_config: Optional[Dict] = None) -> Simulation:
    """
    Build a minimal simulation with two signals controlling four inbound roads.
    """
    default_cycle = {"cycle_length": 90, "phase_durations": [25, 20, 25, 20]}
    sim = Simulation()
    # signal0 controls road0 and road1
    sig0_cfg = cycle_config.get("signal0", default_cycle) if cycle_config else default_cycle
    sig1_cfg = cycle_config.get("signal1", default_cycle) if cycle_config else default_cycle

    sig0 = sim.create_signal(
        cycle_length=int(sig0_cfg["cycle_length"]),
        phase_durations=list(sig0_cfg["phase_durations"]),
        controlled_roads=[0, 1],
    )
    sig1 = sim.create_signal(
        cycle_length=int(sig1_cfg["cycle_length"]),
        phase_durations=list(sig1_cfg["phase_durations"]),
        controlled_roads=[2, 3],
    )

    if cycle_config:
        _apply_signal_config([sig0, sig1], cycle_config)

    return sim


def run_visual(cycle_config: Optional[Dict] = None) -> None:
    """
    Simple Tkinter visualization showing queue lengths and signal phases over time.
    """
    import tkinter as tk

    sim = build_sim(cycle_config)
    steps = 2000
    delay_ms = 30

    root = tk.Tk()
    root.title("Traffic Signal GA - Visualization")
    canvas = tk.Canvas(root, width=520, height=320, bg="white")
    canvas.pack(padx=10, pady=10)

    road_positions = {
        0: (40, 250),
        1: (150, 250),
        2: (260, 250),
        3: (370, 250),
    }

    def draw_state(step: int):
        canvas.delete("all")
        canvas.create_text(10, 10, anchor="nw", text=f"Step: {step}/{steps}")
        canvas.create_text(10, 30, anchor="nw", text=f"Arrived vehicles: {len(sim.arrived_vehicles)}")

        # Draw roads as vertical bars representing queue lengths
        max_queue_display = 25
        for ridx, (x, y_base) in road_positions.items():
            queue_len = len(sim.roads[ridx].vehicles)
            bar_height = min(queue_len * 5, max_queue_display * 5)
            canvas.create_rectangle(x, y_base - bar_height, x + 60, y_base, fill="#6fa8dc", outline="black")
            canvas.create_text(x + 30, y_base - bar_height - 12, text=f"Road {ridx}", anchor="s")
            canvas.create_text(x + 30, y_base + 14, text=f"Queue: {queue_len}", anchor="n")

        # Draw signal states
        for sidx, sig in enumerate(sim.signals):
            color = "#6aa84f" if sig.phase_index % 2 == 0 else "#cc0000"
            cx = 100 + sidx * 220
            cy = 120
            canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill=color, outline="black")
            canvas.create_text(cx, cy + 32, text=f"Signal {sidx}", anchor="n")
            canvas.create_text(cx, cy + 48, text=f"Phase {sig.phase_index + 1}/{len(sig.phase_durations)}", anchor="n")
            canvas.create_text(cx, cy + 64, text=f"Time in phase: {sig.phase_elapsed}", anchor="n")

    def tick(step: int = 0):
        if step >= steps:
            canvas.create_text(260, 300, text="Finished. Close window to exit.", anchor="s")
            return
        sim.step()
        draw_state(step)
        root.after(delay_ms, lambda: tick(step + 1))

    draw_state(0)
    root.after(delay_ms, lambda: tick(1))
    root.mainloop()


if __name__ == "__main__":
    sim = build_sim()
    sim.run(steps=100)
    print("Arrived vehicles:", len(sim.arrived_vehicles))
