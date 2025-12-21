import argparse
import copy
import importlib
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


CycleConfig = Dict[str, Dict[str, List[int] | int]]

NUM_PHASES = 4
MIN_PHASE_DURATION = 5
MIN_CYCLE_LENGTH = 40
MAX_CYCLE_LENGTH = 180

SIM_MAIN_OVERRIDE: Optional[Path] = None


def _locate_builders() -> Tuple[Callable[[Optional[CycleConfig]], object], Optional[Callable[[Optional[CycleConfig]], object]]]:
    """
    Try to import build_sim and run_visual from likely modules.
    Raises ImportError with a helpful message if they are not available.
    """
    candidate_modules = ["main", "trafficSim.main", "trafficSim"]
    last_error: Optional[Exception] = None

    def _import_from_path(path: Path, name: str):
        if not path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # Keep trying other options if this fails
            nonlocal last_error
            last_error = exc
            return None

    # First, explicit override: a set_sim_main_path call or env var.
    global SIM_MAIN_OVERRIDE
    env_main = os.environ.get("TRAFFIC_SIM_MAIN")
    candidates_from_override = []
    if SIM_MAIN_OVERRIDE:
        candidates_from_override.append(SIM_MAIN_OVERRIDE)
    if env_main:
        candidates_from_override.append(Path(env_main))

    for candidate in candidates_from_override:
        module = _import_from_path(Path(candidate), "_sim_env_main")
        if module:
            build_sim = getattr(module, "build_sim", None)
            if build_sim:
                run_visual = getattr(module, "run_visual", None)
                return build_sim, run_visual

    # Next, module imports by name.
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            build_sim = getattr(module, "build_sim")
            run_visual = getattr(module, "run_visual", None)
            return build_sim, run_visual
        except Exception as exc:  # ImportError, AttributeError, etc.
            last_error = exc
            continue

    base_dir = Path(__file__).resolve().parent
    file_candidates = [
        base_dir / "main.py",
        base_dir / "Traffic-Simulation-main" / "main.py",
    ]

    # Light recursive search near the project root for a main.py containing build_sim.
    max_depth = 3
    for ancestor in list(base_dir.parents)[:max_depth]:
        for candidate in ancestor.glob("**/main.py"):
            if "site-packages" in str(candidate):
                continue
            file_candidates.append(candidate)

    for idx, candidate in enumerate(file_candidates):
        module = _import_from_path(candidate, f"_sim_local_{idx}")
        if module is None:
            continue
        build_sim = getattr(module, "build_sim", None)
        if build_sim is None:
            continue
        run_visual = getattr(module, "run_visual", None)
        return build_sim, run_visual

    raise ImportError(
        "Could not locate build_sim/run_visual. Set TRAFFIC_SIM_MAIN or call set_sim_main_path "
        "with your simulation main.py, or place the file alongside ga_optimizer.py."
    ) from last_error


def _random_phase_durations(cycle_length: int) -> List[int]:
    """Create a list of NUM_PHASES durations that sum to cycle_length and honor MIN_PHASE_DURATION."""
    base = [MIN_PHASE_DURATION] * NUM_PHASES
    remaining = max(0, cycle_length - MIN_PHASE_DURATION * NUM_PHASES)
    # Randomly distribute the remaining time across phases.
    cuts = sorted([random.randint(0, remaining) for _ in range(NUM_PHASES - 1)])
    parts = []
    prev = 0
    for c in cuts:
        parts.append(c - prev)
        prev = c
    parts.append(remaining - prev)
    phases = [b + p for b, p in zip(base, parts)]
    return _normalize_phases(phases, cycle_length)


def _normalize_phases(phases: List[int], cycle_length: int) -> List[int]:
    """Clamp phase durations to minimums and force the sum to equal cycle_length."""
    phases = [max(MIN_PHASE_DURATION, int(round(p))) for p in phases]
    total = sum(phases)
    if total == 0:
        phases = [cycle_length // NUM_PHASES] * NUM_PHASES
        phases[0] += cycle_length - sum(phases)
        return phases

    diff = cycle_length - total
    safeguard = 0
    while diff != 0 and safeguard < 1000:
        idx = random.randrange(NUM_PHASES)
        if diff > 0:
            phases[idx] += 1
            diff -= 1
        else:
            if phases[idx] > MIN_PHASE_DURATION:
                phases[idx] -= 1
                diff += 1
        safeguard += 1

    if sum(phases) != cycle_length:
        phases[0] += cycle_length - sum(phases)
        phases = [max(MIN_PHASE_DURATION, p) for p in phases]
    return phases


def initialize_population(population_size: int) -> List[CycleConfig]:
    """Create an initial population of random cycle configurations."""
    population: List[CycleConfig] = []
    for _ in range(population_size):
        cycle_length = random.randint(MIN_CYCLE_LENGTH, MAX_CYCLE_LENGTH)
        phases = _random_phase_durations(cycle_length)
        individual: CycleConfig = {
            "signal0": {"cycle_length": cycle_length, "phase_durations": phases},
            "signal1": {"cycle_length": cycle_length, "phase_durations": phases.copy()},
        }
        population.append(individual)
    return population


def _run_simulation(sim: object, steps: int) -> None:
    """Run the simulation for a given number of steps using whatever interface is available."""
    if hasattr(sim, "run") and callable(getattr(sim, "run")):
        try:
            sim.run(steps)
            return
        except TypeError:
            # Some run implementations accept no args; fall through to manual stepping.
            pass

    step_fn = None
    for candidate in ("step", "update", "__call__"):
        fn = getattr(sim, candidate, None)
        if callable(fn):
            step_fn = fn
            break

    if step_fn is None:
        raise RuntimeError("Simulation object has no recognized stepping method (run/step/update).")

    for _ in range(steps):
        step_fn()


def _estimate_average_queue(sim: object) -> float:
    """Estimate congestion via average queue length across roads/segments."""
    queues: List[int] = []
    for attr in ("roads", "segments"):
        roads = getattr(sim, attr, None)
        if roads is None:
            continue
        for road in roads:
            vehicles = getattr(road, "vehicles", None)
            if vehicles is None:
                continue
            try:
                queues.append(len(vehicles))
            except TypeError:
                # vehicles may be a property, not sized; ignore.
                continue

    if queues:
        return float(sum(queues)) / len(queues)

    vehicles = getattr(sim, "vehicles", None)
    if vehicles is None:
        return 0.0
    if isinstance(vehicles, (list, tuple, set, dict)):
        return float(len(vehicles))
    try:
        return float(len(vehicles))
    except Exception:
        return 0.0


def _estimate_throughput(sim: object) -> float:
    """Estimate number of vehicles that finished the network."""
    candidate_attrs = [
        "arrived_vehicles",
        "completed_vehicles",
        "vehicles_done",
        "vehicles_arrived",
        "exited_vehicles",
    ]
    for attr in candidate_attrs:
        val = getattr(sim, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple, set, dict)):
            return float(len(val))
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def evaluate_cycle_config(cycle_config: CycleConfig, sim_steps: int = 2000) -> float:
    """
    Run a headless simulation for the given cycle configuration and return a fitness score.
    Lower fitness is better: high throughput and low queues produce smaller values.
    """
    build_sim, _ = _locate_builders()
    sim = build_sim(cycle_config)

    _run_simulation(sim, sim_steps)

    avg_queue = _estimate_average_queue(sim)
    throughput = _estimate_throughput(sim)

    fitness = avg_queue - 0.1 * throughput
    if math.isnan(fitness) or math.isinf(fitness):
        fitness = float("inf")
    return float(fitness)


def select_parents(population: List[CycleConfig], fitnesses: List[float], tournament_size: int = 3) -> Tuple[CycleConfig, CycleConfig]:
    """Tournament selection returning two parents."""
    def _tournament() -> CycleConfig:
        contenders = random.sample(list(zip(population, fitnesses)), k=tournament_size)
        return min(contenders, key=lambda pair: pair[1])[0]

    return copy.deepcopy(_tournament()), copy.deepcopy(_tournament())


def crossover(parent_a: CycleConfig, parent_b: CycleConfig, crossover_rate: float) -> CycleConfig:
    """Mix timing parameters between two parents."""
    if random.random() > crossover_rate:
        return copy.deepcopy(parent_a)

    child: CycleConfig = {}
    for signal_key in ("signal0", "signal1"):
        sig_a = parent_a[signal_key]
        sig_b = parent_b[signal_key]
        cycle_length = random.choice([sig_a["cycle_length"], sig_b["cycle_length"]])
        phases = [
            random.choice([sig_a["phase_durations"][i], sig_b["phase_durations"][i]])
            for i in range(NUM_PHASES)
        ]
        phases = _normalize_phases(phases, cycle_length)
        child[signal_key] = {"cycle_length": cycle_length, "phase_durations": phases}
    return child


def mutate(individual: CycleConfig, mutation_rate: float) -> CycleConfig:
    """Randomly perturb cycle length and phase durations."""
    if random.random() > mutation_rate:
        return copy.deepcopy(individual)

    mutant = copy.deepcopy(individual)
    for signal_key in ("signal0", "signal1"):
        sig = mutant[signal_key]
        if random.random() < 0.5:
            delta = random.randint(-10, 10)
            sig["cycle_length"] = int(
                max(MIN_CYCLE_LENGTH, min(MAX_CYCLE_LENGTH, sig["cycle_length"] + delta))
            )

        phases = sig["phase_durations"]
        for i in range(NUM_PHASES):
            if random.random() < 0.3:
                phases[i] += random.randint(-5, 5)

        sig["phase_durations"] = _normalize_phases(phases, sig["cycle_length"])
    return mutant


def optimize_signal_timings_ga(
    population_size: int = 30,
    generations: int = 200,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> CycleConfig:
    """
    Run a genetic algorithm to search for traffic-signal timings.
    Returns the best cycle_config discovered.
    """
    if random_seed is not None:
        random.seed(random_seed)

    population = initialize_population(population_size)
    best_individual: Optional[CycleConfig] = None
    best_fitness = float("inf")

    for gen in range(generations):
        fitnesses = [evaluate_cycle_config(ind) for ind in population]

        gen_best_idx, gen_best_fit = min(enumerate(fitnesses), key=lambda pair: pair[1])
        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_individual = copy.deepcopy(population[gen_best_idx])

        if verbose:
            print(f"Generation {gen + 1}/{generations} - best fitness: {gen_best_fit:.4f}")

        next_pop: List[CycleConfig] = []
        while len(next_pop) < population_size:
            parent_a, parent_b = select_parents(population, fitnesses)
            child = crossover(parent_a, parent_b, crossover_rate)
            child = mutate(child, mutation_rate)
            next_pop.append(child)
        population = next_pop

    assert best_individual is not None
    if verbose:
        print("Best configuration found:", best_individual)
    return best_individual


def set_sim_main_path(path: str | Path) -> None:
    """Manually set the simulation main.py path for builder discovery."""
    global SIM_MAIN_OVERRIDE
    SIM_MAIN_OVERRIDE = Path(path).resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic algorithm optimizer for traffic signals.")
    parser.add_argument(
        "--sim-path",
        type=str,
        help="Path to simulation main.py containing build_sim/run_visual (optional).",
    )
    parser.add_argument("--population-size", type=int, default=30, help="Population size.")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations.")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate.")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging.")
    args = parser.parse_args()

    if args.sim_path:
        set_sim_main_path(args.sim_path)

    best_config = optimize_signal_timings_ga(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        random_seed=args.seed,
        verbose=not args.quiet,
    )
    print("Best config:", best_config)
    try:
        _, run_visual = _locate_builders()
        if run_visual is not None:
            run_visual(best_config)
    except ImportError:
        # Simulation visualization is optional; skip if we cannot import it.
        pass
