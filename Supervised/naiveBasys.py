from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------- CONFIG ---------
from pathlib import Path

# Folder where this script lives: .../Supervised
SCRIPT_DIR = Path(__file__).resolve().parent

# Data directory = same folder as this script
DATA_DIR = SCRIPT_DIR

# Where processed arrays (npz / npy) will be saved
ROOT = SCRIPT_DIR / "2021"

SENSOR_SAMPLE = 200
HORIZON_STEPS = 6  # predict 30 minutes ahead
BINS = 3
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [6, 12]
TRAIN_FRAC, VAL_FRAC = 0.75, 0.10

# Traffic signal optimization parameters
SIGNAL_CYCLE_TIME = 120  # seconds (2 minutes)
MIN_GREEN_TIME = 15      # minimum green phase
MAX_GREEN_TIME = 90      # maximum green phase

# Ensure processed-data folder exists
ROOT.mkdir(parents=True, exist_ok=True)


# --------- LOAD DATA ---------
def try_load_processed(root: Path):
    """
    Try to load already-processed flow arrays from ROOT.
    """
    # Look for NPZ files first
    for p in [root / "flow_2021.npz", root / "flow.npz", root / "processed.npz"]:
        if p.exists():
            print(f"Loading processed data from {p}")
            z = np.load(p)
            if "flow" in z:
                return z["flow"]
            if "x" in z and "y" in z:
                # If you ever want to reconstruct from x/y, do it here
                X = z["x"]
                return None

    # Fallback: NPY files
    for p in [root / "flow_2021.npy", root / "flow.npy"]:
        if p.exists():
            print(f"Loading processed data from {p}")
            return np.load(p)

    return None


flow = try_load_processed(ROOT)

if flow is None:
    import glob
    import h5py

    # Look for your raw 2021 HDF5 file in the same folder as this script
    h5_candidates = (
        glob.glob(str(DATA_DIR / "*2021*.h5")) +
        glob.glob(str(DATA_DIR / "*_2021.h5"))
    )

    if not h5_candidates:
        raise FileNotFoundError(
            f"Couldn't find processed arrays or a 2021 .h5 in {DATA_DIR}. "
            f"Make sure 'ca_his_raw_2021.h5' is placed there."
        )

    h5_path = h5_candidates[0]
    print(f"Using HDF5 file: {h5_path}")

    def load_flow_from_h5(path: str):
        # ---- Try via pandas first (any key) ----
        try:
            with pd.HDFStore(path, "r") as store:
                print("Available HDF5 keys:", list(store.keys()))
                best_df = None
                best_shape = (0, 0)

                for key in store.keys():
                    try:
                        obj = store[key]
                        if isinstance(obj, pd.DataFrame):
                            print(f"Key {key}: DataFrame shape={obj.shape}")
                            # Pick the biggest DataFrame
                            if obj.shape[0] > best_shape[0] and obj.shape[1] > 1:
                                best_df = obj
                                best_shape = obj.shape
                    except Exception as e:
                        print(f"   Could not read key {key}: {e}")

                if best_df is not None and best_shape[0] > 1000:
                    best_df = best_df.sort_index()
                    return best_df.to_numpy()
        except Exception as e:
            print("Pandas HDFStore failed:", e)

        # ---- Fallback: scan datasets with h5py ----
        print("Falling back to raw HDF5 inspection with h5py...")

        with h5py.File(path, "r") as f:
            def iter_datasets(group, prefix=""):
                for name, item in group.items():
                    full_name = f"{prefix}/{name}" if prefix else f"/{name}"
                    if isinstance(item, h5py.Dataset):
                        yield full_name, item
                    elif isinstance(item, h5py.Group):
                        yield from iter_datasets(item, full_name)

            best_array = None
            best_shape = (0, 0)
            for name, ds in iter_datasets(f):
                try:
                    shape = ds.shape
                    print(f"Dataset {name}: shape={shape}, ndim={ds.ndim}")
                    # We want a large 2D dataset (time × sensors)
                    if ds.ndim == 2 and shape[0] > best_shape[0] and shape[1] > 1:
                        best_array = ds[...]
                        best_shape = shape
                except Exception as e:
                    print(f"   Could not read dataset {name}: {e}")

            if best_array is None or best_shape[0] <= 1000:
                raise RuntimeError(
                    f"Could not find a suitable 2D dataset in {path}. "
                    f"Inspect the file manually to see its structure."
                )

            return best_array

    flow = load_flow_from_h5(h5_path)
    T, N = flow.shape
    print(f"Flow matrix shape: T={T}, N={N}")


# Subsample sensors
rng = np.random.default_rng(42)
if N > SENSOR_SAMPLE:
    cols = np.sort(rng.choice(N, size=SENSOR_SAMPLE, replace=False))
    flow = flow[:, cols]
    N = flow.shape[1]

# --------- FEATURE ENGINEERING ---------
def build_supervised(flow, horizon, lags, roll_windows):
    T, N = flow.shape
    df = pd.DataFrame(flow)
    feats = []
    feat_names = []
    
    for L in lags:
        lagged = df.shift(L).to_numpy()
        feats.append(lagged)
        feat_names.append(f"lag{L}")
    
    for w in roll_windows:
        rmean = df.rolling(w).mean().to_numpy()
        rstd  = df.rolling(w).std().to_numpy()
        feats.append(rmean); feat_names.append(f"rmean{w}")
        feats.append(rstd);  feat_names.append(f"rstd{w}")

    X_all = np.stack(feats, axis=-1)
    y_reg = df.shift(-horizon).to_numpy()

    valid_t = ~np.isnan(X_all).any(axis=(1,2)) & ~np.isnan(y_reg).any(axis=1)
    X_all = X_all[valid_t]
    y_reg = y_reg[valid_t]
    return X_all, y_reg, valid_t, feat_names

X_all, y_reg, mask, feat_names = build_supervised(flow, HORIZON_STEPS, LAGS, ROLL_WINDOWS)
T2 = X_all.shape[0]
F  = X_all.shape[2]
print(f"Supervised tensor: (T2={T2}, N={N}, F={F})")

# Flatten
X = X_all.reshape(T2 * N, F)
y_cont = y_reg.reshape(T2 * N)

# Bin to classes
y = pd.qcut(y_cont, q=BINS, labels=False, duplicates='drop')
valid_rows = ~pd.isna(y)
X = X[valid_rows]
y = y[valid_rows].astype(int)

# Scale
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# --------- TIME-AWARE SPLIT ---------
times_compact = np.repeat(np.arange(T2), N)[valid_rows]
t_train_end = int(TRAIN_FRAC * T2)
t_val_end   = int((TRAIN_FRAC + VAL_FRAC) * T2)

train_idx = times_compact <  t_train_end
val_idx   = (times_compact >= t_train_end) & (times_compact < t_val_end)
test_idx  = times_compact >= t_val_end

Xtr, ytr = X_scaled[train_idx], y[train_idx]
Xva, yva = X_scaled[val_idx], y[val_idx]
Xte, yte = X_scaled[test_idx], y[test_idx]

print(f"Train/Val/Test sizes: {len(ytr)}/{len(yva)}/{len(yte)} (classes: {np.unique(y)})")

# --------- TRAIN NAIVE BAYES ---------
nb = GaussianNB()
nb.fit(Xtr, ytr)

# --------- EVALUATE ---------
def eval_split(name, Xs, ys):
    yhat = nb.predict(Xs)
    print(f"\n{name} F1(macro): {f1_score(ys, yhat, average='macro'):.3f}")
    print(classification_report(ys, yhat, digits=3))
    print("Confusion matrix:\n", confusion_matrix(ys, yhat))
    return yhat

print("\n" + "="*60)
print("PREDICTION MODEL EVALUATION")
print("="*60)
yhat_val = eval_split("Validation", Xva, yva)
yhat_test = eval_split("Test", Xte, yte)

# --------- TRAFFIC SIGNAL OPTIMIZATION ---------
print("\n" + "="*60)
print("TRAFFIC SIGNAL OPTIMIZATION")
print("="*60)

# Reshape predictions back to (time, sensor) format
test_times = times_compact[test_idx]
unique_times = np.unique(test_times)

# For each time step in test set, get predictions per sensor
predictions_by_time = {}
for t in unique_times:
    mask_t = test_times == t
    preds_t = yhat_test[mask_t]
    predictions_by_time[t] = preds_t

def optimize_signal_timing(predicted_classes):
    """
    Allocate green time based on predicted traffic classes.
    
    Args:
        predicted_classes: array of predicted classes (0=low, 1=med, 2=high)
    
    Returns:
        green_time: optimal green light duration in seconds
    """
    # Map classes to demand weights
    demand_weights = {0: 0.5, 1: 1.0, 2: 1.5}
    
    # Calculate weighted demand
    total_demand = sum(demand_weights[c] for c in predicted_classes)
    avg_demand = total_demand / len(predicted_classes)
    
    # Allocate time proportional to demand
    # Base green time scaled by demand ratio
    base_time = SIGNAL_CYCLE_TIME / 2  # 60 seconds baseline
    optimal_time = base_time * (avg_demand / 1.0)  # normalized to medium demand
    
    # Enforce limits
    optimal_time = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, optimal_time))
    
    return optimal_time

# Simulate baseline (fixed timing) vs optimized
baseline_green_time = SIGNAL_CYCLE_TIME / 2  # fixed 60 seconds
optimized_green_times = []
traffic_classes = []

for t in sorted(predictions_by_time.keys())[:100]:  # sample first 100 timesteps
    preds = predictions_by_time[t]
    optimal_time = optimize_signal_timing(preds)
    optimized_green_times.append(optimal_time)
    traffic_classes.append(np.mean(preds))  # avg traffic level

optimized_green_times = np.array(optimized_green_times)
traffic_classes = np.array(traffic_classes)

# --------- CALCULATE PERFORMANCE METRICS ---------
def calculate_delay(green_time, traffic_class, num_sensors):
    """
    Estimate average vehicle delay based on signal timing.
    Uses Webster's delay formula (simplified).
    
    Args:
        green_time: green light duration in seconds
        traffic_class: average predicted class (0-2)
        num_sensors: number of sensors/approaches
    
    Returns:
        avg_delay: average delay per vehicle in seconds
    """
    cycle_time = SIGNAL_CYCLE_TIME
    red_time = cycle_time - green_time
    
    # Map class to arrival rate (vehicles per second)
    # 0=low: 0.2 veh/s, 1=med: 0.5 veh/s, 2=high: 0.8 veh/s
    arrival_rate = 0.2 + traffic_class * 0.3
    
    # Service rate (vehicles that can pass during green)
    service_rate = 0.6  # vehicles per second when green
    
    # Simplified delay calculation
    # Uniform delay component
    uniform_delay = 0.5 * cycle_time * (1 - green_time/cycle_time)**2 / (1 - min(arrival_rate/service_rate, 0.95))
    
    # Random delay component (simplified)
    random_delay = (arrival_rate * cycle_time) / (2 * num_sensors)
    
    total_delay = uniform_delay + random_delay
    return max(0, total_delay)

# Calculate delays for baseline and optimized
num_sensors = N
baseline_delays = []
optimized_delays = []

for i, traffic_class in enumerate(traffic_classes):
    baseline_delay = calculate_delay(baseline_green_time, traffic_class, num_sensors)
    optimized_delay = calculate_delay(optimized_green_times[i], traffic_class, num_sensors)
    
    baseline_delays.append(baseline_delay)
    optimized_delays.append(optimized_delay)

baseline_delays = np.array(baseline_delays)
optimized_delays = np.array(optimized_delays)

# --------- RESULTS ---------
print(f"\nOptimization Results (based on {len(traffic_classes)} timesteps):")
print(f"{'='*60}")
print(f"Average Green Time:")
print(f"  Baseline (fixed):     {baseline_green_time:.1f} seconds")
print(f"  Optimized (adaptive): {np.mean(optimized_green_times):.1f} seconds")
print(f"\nAverage Vehicle Delay:")
print(f"  Baseline:   {np.mean(baseline_delays):.2f} seconds")
print(f"  Optimized:  {np.mean(optimized_delays):.2f} seconds")
print(f"  Improvement: {((np.mean(baseline_delays) - np.mean(optimized_delays)) / np.mean(baseline_delays) * 100):.1f}%")
print(f"\nTotal Time Saved:")
print(f"  {np.sum(baseline_delays) - np.sum(optimized_delays):.1f} seconds")
print(f"  ({(np.sum(baseline_delays) - np.sum(optimized_delays)) / 60:.1f} minutes)")

# --------- VISUALIZATION ---------
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Green time adaptation
    ax = axes[0, 0]
    timesteps = np.arange(len(optimized_green_times))
    ax.plot(timesteps, [baseline_green_time]*len(timesteps), 'r--', label='Baseline (Fixed)', linewidth=2)
    ax.plot(timesteps, optimized_green_times, 'g-', label='Optimized (Adaptive)', linewidth=1.5)
    ax.fill_between(timesteps, optimized_green_times, baseline_green_time, 
                     where=(optimized_green_times < baseline_green_time), 
                     color='green', alpha=0.2, label='Time saved')
    ax.fill_between(timesteps, optimized_green_times, baseline_green_time,
                     where=(optimized_green_times > baseline_green_time),
                     color='red', alpha=0.2, label='Extra time for high demand')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Green Light Duration (seconds)')
    ax.set_title('Signal Timing: Baseline vs Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Traffic class vs green time
    ax = axes[0, 1]
    colors = ['green', 'yellow', 'red']
    for cls in [0, 1, 2]:
        mask = np.abs(traffic_classes - cls) < 0.3
        if np.any(mask):
            ax.scatter(traffic_classes[mask], optimized_green_times[mask], 
                      c=colors[cls], label=f'Class {cls} ({"Low" if cls==0 else "Med" if cls==1 else "High"})',
                      alpha=0.6, s=50)
    ax.axhline(baseline_green_time, color='red', linestyle='--', label='Baseline', linewidth=2)
    ax.set_xlabel('Average Traffic Class')
    ax.set_ylabel('Green Light Duration (seconds)')
    ax.set_title('Adaptive Timing Based on Predicted Traffic')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Delay comparison
    ax = axes[1, 0]
    ax.plot(timesteps, baseline_delays, 'r-', label='Baseline Delay', linewidth=1.5, alpha=0.7)
    ax.plot(timesteps, optimized_delays, 'g-', label='Optimized Delay', linewidth=1.5, alpha=0.7)
    ax.fill_between(timesteps, baseline_delays, optimized_delays, 
                     where=(baseline_delays > optimized_delays),
                     color='green', alpha=0.3, label='Delay reduction')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Delay per Vehicle (seconds)')
    ax.set_title('Vehicle Delay: Baseline vs Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative time saved
    ax = axes[1, 1]
    cumulative_saved = np.cumsum(baseline_delays - optimized_delays)
    ax.plot(timesteps, cumulative_saved / 60, 'b-', linewidth=2)
    ax.fill_between(timesteps, 0, cumulative_saved / 60, color='blue', alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Time Saved (minutes)')
    ax.set_title('Total Time Savings Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traffic_optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to 'traffic_optimization_results.png'")
    plt.show()
    
except Exception as e:
    print(f"\nNote: Could not create visualization: {e}")

# --------- SAVE OPTIMIZATION MODEL ---------
print(f"\n{'='*60}")
print("Saving optimization parameters...")

optimization_results = {
    'model': nb,
    'scaler': scaler,
    'feature_names': feat_names,
    'class_to_demand': {0: 0.5, 1: 1.0, 2: 1.5},
    'signal_params': {
        'cycle_time': SIGNAL_CYCLE_TIME,
        'min_green': MIN_GREEN_TIME,
        'max_green': MAX_GREEN_TIME,
        'baseline_green': baseline_green_time
    },
    'performance': {
        'avg_baseline_delay': np.mean(baseline_delays),
        'avg_optimized_delay': np.mean(optimized_delays),
        'improvement_pct': ((np.mean(baseline_delays) - np.mean(optimized_delays)) / np.mean(baseline_delays) * 100)
    }
}

import pickle
with open('traffic_signal_optimizer.pkl', 'wb') as f:
    pickle.dump(optimization_results, f)

print(f"✓ Optimization model saved to 'traffic_signal_optimizer.pkl'")
print(f"\nTo use in production:")
print(f"  1. Load model: model_data = pickle.load(open('traffic_signal_optimizer.pkl', 'rb'))")
print(f"  2. Extract features from live traffic")
print(f"  3. Predict classes: predictions = model_data['model'].predict(features)")
print(f"  4. Optimize timing based on predictions")
print(f"\n{'='*60}")
