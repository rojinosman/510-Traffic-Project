# pip install numpy pandas scikit-learn h5py tables
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --------- CONFIG ---------
ROOT = Path("data/ca/2019")   # where LargeST's processed 2019 data live after running their script
SENSOR_SAMPLE = 200           # to keep it light: use a subset of sensors
HORIZON_STEPS = 6             # predict 30 minutes ahead if data are 5-min samples (6 * 5 = 30)
BINS = 3                      # e.g., 3 classes: Low/Med/High
LAGS = [1,2,3,6,12]           # 5min, 10min, 15min, 30min, 60min history
ROLL_WINDOWS = [6,12]         # 30min/60min rolling mean/stdev
TRAIN_FRAC, VAL_FRAC = 0.75, 0.10  # remaining is test

# --------- LOAD (processed) ---------
# LargeST repo creates processed files under data/ca/2019. File names can vary; we’ll look for arrays.
# We'll assume a common shape: (T, N) flow matrix where rows are timesteps and columns are sensors.
# Adjust the loader below to match the exact filenames your processing step emitted.
def try_load_processed(root: Path):
    # 1) Try NPZ with keys 'flow' or 'x'/'y' (common patterns)
    for p in [root/"flow_2019.npz", root/"flow.npz", root/"processed.npz"]:
        if p.exists():
            z = np.load(p)
            if "flow" in z: return z["flow"]  # (T, N)
            if "x" in z and "y" in z:
                # Some pipelines create supervised tensors; we can reconstruct a flow matrix from x[...,0]
                X = z["x"]  # (samples, seq_len, N) or similar; adapt if needed
                # fall back: return None to force HDF5 path
    # 2) Try NPY
    for p in [root/"flow_2019.npy", root/"flow.npy"]:
        if p.exists():
            return np.load(p)
    return None

flow = try_load_processed(ROOT)

if flow is None:
    # Fallback: read a raw .h5 file if you didn’t run their generator; pick the 2019 file.
    # We'll search for any 2019 h5 under data/ca and read it with pandas.read_hdf
    import glob
    h5_candidates = glob.glob("data/ca/*2019*.h5") + glob.glob("data/ca/*_2019.h5")
    if not h5_candidates:
        raise FileNotFoundError("Couldn't find processed arrays or a 2019 .h5. "
                                "Run the repo's generate_data_for_training.py or place a 2019 .h5 under data/ca.")
    # LargeST stores a (time x sensors) table; common key is 'df' or something similar.
    # We'll attempt typical keys; adjust if needed once you inspect the file.
    import pandas as pd
    h5_path = h5_candidates[0]
    for key in ["df", "data", "flow", "table", "/"]:
        try:
            df = pd.read_hdf(h5_path, key=key)
            if isinstance(df, pd.DataFrame) and df.shape[0] > 1000:
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not read a valid DataFrame from {h5_path}; inspect available keys.")
    # Expect index as timestamps and columns as sensor IDs
    df = df.sort_index()
    flow = df.to_numpy()  # (T, N)

T, N = flow.shape
print(f"Flow matrix shape: T={T}, N={N}")

# --------- (Optional) subsample sensors for speed ----------
rng = np.random.default_rng(42)
if N > SENSOR_SAMPLE:
    cols = np.sort(rng.choice(N, size=SENSOR_SAMPLE, replace=False))
    flow = flow[:, cols]
    N = flow.shape[1]

# --------- feature engineering (per-timestep across sensors) ----------
# We will build a supervised dataset where each row corresponds to (time t, sensor i).
# Features: lagged values and rolling stats; target: class of flow at t+HORIZON_STEPS.
def build_supervised(flow, horizon, lags, roll_windows):
    T, N = flow.shape
    # rolling statistics per sensor on the time axis
    df = pd.DataFrame(flow)
    feats = []
    feat_names = []
    # lag features
    for L in lags:
        lagged = df.shift(L).to_numpy()
        feats.append(lagged)
        feat_names.append(f"lag{L}")
    # rolling mean/std
    for w in roll_windows:
        rmean = df.rolling(w).mean().to_numpy()
        rstd  = df.rolling(w).std().to_numpy()
        feats.append(rmean); feat_names.append(f"rmean{w}")
        feats.append(rstd);  feat_names.append(f"rstd{w}")

    X_all = np.stack(feats, axis=-1)  # shape (T, N, F)
    # target: flow at t+horizon
    y_reg = df.shift(-horizon).to_numpy()  # (T, N)

    # drop rows with NaNs introduced by shift/rolling
    valid_t = ~np.isnan(X_all).any(axis=(1,2)) & ~np.isnan(y_reg).any(axis=1)
    X_all = X_all[valid_t]
    y_reg = y_reg[valid_t]
    return X_all, y_reg, valid_t, feat_names

X_all, y_reg, mask, feat_names = build_supervised(flow, HORIZON_STEPS, LAGS, ROLL_WINDOWS)
T2 = X_all.shape[0]
F  = X_all.shape[2]
print(f"Supervised tensor: (T2={T2}, N={N}, F={F})")

# flatten (time, sensor) to samples
X = X_all.reshape(T2 * N, F)
y_cont = y_reg.reshape(T2 * N)

# bin continuous flow to classes (quantile bins per whole dataset)
y = pd.qcut(y_cont, q=BINS, labels=False, duplicates='drop')  # 0..BINS-1

# remove any rows falling into dropped bins
valid_rows = ~pd.isna(y)
X = X[valid_rows]
y = y[valid_rows].astype(int)

# Optional scaling (GaussianNB doesn't need it, but can help numerics)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# --------- time-aware split ----------
# Map flattened rows back to time for a chronological split.
# Each block of N rows corresponds to one timestep in the masked timeline.
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

# --------- train Gaussian Naive Bayes ----------
nb = GaussianNB()
nb.fit(Xtr, ytr)

# --------- evaluate ----------
def eval_split(name, Xs, ys):
    yhat = nb.predict(Xs)
    print(f"\n{name} F1(macro): {f1_score(ys, yhat, average='macro'):.3f}")
    print(classification_report(ys, yhat, digits=3))
    print("Confusion matrix:\n", confusion_matrix(ys, yhat))

eval_split("Validation", Xva, yva)
eval_split("Test", Xte, yte)
