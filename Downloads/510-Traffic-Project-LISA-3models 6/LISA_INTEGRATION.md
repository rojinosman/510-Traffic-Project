
# Using the Kaggle LISA Traffic Light Dataset (mbornoe/lisa-traffic-light-dataset) with this repo

This repo originally contained separate experiments:
- Supervised (GaussianNB) on a different dataset
- Reinforcement learning (tabular Q-learning) in a synthetic simulator
- GA optimization in a synthetic simulator

To make **all algorithms use the same dataset**, this repo now includes a shared loader + feature builder for the
**LISA Traffic Light Dataset** and a single evaluation script that trains multiple methods on the **same** features.

## 1) Download / extract the dataset

Recommended (Kaggle API):
- `kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p <some_dir> --unzip`

Then set `DATA_ROOT=<some_dir>/<extracted_dataset_root>`.

## 2) Build a shared feature file

From the repo root:

```bash
python build_lisa_features.py --data "$DATA_ROOT" --out lisa_features.npz --prefer BULB
```

- `--prefer BULB` uses `frameAnnotationsBULB.csv` (tight box around lit lamp) when present; otherwise falls back to BOX.

## 3) Train & compare models (same dataset)

```bash
python run_models_lisa.py --features lisa_features.npz
```

This trains and evaluates (all on the same LISA-derived features):
- Logistic Regression (baseline)
- Gaussian Naive Bayes
- Q-learning contextual bandit (tabular RL on discretized features)
- GA-tuned GaussianNB (GA used to tune var_smoothing)

## Notes
- The feature extractor is intentionally simple (resized pixels + basic color stats/histograms) so classical models can run.
- For a stronger CV baseline, replace `extract_features_rgb` with a small CNN embedding (e.g., ResNet18) and keep the rest the same.
