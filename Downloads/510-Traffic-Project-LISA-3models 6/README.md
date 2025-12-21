# LISA Traffic Light Dataset — Same Dataset for Your 3 Original Algorithms

This repo makes **all three algorithms from your original traffic project** run on the **same dataset**,
and adds a **4th model**: a simple feed-forward **neural network**.

- **Supervised:** Gaussian Naive Bayes  
- **Reinforcement:** Tabular **Q-learning** (implemented as a contextual bandit classifier)  
- **GA (optimization):** Genetic Algorithm tuning a model hyperparameter  
- **Neural network:** Multilayer Perceptron (scikit-learn `MLPClassifier`)  

The shared dataset is the **Kaggle LISA Traffic Light Dataset** (`mbornoe/lisa-traffic-light-dataset`).
The shared ML task is **traffic-light state/tag classification** from annotated image crops.

Why classification? LISA is a **computer-vision dataset (images + bounding boxes + state labels)**, not an
intersection flow/signal-timing dataset. The clean way to reuse your 3 algorithms on the same data is to
make them solve the same *label prediction* task.

## Files you actually need

- `common/lisa_loader.py` — loads LISA annotation CSVs (BULB/BOX), resolves image paths, crops boxes, extracts features
- `build_lisa_features.py` — builds `lisa_features.npz` from the downloaded dataset
- `Supervised/naive_bayes_lisa.py` — GaussianNB on the LISA features
- `Reinforced/q_learning_lisa.py` — Q-learning (contextual bandit) on the same LISA features
- `Unsupervised/ga_optimizer_lisa.py` — GA tunes GaussianNB(var_smoothing) using the same LISA features
- `NeuralNetwork/mlp_lisa.py` — MLP neural network on the same LISA features
- `compare_all.py` — runs all four and prints one comparison summary
- `requirements.txt`

## Setup

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download the dataset
Download and unzip the Kaggle dataset somewhere on your machine:
`https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset/data`

### 3) Build one shared feature file
```bash
python build_lisa_features.py --data "/path/to/unzipped/lisa-dataset" --out lisa_features.npz --prefer BULB
```

Helpful speed options for quick testing:
- `--max-samples 5000`
- `--size 16` (default)
- `--hist-bins 8` (default)

### 4) Run all models on the same split + metric
```bash
python compare_all.py --features lisa_features.npz
```

Or run each individually:
```bash
python Supervised/naive_bayes_lisa.py --features lisa_features.npz
python Reinforced/q_learning_lisa.py --features lisa_features.npz
python Unsupervised/ga_optimizer_lisa.py --features lisa_features.npz
python NeuralNetwork/mlp_lisa.py --features lisa_features.npz
python gui_demo.py --features "$HOME/Downloads/510-Traffic-Project-LISA-3models/510-Traffic-Project-main/lisa_features.npz"

```

The comparison metric is **macro-F1 on the test split**.
