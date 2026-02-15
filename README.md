# MNIST Neural Network from Scratch

Classify MNIST digits (0–9) using a neural network implemented **from scratch** in NumPy (no Keras or PyTorch for the core model).

## Contents

- **`mnist_nn_from_scratch.ipynb`** — Full pipeline: data load (`fetch_openml("mnist_784", version=1)`), preprocessing, EDA, model (Dense + ReLU + softmax, Adam), training with early stopping, evaluation, and report.
- **`figures/`** — EDA plots, training curves, confusion matrix, per-class metrics, sample predictions.
- **`Neural_network_report.pdf`** — Project report.
- **`requirements.txt`** — `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Setup and run

```bash
pip install -r requirements.txt
```

Open and run all cells in `mnist_nn_from_scratch.ipynb`. Training takes ~10 minutes on CPU; test accuracy is ~98.4%.

## Results

- **Test accuracy:** ~98.4%
- **Architecture:** 784 → 512 → 256 → 128 → 10 (ReLU, dropout 0.3, Adam, L2)
