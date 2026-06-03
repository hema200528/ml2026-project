# Interactions Between Pruning, Quantization, and Batch Size in MLPs

ML 2026 Course Project — study of how three compression knobs (pruning, quantization, batch size) interact when applied to MLPs, compared against classical ML baselines, on medical image datasets.

## Research questions

1. Does the order of pruning and quantization (prune→quantize vs quantize→prune) change final accuracy at the same compression?
2. Does the best batch size shift once a model is compressed?
3. Is there a combined sweet spot that beats any single method alone?

## Datasets

| Dataset | Type | Task | Size | Train / Val / Test |
|---|---|---|---|---|
| PneumoniaMNIST | Grayscale (28×28) | Binary (normal / pneumonia) | 5,856 | 4708 / 524 / 624 |
| BloodMNIST | RGB (28×28) | Multi-class (8 cell types) | 17,092 | 11959 / 1712 / 3421 |

Both from MedMNIST, loaded via `pip install medmnist`. Pre-cleaned, standard splits, no missing values.

## Models

| Type | Model | Notes |
|---|---|---|
| Classical | Random Forest | Fast baseline, no scaling needed |
| Classical | RBF SVM | Stronger image baseline |
| Neural | MLP (2 hidden: 512, 256) | ReLU + dropout; the model that gets compressed |

## Current results (D1 baselines)

| Model | Pneu Acc | Pneu F1 | Blood Acc | Blood F1 |
|---|---|---|---|---|
| Random Forest | 0.853 | 0.829 | 0.837 | 0.808 |
| RBF SVM | 0.854 | 0.829 | 0.847 | 0.823 |
| MLP (tuned) | 0.851 | 0.825 | 0.862 | 0.846 |

Note: Pneumonia caps ~85% (val/test come from different sources — known distribution shift). Blood MLP is the clear winner on clean data.

## Baseline papers

| Paper | Venue | Topic |
|---|---|---|
| Han et al., Learning both Weights and Connections (2015) | NeurIPS | Pruning |
| Han et al., Deep Compression (2016) | ICLR | Pruning + quantization |
| Keskar et al., On Large-Batch Training (2017) | ICLR | Batch size |

## Setup

```bash
pip install -r requirements.txt
```

## Reproducibility

Fixed seed (42) for numpy and torch. Standardisation computed on train split only. MLP uses validation-based early stopping; best model kept by validation accuracy.

## Repo structure

```
notebooks/   EDA + experiments
src/         reusable code
configs/     hyperparameter YAMLs
results/     figures and logs
```
