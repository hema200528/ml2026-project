# Interactions Between Pruning, Quantization, and Batch Size in MLPs

ML 2026 Course Project — a study of how three compression methods (pruning, quantization, batch size) interact when applied to MLPs, compared against classical ML baselines, on medical image datasets.

## Research questions
1. Does the order of pruning and quantization (prune→quantize vs quantize→prune) change final accuracy at the same compression?
2. Does the best batch size shift once a model is compressed?
3. Is there a combined sweet spot that beats any single method alone?

## Datasets
| Dataset | Type | Task | Train / Val / Test |
|---|---|---|---|
| PneumoniaMNIST | Grayscale 28×28 | Binary | 4708 / 524 / 624 |
| BloodMNIST | RGB 28×28 | Multi-class (8) | 11959 / 1712 / 3421 |

Both from MedMNIST (`pip install medmnist`). Pre-cleaned, standard splits, no missing values.

## Methods
| Type | Model / Method |
|---|---|
| Classical | Logistic Regression, Random Forest, RBF SVM |
| Neural | MLP (512→256, ReLU, dropout) |
| Compression | Magnitude pruning (+ fine-tune), post-training quantization, coreset selection, PCA |

## Results (D2)
**Classical vs MLP** (test acc / macro-F1):
| Model | Pneu Acc | Blood Acc |
|---|---|---|
| Logistic Regression | 0.832 | 0.786 |
| Random Forest | 0.853 | 0.838 |
| RBF SVM | 0.857 | 0.853 |
| MLP | 0.858 ± 0.002 | 0.867 ± 0.002 |

**Compression highlights:**
- Pruning: recovers to baseline even at 90% sparsity after fine-tuning.
- Quantization: lossless down to 3-bit, breaks at 2-bit.
- Coreset: 50% of data reaches most of the accuracy.
- PCA: 784→128 dims improves Pneumonia to 0.883.

## Baseline papers
- Han et al. (2016) Deep Compression — ICLR
- Keskar et al. (2017) On Large-Batch Training — ICLR
- Yang et al. (2023) MedMNIST v2 — Scientific Data

## Reproducibility
Fixed seed (42). Standardisation on train split only. MLP uses validation-based early stopping; best model kept by validation accuracy. MLP baseline reported as mean ± std over 3 seeds.
