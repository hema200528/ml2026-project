# Interactions Between Pruning, Quantization, and Batch Size in MLPs

ML 2026 Course Project : a study of how three compression methods (pruning, quantization, batch size) interact when applied to MLPs, compared against classical ML baselines, on medical image datasets. The main finding is that combining aggressive pruning with low-bit quantization causes standard post-training quantization to collapse, and that **quantization-aware training (QAT) rescues it**, recovering up to +0.21 accuracy and reducing variance.

## Research questions

1. Does the order of pruning and quantization (prune→quantize vs quantize→prune) change final accuracy at the same compression?
2. Does the best batch size shift once a model is compressed?
3. Can a combined pruning + quantization strategy be made to work where naive compression fails?

## Datasets

| Dataset | Type | Task | Train / Val / Test |
|---|---|---|---|
| PneumoniaMNIST | Grayscale 28×28 | Binary | 4708 / 524 / 624 |
| BloodMNIST | RGB 28×28 | Multi-class (8) | 11959 / 1712 / 3421 |

Both from MedMNIST (`pip install medmnist`). Pre-cleaned, standard splits, no missing values. One easy task (Pneumonia) and one harder task (Blood) were chosen deliberately to test whether findings depend on difficulty.

## Methods

| Type | Model / Method |
|---|---|
| Classical | Logistic Regression, Random Forest, RBF SVM |
| Neural | MLP (512→256, ReLU, dropout, Adam) |
| Compression | Magnitude pruning (+ fine-tune), post-training quantization (PTQ), coreset selection, PCA |
| Improvement | Quantization-Aware Training (QAT) — custom FakeQuant autograd function with a straight-through estimator |

## Results

### Classical vs MLP (test acc / macro-F1)

| Model | Pneu Acc | Blood Acc |
|---|---|---|
| Logistic Regression | 0.832 | 0.786 |
| Random Forest | 0.853 | 0.838 |
| RBF SVM | 0.857 | 0.853 |
| MLP | 0.858 ± 0.002 | 0.867 ± 0.002 |

### Single-method compression highlights

- **Pruning:** recovers to baseline even at 90% sparsity after fine-tuning (network is ~90% redundant).
- **Quantization:** lossless down to 3-bit, breaks at 2-bit on the harder task.
- **Coreset:** 50% of the data reaches most of the accuracy.
- **PCA:** 784→128 dims improves Pneumonia to 0.883 by removing noise.

### Finding 1 — Order of pruning and quantization

Prune→quantize is consistently better on the simpler task, by up to +0.19 at aggressive settings; the advantage grows with compression severity. On the harder task at sensible compression levels the two orders are equivalent. Comparison uses a fair protocol: same compression level and fine-tuning for both orders.

### Finding 2 — Batch size and compression

The best batch size depends on the dataset, not the compression (Blood prefers 64, Pneumonia prefers 256), matching Keskar et al. (2017). It does not shift after compression. Compressed models also slightly outperform uncompressed ones, consistent with a regularization effect.

### Finding 3 (headline) — QAT rescues combined compression

Combining pruning with low-bit PTQ collapses on the harder task. QAT recovers it in every configuration, with the largest gains where PTQ fails most, and with much lower variance (mean ± std over 3 seeds):

| Dataset | Prune | Bits | PTQ | QAT | Gain |
|---|---|---|---|---|---|
| Pneumonia | 70% | 3 | 0.847 ± 0.008 | 0.867 ± 0.007 | +0.020 |
| Pneumonia | 90% | 2 | 0.810 ± 0.026 | 0.878 ± 0.011 | +0.068 |
| Blood | 70% | 3 | 0.790 ± 0.020 | 0.859 ± 0.005 | +0.069 |
| Blood | 90% | 3 | 0.764 ± 0.008 | 0.861 ± 0.004 | +0.097 |
| Blood | 70% | 2 | 0.602 ± 0.083 | 0.809 ± 0.038 | +0.207 |
| Blood | 90% | 2 | 0.566 ± 0.029 | 0.736 ± 0.026 | +0.170 |

PTQ's standard deviation reaches 0.083 in the failure cases; QAT stays at 0.038 or lower. QAT is therefore both more accurate and more reproducible.

## Baseline papers

- LeCun et al. (1990) Optimal Brain Damage — NeurIPS
- Han et al. (2015) Learning both Weights and Connections — NeurIPS
- Han et al. (2016) Deep Compression — ICLR
- Frankle & Carbin (2019) The Lottery Ticket Hypothesis — ICLR
- Keskar et al. (2017) On Large-Batch Training — ICLR
- Smith et al. (2018) Don't Decay the Learning Rate, Increase the Batch Size — ICLR
- Yang et al. (2023) MedMNIST v2 — Scientific Data

## Reproducibility

Fixed seed (42). Standardisation on the train split only. MLP uses validation-based early stopping, keeping the best model by validation accuracy. The MLP baseline and the QAT improvement are reported as mean ± std over 3 seeds (42, 43, 44); other sweeps are single-run.

## Repository structure

- `D1/` — EDA, classical + MLP baselines, intro video
- `D2/` — all compression methods + preliminary report
- `D3/` — comparison with studies, QAT improvement, presentation
- `D4/` — final report (PDF) and final presentation deck
- `fig/` — all figures (pruning, quantization, coreset, PCA, order, batch, QAT)

## Limitations

MLPs discard spatial structure, so absolute accuracies sit below a CNN's; the focus is compression behaviour, not peak accuracy. Two datasets; most compression sweeps are single-run while the baseline and QAT carry error bars. Combined compression is unstable at extreme settings (2-bit).
