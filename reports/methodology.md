# Methodology & Results

This document is the detailed write-up of the project. The structure mirrors a
short research report — fill in the tables and figures as you complete runs.

## 1. Problem statement

Binary classification of dermatoscopic skin-lesion images as **benign** or
**malignant**, with a focus on *explainability*: understanding which image
regions drive each model's decision.

## 2. Dataset

- **Source:** HAM10000 (`marmal88/skin_cancer` on the HuggingFace Hub).
- **Size:** ~13,400 images across train / validation / test splits.
- **Label engineering:** the 7 original diagnoses are collapsed to 2 classes
  (see mapping in `config.py`).
- **Class balance:** _(fill in from `scripts/01_explore_data.py` output)_

| Split | n | benign | malignant |
|-------|---|--------|-----------|
| train | | | |
| validation | | | |
| test | | | |

> Note the imbalance — malignant cases are the minority. This is why we report
> recall and F1, not just accuracy.

## 3. Models

| Key | Architecture | HuggingFace checkpoint | Family |
|-----|--------------|------------------------|--------|
| `vit` | Vision Transformer | `google/vit-base-patch16-224` | attention |
| `resnet` | ResNet-50 | `microsoft/resnet-50` | convolutional |

Both are pretrained on ImageNet, then fine-tuned with a new 2-class head
(transfer learning).

## 4. Training setup

- Optimiser / schedule: HuggingFace `Trainer` defaults (AdamW).
- Hyperparameters: epochs, batch size, learning rate — _(record actual values)_
- Hardware: _(e.g. Kaggle T4 GPU)_
- Experiment tracking: Weights & Biases project `skin-lesion-xai`.
- Random seed: 42 (set everywhere for reproducibility).

## 5. Quantitative results

_(Fill in from the test-set evaluation printed at the end of training and from
W&B.)_

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|----------|----|-----------|--------|---------|
| ViT | 0.984 | 0.958 | 0.960 | 0.957 | 0.998 |
| ResNet-50 | _(run next)_ | | | | |

> Training: 3 epochs, batch size 32, lr 5e-5, seed 42, Kaggle T4 GPU.
> ViT validation recall climbed 0.63 → 0.87 → 0.96 across epochs (see W&B).

Confusion matrices: see `reports/figures/`.

## 6. Explainability analysis (LIME)

For each model, LIME was run on the same test images. Compare the highlighted
regions:

- Do both models focus on the **lesion** itself, or on surrounding skin /
  artifacts (hair, ruler marks, ink)?
- On **correct** predictions, are the explanations clinically plausible?
- On **incorrect** predictions, does LIME reveal a spurious focus?

| Observation | ViT | ResNet-50 |
|-------------|-----|-----------|
| Focuses on lesion centre | | |
| Sensitive to artifacts | | |
| Agreement on shared images | | |

LIME figures: see `reports/figures/lime_*.png`.

## 7. Discussion & conclusions

_(2–3 paragraphs: which model you'd trust more and why — combine the metrics
with the explanation quality. A model that is slightly less accurate but
consistently looks at the lesion may be preferable in a medical setting.)_

## 8. Limitations & future work

- Binary collapse loses fine-grained diagnostic information.
- LIME explanations are approximate and can vary between runs.
- Possible extensions: add a third model, try Grad-CAM or SHAP alongside LIME,
  or address class imbalance with weighted loss / augmentation.
