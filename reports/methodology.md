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
| ViT | 0.984 | 0.958 | 0.960 | **0.957** | **0.998** |
| ResNet-50 | **0.989** | **0.973** | **0.965** | **0.980** | 0.996 |

> Hardware/seed: Kaggle T4 GPU, batch size 32, seed 42.
> **ViT:** 3 epochs, learning rate 5e-5.
> **ResNet-50:** 5 epochs, learning rate 1e-3.

### Key methodological note

ResNet-50 initially **underfit badly** at the ViT learning rate (5e-5): after
3 epochs its training loss had barely moved (0.81 → 0.69) and recall was only
0.17 — essentially the majority-class baseline. Raising the learning rate to
1e-3 and training for 5 epochs let it converge properly (recall 0.31 → 0.96
across epochs). This illustrates that CNNs and Vision Transformers have
**different optimal learning rates**, and that a single shared hyperparameter
setting can unfairly disadvantage one architecture. The comparison above uses
each model's properly-tuned configuration.

### Headline finding

Once both were fairly trained, performance was **very close**. ResNet-50 edged
ahead on recall (0.980 vs 0.957) and F1 (0.973 vs 0.958) — important for a
screening task where missing a malignant case is costly — while ViT had a
marginally higher ROC-AUC (0.998 vs 0.996). The explainability analysis (next
section) examines whether they reach these similar scores by looking at the
**same** image regions.

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
