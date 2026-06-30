# Methodology & Results

This document is the detailed write-up of the project. The structure mirrors a
short research report. Fill in the tables and figures as you complete runs.

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

> Note the imbalance: malignant cases are the minority. This is why we report
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
- Hyperparameters: epochs, batch size, learning rate _(record actual values)_
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
0.17, essentially the majority-class baseline. Raising the learning rate to
1e-3 and training for 5 epochs let it converge properly (recall 0.31 → 0.96
across epochs). This illustrates that CNNs and Vision Transformers have
**different optimal learning rates**, and that a single shared hyperparameter
setting can unfairly disadvantage one architecture. The comparison above uses
each model's properly-tuned configuration.

### Headline finding

Once both were fairly trained, performance was **very close**. ResNet-50 edged
ahead on recall (0.980 vs 0.957) and F1 (0.973 vs 0.958), important for a
screening task where missing a malignant case is costly, while ViT had a
marginally higher ROC-AUC (0.998 vs 0.996). The explainability analysis (next
section) examines whether they reach these similar scores by looking at the
**same** image regions.

Confusion matrices: see `reports/figures/`.

## 6. Explainability analysis (LIME)

LIME was run on the **same 5 test lesions** for both models (1000 perturbations
per image, top-6 positive superpixels shown). Both models classified all 5
images **correctly and identically** (3 benign, 2 malignant), so any
difference is in *reasoning*, not the prediction.

| Observation | ViT | ResNet-50 |
|-------------|-----|-----------|
| Focus on the lesion (malignant cases) | Tight, concentrated **on the lesion** | More **scattered**, often on peripheral skin |
| Focus on benign cases | Diffuse / peripheral skin | Diffuse / peripheral skin |
| Predictions on shared images | 5/5 correct | 5/5 correct (full agreement with ViT) |
| Explanation plausibility (malignant) | Higher: highlights the pigmented lesion | Lower: highlights normal skin at the edges |

**Per-image notes (malignant):** on `lime_*_03`, ViT placed a single tight
region directly on the central pigmented lesion, whereas ResNet's important
regions sat in the image corners (normal skin). The same pattern appears on
`lime_*_00`. On **benign** cases, both models relied on diffuse, peripheral
regions, which is reasonable since "benign" is partly evidenced by the *absence* of
malignant features in the surrounding skin.

LIME figures: see `reports/figures/lime_*.png`.

## 7. Discussion & conclusions

Both architectures reached near-identical, strong predictive performance
(accuracy ≈ 0.98–0.99) and agreed on every shared test image. The
explainability analysis, however, revealed a difference the metrics alone hide:
**on malignant cases, ViT's explanations were consistently more lesion-focused**,
while ResNet, despite its marginally higher recall, more often based its
decision on peripheral skin regions. In a medical screening setting, a model
whose reasoning concentrates on the actual lesion is easier to trust and audit,
even if a competing model scores a fraction higher on a metric.

This is the central lesson of the project: **accuracy and explainability are
distinct axes of model quality.** Two models can be statistically
indistinguishable yet arrive at their answers by attending to very different
evidence, which only becomes visible through a tool like LIME.

## 8. Limitations & future work

- Only 5 images were qualitatively inspected; LIME is stochastic and shows
  approximate, superpixel-level importance, so conclusions are illustrative, not
  statistically robust. A larger, quantified study (e.g. measuring the fraction
  of LIME importance falling inside the lesion mask) would strengthen this.
- The binary collapse of the 7 HAM10000 diagnoses discards finer detail.
- Future work: add a third architecture, complement LIME with Grad-CAM/SHAP,
  and address class imbalance with weighted loss or augmentation.
