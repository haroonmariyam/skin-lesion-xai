# Explainable Skin-Lesion Classification: ViT vs ResNet with LIME

A comparative study of two deep-learning architectures for **benign vs.
malignant** skin-lesion classification, using **LIME** to make their
predictions interpretable. Built with HuggingFace, tracked with Weights &
Biases, and packaged reproducibly with UV.

> **Why this matters:** In medical AI, an accurate model is not enough — a
> clinician needs to know *why* a prediction was made. This project measures
> not just *how well* two models classify lesions, but *whether they look at
> the right things* when they do.

---

## ⚠️ Project status & disclosure

> **Work in progress — not yet independently reviewed, tested, or reproduced.**
> The results reported here come from initial training runs and a small,
> qualitative LIME analysis (5 test images). They have **not** been
> independently verified, the experiments have not been repeated across multiple
> seeds, and the code has not undergone formal review. Treat all numbers and
> conclusions as **preliminary**. This is a learning/portfolio project and is
> **not** intended for clinical or diagnostic use.

> **AI assistance disclosure:** This project was built with the assistance of
> **Claude Code** (Anthropic's agentic coding tool), partly as a test case to
> evaluate how such tools perform on a real, end-to-end machine-learning
> workflow (environment setup, training on Kaggle, experiment tracking, and
> explainability). The methodology, decisions, and results were directed and
> reviewed by the author, but readers should be aware that significant portions
> of the scaffolding, code, and documentation were AI-assisted and still warrant
> careful human verification.

---

## Objectives

1. **Build a reproducible pipeline** to fine-tune image classifiers on the
   HAM10000 dermatoscopy dataset for a binary (benign/malignant) task.
2. **Compare two architecture families** — a Vision Transformer (ViT) and a
   convolutional network (ResNet-50) — on the same data and metrics.
3. **Apply LIME explainability** to both models and assess *where* each model
   focuses when making a decision.
4. **Evaluate beyond accuracy**, using metrics appropriate for imbalanced
   medical data (F1, recall, ROC-AUC), and judge explanation quality
   qualitatively (does the model attend to the lesion, or to skin/artifacts?).
5. **Track every experiment** in Weights & Biases for transparent, comparable
   results.

## Research questions

- Does the transformer (ViT) or the CNN (ResNet) achieve better malignant-case
  recall on HAM10000?
- Do the two models rely on the **same image regions**, according to LIME?
- When a model is *wrong*, does LIME reveal that it focused on an irrelevant
  region (e.g. ruler marks, hair, skin) rather than the lesion?

---

## Methodology

| Stage | What happens | Tools |
|-------|--------------|-------|
| **1. Data** | Load HAM10000; collapse 7 diagnoses into benign/malignant; split train/val/test | HuggingFace `datasets` |
| **2. Models** | Load pretrained ViT and ResNet-50; replace head with a 2-class head (transfer learning) | HuggingFace `transformers` |
| **3. Training** | Fine-tune each model; log loss & metrics live | `Trainer` + Weights & Biases |
| **4. Evaluation** | Accuracy, F1, precision, recall, ROC-AUC on the held-out test set; confusion matrices | `scikit-learn` |
| **5. Explainability** | Run LIME on test images for both models; overlay important regions | `lime` |
| **6. Comparison** | Compare metrics *and* explanation maps side by side | report + figures |

### Binary label mapping

The original HAM10000 has 7 diagnoses. They are grouped as:

- **Malignant (1):** melanoma, basal cell carcinoma, actinic keratoses
- **Benign (0):** melanocytic nevi, benign keratosis, dermatofibroma, vascular

(See `src/skin_lesion_xai/config.py`; any unexpected diagnosis raises an error
so nothing is mislabelled silently.)

---

## Project structure

```
skin-lesion-xai/
├── README.md                  ← you are here (objectives + methodology)
├── pyproject.toml             ← dependencies, managed by UV
├── reports/
│   ├── methodology.md         ← detailed write-up + results to fill in
│   └── figures/               ← saved plots (data samples, LIME maps, etc.)
├── scripts/
│   ├── 01_explore_data.py     ← inspect data + class balance
│   ├── 02_train.py            ← fine-tune one model (logs to W&B)
│   └── 03_explain.py          ← run LIME on a trained model
├── notebooks/                 ← Kaggle training notebook lives here
└── src/skin_lesion_xai/       ← the reusable package
    ├── config.py              ← all settings in one place
    ├── data.py                ← load + label + preprocess
    ├── models.py              ← load ViT / ResNet from HuggingFace
    ├── train.py               ← training loop + W&B
    ├── evaluate.py            ← metrics + confusion matrix
    └── explain.py             ← LIME explanations
```

---

## Setup

This project uses [UV](https://docs.astral.sh/uv/) as its package manager.

```bash
# 1. Install dependencies into an isolated environment (.venv)
uv sync

# 2. Add your Weights & Biases key
cp .env.example .env        # then edit .env and paste your key
```

## Running it

```bash
# Explore the data and save a sample image grid
uv run python scripts/01_explore_data.py

# Fine-tune each model (use --subset 0.1 --epochs 1 for a quick test first)
uv run python scripts/02_train.py --model vit
uv run python scripts/02_train.py --model resnet

# Generate LIME explanations for each trained model
uv run python scripts/03_explain.py --model-dir models/vit-ham10000-binary
uv run python scripts/03_explain.py --model-dir models/resnet-ham10000-binary
```

> **Where to train?** Training is GPU-heavy. The recommended workflow is to run
> the training step on **Kaggle** (free GPUs + the HAM10000 dataset is hosted
> there). See [`notebooks/README.md`](notebooks/README.md). All the code here
> is written to run identically on your laptop or on Kaggle.

---

## Results

See [`reports/methodology.md`](reports/methodology.md) — fill in the metrics
table and LIME comparison as you complete each run.

## Tech stack

`Python` · `UV` · `PyTorch` · `HuggingFace Transformers & Datasets` ·
`Weights & Biases` · `LIME` · `scikit-learn`
