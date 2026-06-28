"""Central configuration.

Keeping all the "magic values" (model names, label definitions, paths) in one
file means you change a setting in exactly one place, and every script stays
consistent. This is a small habit that makes a project look professional.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the top-level folder (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"  # trained weights are saved here (gitignored)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# The HAM10000 dermatoscopy dataset, hosted on the HuggingFace Hub.
HF_DATASET = "marmal88/skin_cancer"

# The original dataset has 7 fine-grained diagnoses in its "dx" column. We
# collapse them into a BINARY problem: benign (0) vs malignant (1).
#
# Clinical grouping used here:
#   malignant = melanoma, basal cell carcinoma, actinic keratoses (pre-cancer)
#   benign    = melanocytic nevi, benign keratosis, dermatofibroma, vascular
#
# The dataset stores diagnoses as descriptive strings, but to be safe we map
# BOTH the short codes (mel, bcc, ...) and the long names. Anything not listed
# here raises an error in data.py, so a surprise label can never be silently
# mislabelled.
MALIGNANT_DX = {
    "mel", "melanoma",
    "bcc", "basal_cell_carcinoma",
    "akiec", "actinic_keratoses",
}
BENIGN_DX = {
    "nv", "melanocytic_nevi", "melanocytic_Nevi",
    "bkl", "benign_keratosis-like_lesions", "benign_keratosis",
    "df", "dermatofibroma",
    "vasc", "vascular_lesions",
}

# Binary label scheme used everywhere downstream.
ID2LABEL = {0: "benign", 1: "malignant"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# ---------------------------------------------------------------------------
# Models (pulled from the HuggingFace Hub)
# ---------------------------------------------------------------------------
# Two DIFFERENT architecture families, so the comparison is meaningful:
#   - ViT  : a Vision Transformer (attention-based)
#   - ResNet: a classic convolutional neural network
MODELS = {
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
