"""Project configuration: paths, dataset, label scheme, and model names."""

from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"  # gitignored

# Dataset (HAM10000 on the HuggingFace Hub)
HF_DATASET = "marmal88/skin_cancer"

# Collapse the 7 HAM10000 diagnoses into a binary label.
#   malignant: melanoma, basal cell carcinoma, actinic keratoses
#   benign:    melanocytic nevi, benign keratosis, dermatofibroma, vascular
# Both short codes and long names are listed; anything else raises in data.py.
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

ID2LABEL = {0: "benign", 1: "malignant"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# Models to compare: a transformer and a CNN.
MODELS = {
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
}

SEED = 42
