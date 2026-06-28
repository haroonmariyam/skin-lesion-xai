"""Evaluation metrics and plots.

For a medical screening task, raw accuracy is misleading because the classes
are imbalanced (far more benign than malignant). So we also report F1, recall
(how many malignant cases we catch), precision, and ROC-AUC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Convert raw model scores (logits) into probabilities that sum to 1."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(eval_pred):
    """Metric function in the exact shape the HuggingFace Trainer expects.

    `eval_pred` is a tuple (logits, labels). Returns a dict of scalar metrics,
    all of which the Trainer automatically logs to Weights & Biases.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)
    malignant_prob = probs[:, config.LABEL2ID["malignant"]]

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        # ROC-AUC needs both classes present in the batch; guard against errors.
        "roc_auc": roc_auc_score(labels, malignant_prob)
        if len(np.unique(labels)) > 1
        else float("nan"),
    }


def save_confusion_matrix(labels, preds, title: str, out_path: Path) -> Path:
    """Render and save a confusion matrix figure for the report."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[config.ID2LABEL[0], config.ID2LABEL[1]],
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
