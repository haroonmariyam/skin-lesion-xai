"""Load HAM10000, map diagnoses to binary labels, and preprocess images."""

from __future__ import annotations

from datasets import DatasetDict, load_dataset

from . import config


def _dx_to_binary(dx: str) -> int:
    """Map a diagnosis string to 0 (benign) or 1 (malignant)."""
    key = dx.strip().lower()
    malignant = {x.lower() for x in config.MALIGNANT_DX}
    benign = {x.lower() for x in config.BENIGN_DX}
    if key in malignant:
        return 1
    if key in benign:
        return 0
    raise ValueError(
        f"Unknown diagnosis {dx!r}. Add it to MALIGNANT_DX or BENIGN_DX in config.py."
    )


def load_binary_ham10000() -> DatasetDict:
    """Load HAM10000 with an added integer `label` column (0/1)."""
    ds = load_dataset(config.HF_DATASET)

    def add_label(batch):
        batch["label"] = [_dx_to_binary(dx) for dx in batch["dx"]]
        return batch

    return ds.map(add_label, batched=True, desc="Mapping dx -> binary label")


def make_transform(image_processor):
    """Return a transform that turns PIL images into `pixel_values` tensors.

    ViT and ResNet need different input sizes/normalisation, so the transform
    is built from each model's own processor. Applied lazily during training.
    """

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        encoded = image_processor(images, return_tensors="pt")
        batch["pixel_values"] = encoded["pixel_values"]
        return batch

    return transform


def class_distribution(split) -> dict[str, int]:
    """Count benign vs malignant examples in a split."""
    counts = {label: 0 for label in config.ID2LABEL.values()}
    for label_id in split["label"]:
        counts[config.ID2LABEL[label_id]] += 1
    return counts
