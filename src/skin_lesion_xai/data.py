"""Data loading and preprocessing for the binary skin-lesion task.

Pipeline:
  1. Download HAM10000 from the HuggingFace Hub (cached after first run).
  2. Map the 7 diagnoses -> a single binary label (0 benign / 1 malignant).
  3. Provide a transform that turns PIL images into model-ready tensors.

The transform is model-specific (ViT and ResNet expect different input sizes
and normalisation), so we build it from each model's own image processor.
"""

from __future__ import annotations

from datasets import DatasetDict, load_dataset

from . import config


def _dx_to_binary(dx: str) -> int:
    """Map one diagnosis string to 0 (benign) or 1 (malignant).

    Raises a clear error on any unexpected value so mislabelling is impossible.
    """
    key = dx.strip().lower()
    # config sets may contain mixed-case names; compare case-insensitively.
    malignant = {x.lower() for x in config.MALIGNANT_DX}
    benign = {x.lower() for x in config.BENIGN_DX}
    if key in malignant:
        return 1
    if key in benign:
        return 0
    raise ValueError(
        f"Unknown diagnosis value {dx!r}. Add it to MALIGNANT_DX or BENIGN_DX "
        "in config.py so it is classified explicitly."
    )


def load_binary_ham10000() -> DatasetDict:
    """Load HAM10000 with an added integer `label` column (0/1).

    Returns a DatasetDict with `train`, `validation`, and `test` splits.
    Images stay as PIL objects here; tensor conversion happens in the
    transform (see `make_transform`) so we only pay that cost during training.
    """
    ds = load_dataset(config.HF_DATASET)

    def add_label(batch):
        batch["label"] = [_dx_to_binary(dx) for dx in batch["dx"]]
        return batch

    ds = ds.map(add_label, batched=True, desc="Mapping dx -> binary label")
    return ds


def make_transform(image_processor):
    """Build a `set_transform` function for a given HuggingFace image processor.

    The returned function is applied lazily, batch by batch, during training.
    It converts each PIL image to a normalised pixel tensor named
    `pixel_values` — the exact key HuggingFace image models expect.
    """

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        encoded = image_processor(images, return_tensors="pt")
        batch["pixel_values"] = encoded["pixel_values"]
        return batch

    return transform


def class_distribution(split) -> dict[str, int]:
    """Count benign vs malignant examples in a split (handy for the report)."""
    counts = {label: 0 for label in config.ID2LABEL.values()}
    for label_id in split["label"]:
        counts[config.ID2LABEL[label_id]] += 1
    return counts
