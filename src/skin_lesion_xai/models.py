"""Load the HuggingFace models with a fresh 2-class head (transfer learning)."""

from __future__ import annotations

from transformers import AutoImageProcessor, AutoModelForImageClassification

from . import config


def load_model_and_processor(model_key: str):
    """Return (model, image_processor) for "vit" or "resnet".

    The pretrained 1000-class head is replaced with a 2-class head;
    `ignore_mismatched_sizes=True` allows that swap.
    """
    if model_key not in config.MODELS:
        raise KeyError(
            f"Unknown model {model_key!r}. Choose one of: {list(config.MODELS)}"
        )

    checkpoint = config.MODELS[model_key]
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model, image_processor
