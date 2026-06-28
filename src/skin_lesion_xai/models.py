"""Loading the HuggingFace models we compare.

Both models are pretrained on ImageNet (1000 classes). We swap their final
classification head for a fresh 2-class head (benign / malignant). This is
called *transfer learning*: keep the general visual features, retrain the
decision layer for our task.
"""

from __future__ import annotations

from transformers import AutoImageProcessor, AutoModelForImageClassification

from . import config


def load_model_and_processor(model_key: str):
    """Return (model, image_processor) for a key in config.MODELS.

    `model_key` is "vit" or "resnet".

    `ignore_mismatched_sizes=True` lets us replace the original 1000-class head
    with our 2-class head without an error.
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
