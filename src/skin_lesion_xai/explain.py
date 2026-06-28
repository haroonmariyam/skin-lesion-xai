"""LIME explanations for the trained classifiers.

LIME (Local Interpretable Model-agnostic Explanations) answers: "WHICH parts
of *this* image made the model predict malignant?" It works by:
  1. Splitting the image into small regions (superpixels).
  2. Turning regions on/off thousands of times and watching how the model's
     prediction changes.
  3. Fitting a simple, interpretable model to those results to score each
     region's importance.

Because LIME only needs the model's predict function (not its internals), the
*exact same* explainer works for both ViT and ResNet — which is what makes a
fair side-by-side comparison possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from transformers import AutoImageProcessor, AutoModelForImageClassification

from . import config


def load_trained(model_dir: str | Path):
    """Reload a fine-tuned model + its image processor from disk."""
    model_dir = str(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model.eval()  # turn off training-only behaviour (dropout, etc.)
    return model, processor


def make_predict_fn(model, processor):
    """Build the function LIME calls: images (numpy) -> class probabilities.

    LIME passes a batch of RGB images as a numpy array of shape
    (N, H, W, 3) with values in 0-255. We convert each to the tensor the model
    expects, run a forward pass, and return softmax probabilities.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict(images: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil_images = [Image.fromarray(img.astype("uint8")) for img in images]
        inputs = processor(pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    return predict


def explain_image(
    model,
    processor,
    image,
    num_samples: int = 1000,
    num_features: int = 6,
):
    """Run LIME on one PIL image. Returns (explanation, predicted_label_id)."""
    image_np = np.array(image.convert("RGB"))
    predict_fn = make_predict_fn(model, processor)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=config.NUM_LABELS,
        hide_color=0,
        num_samples=num_samples,  # more samples = sharper but slower
    )
    predicted_label = int(explanation.top_labels[0])
    return explanation, predicted_label


def save_explanation_figure(
    explanation,
    predicted_label: int,
    title: str,
    out_path: Path,
    num_features: int = 6,
) -> Path:
    """Overlay LIME's important regions on the image and save the figure."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Regions that pushed the prediction toward the chosen class.
    temp, mask = explanation.get_image_and_mask(
        predicted_label,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(f"{title}\npredicted: {config.ID2LABEL[predicted_label]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
