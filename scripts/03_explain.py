"""Step 3 — generate LIME explanations for a trained model.

Run AFTER training, e.g.:
  uv run python scripts/03_explain.py --model-dir models/vit-ham10000-binary --n 5

It loads the fine-tuned model, picks a few test images, runs LIME on each, and
saves the highlighted-region figures to reports/figures/. Run it for BOTH
trained models, then compare the figures side by side in your report: do ViT
and ResNet focus on the same part of the lesion?
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skin_lesion_xai import config, data, explain


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIME on a trained model.")
    parser.add_argument(
        "--model-dir", required=True,
        help="Folder of a fine-tuned model, e.g. models/vit-ham10000-binary",
    )
    parser.add_argument("--n", type=int, default=5, help="How many test images.")
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="LIME perturbations per image (higher = sharper but slower).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tag = model_dir.name  # e.g. "vit-ham10000-binary"
    model, processor = explain.load_trained(model_dir)

    print("Loading test images...")
    ds = data.load_binary_ham10000()
    test = ds["test"].shuffle(seed=config.SEED).select(range(args.n))

    for i, example in enumerate(test):
        image = example["image"]
        true_label = config.ID2LABEL[example["label"]]
        print(f"[{i + 1}/{args.n}] explaining (true label: {true_label})...")

        explanation, pred = explain.explain_image(
            model, processor, image, num_samples=args.num_samples
        )
        out = config.FIGURES_DIR / f"lime_{tag}_{i:02d}_true-{true_label}.png"
        explain.save_explanation_figure(
            explanation, pred, title=tag, out_path=out
        )
        print(f"    saved -> {out}")


if __name__ == "__main__":
    main()
