"""Step 1 — sanity-check the dataset before any training.

Run with:  uv run python scripts/01_explore_data.py

It downloads HAM10000 (first run only), applies our benign/malignant mapping,
prints the class balance for each split, and saves a grid of sample images to
reports/figures/ so you can see what the model will learn from.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from skin_lesion_xai import config, data


def main() -> None:
    print("Loading HAM10000 (first run downloads ~5 GB and is cached)...")
    ds = data.load_binary_ham10000()

    print("\nSplit sizes and class balance (benign vs malignant):")
    for split in ("train", "validation", "test"):
        counts = data.class_distribution(ds[split])
        total = sum(counts.values())
        print(f"  {split:<11} n={total:>6}  {counts}")

    # Save a small grid of example images with their binary labels.
    sample = ds["train"].shuffle(seed=config.SEED).select(range(8))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, example in zip(axes.ravel(), sample):
        ax.imshow(example["image"])
        ax.set_title(config.ID2LABEL[example["label"]])
        ax.axis("off")
    fig.suptitle("HAM10000 sample lesions (binary labels)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "data_samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved sample grid -> {out}")


if __name__ == "__main__":
    main()
