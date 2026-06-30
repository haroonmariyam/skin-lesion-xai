"""Fine-tune one model, logging to Weights & Biases.

    uv run python scripts/02_train.py --model vit --subset 0.1 --epochs 1  # quick test
    uv run python scripts/02_train.py --model vit                          # full run
    uv run python scripts/02_train.py --model resnet
"""

from __future__ import annotations

import argparse

from skin_lesion_xai import config, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a skin-lesion model.")
    parser.add_argument(
        "--model", required=True, choices=list(config.MODELS),
        help="Which architecture to train (vit or resnet).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--subset", type=float, default=None,
        help="Fraction of data to use, e.g. 0.1 for a fast test. Omit for full.",
    )
    args = parser.parse_args()

    train.train(
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_fraction=args.subset,
    )


if __name__ == "__main__":
    main()
