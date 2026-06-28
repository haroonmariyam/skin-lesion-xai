"""Fine-tuning a model on the binary skin-lesion task, tracked with W&B.

This uses the HuggingFace `Trainer`, which handles the training loop, GPU
placement, evaluation, and checkpointing for us. Every metric is streamed live
to Weights & Biases so you get charts you can screenshot for your report.
"""

from __future__ import annotations

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import config, data, evaluate, models


def _collate(batch):
    """Stack a list of examples into one batch of tensors for the model."""
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def train(
    model_key: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    subset_fraction: float | None = None,
    run_name: str | None = None,
):
    """Fine-tune one model and return the trained Trainer.

    Parameters
    ----------
    model_key        : "vit" or "resnet"
    epochs           : passes over the training data
    batch_size       : images per step (lower this if you run out of memory)
    learning_rate    : how big each weight-update step is
    subset_fraction  : e.g. 0.1 trains on 10% of data for a fast smoke-test;
                       use None for the full dataset.
    run_name         : label shown in the Weights & Biases dashboard
    """
    set_seed(config.SEED)

    # --- data -------------------------------------------------------------
    ds = data.load_binary_ham10000()
    model, image_processor = models.load_model_and_processor(model_key)

    if subset_fraction is not None:
        for split in ("train", "validation", "test"):
            n = int(len(ds[split]) * subset_fraction)
            ds[split] = ds[split].shuffle(seed=config.SEED).select(range(n))

    transform = data.make_transform(image_processor)
    ds = ds.with_transform(transform)

    # --- training configuration ------------------------------------------
    run_name = run_name or f"{model_key}-ham10000-binary"
    output_dir = config.MODELS_DIR / run_name

    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        remove_unused_columns=False,  # we need the raw "image"/"label" columns
        report_to="wandb",            # <-- sends everything to Weights & Biases
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=_collate,
        compute_metrics=evaluate.compute_metrics,
    )

    trainer.train()

    # Final evaluation on the held-out TEST split (data never seen in training).
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # Save the fine-tuned model + processor so we can reload it for LIME later.
    trainer.save_model(str(output_dir))
    image_processor.save_pretrained(str(output_dir))

    return trainer
