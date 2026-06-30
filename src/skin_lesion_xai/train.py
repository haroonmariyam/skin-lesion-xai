"""Fine-tune a model on the binary skin-lesion task, logging to W&B."""

from __future__ import annotations

import torch
from transformers import Trainer, TrainingArguments, set_seed

from . import config, data, evaluate, models


def _collate(batch):
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

    model_key:       "vit" or "resnet"
    subset_fraction: e.g. 0.1 for a quick smoke-test; None for the full dataset.
    """
    set_seed(config.SEED)

    ds = data.load_binary_ham10000()
    model, image_processor = models.load_model_and_processor(model_key)

    if subset_fraction is not None:
        for split in ("train", "validation", "test"):
            n = int(len(ds[split]) * subset_fraction)
            ds[split] = ds[split].shuffle(seed=config.SEED).select(range(n))

    transform = data.make_transform(image_processor)
    ds = ds.with_transform(transform)

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
        remove_unused_columns=False,
        report_to="wandb",
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

    # Evaluate on the held-out test split.
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # Save model + processor for reloading in the LIME step.
    trainer.save_model(str(output_dir))
    image_processor.save_pretrained(str(output_dir))

    return trainer
