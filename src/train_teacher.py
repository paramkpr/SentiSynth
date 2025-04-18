import argparse
import logging
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Set OMP_NUM_THREADS to 1 to avoid potential CPU over-subscription
os.environ['OMP_NUM_THREADS'] = '1'
logger.info(f"Setting OMP_NUM_THREADS=1")


def load_config(config_path: str) -> Dict:
    """Loads configuration from a YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {config}")
    return config


def load_datasets(dataset_path: str) -> DatasetDict:
    """Loads datasets from disk."""
    logger.info(f"Loading datasets from {dataset_path}")
    datasets = load_from_disk(dataset_path)
    logger.info(f"Datasets loaded: {datasets}")
    # Ensure standard column names
    if "sentence" in datasets["train"].column_names:
        if "text" not in datasets["train"].column_names:
            datasets = datasets.rename_column("sentence", "text")
    if "label" in datasets["train"].column_names:
        if "labels" not in datasets["train"].column_names:
            datasets = datasets.rename_column("label", "labels")
    # Make sure 'labels' column exists
    if "labels" not in datasets["train"].column_names:
        raise ValueError(
            "Dataset must contain a 'labels' column (or 'label' to be renamed)."
        )
    return datasets


def tokenize_datasets(
    datasets: DatasetDict, tokenizer: AutoTokenizer, max_len: int, config: Dict
) -> DatasetDict:
    """Tokenizes the text data in the datasets."""
    logger.info("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=False, max_length=max_len
        )

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Select only necessary columns
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    tokenized_datasets = tokenized_datasets.remove_columns(
        [
            col
            for col in tokenized_datasets[config["train_split"]].column_names
            if col not in columns_to_keep
        ]
    )
    logger.info(f"Tokenization complete. Final columns: {tokenized_datasets[config['train_split']].column_names}")
    return tokenized_datasets


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Computes precision, recall, F1, and accuracy."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary" # Using binary as SST-2 is binary classification
    )
    acc = (preds == labels).mean()
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main(config_path: str):
    """Main training function."""
    config = load_config(config_path)

    # --- Setup W&B ---
    run_name = f"train_teacher_{int(time.time())}"
    if config.get("report_to") == "wandb":
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_PROJECT"] = config["project_name"]
        logger.info(f"Logging to W&B project: {config['project_name']}")
    else:
        # Disable W&B if not specified
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("W&B reporting disabled.")


    # --- Load Model and Tokenizer ---
    logger.info(f"Loading model: {config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2 # Assuming binary classification for SST-2
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    # --- Load and Prepare Data ---
    raw_datasets = load_datasets(config["dataset_path"])
    tokenized_datasets = tokenize_datasets(
        raw_datasets, tokenizer, config["max_len"], config
    )

    train_dataset = tokenized_datasets[config["train_split"]]
    eval_dataset = tokenized_datasets[config["eval_split"]]
    sanity_dataset = tokenized_datasets[config["sanity_split"]]

    ## TODO: COMMENT FOR REAL RUN::
    train_dataset = train_dataset.shuffle(seed=42).select(range(256))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(128))
    # sanity_dataset = sanity_dataset.shuffle(seed=42).select(range(256))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Detect device (CUDA, MPS, or CPU) ----------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    logger.info(f"Using device: {device}")

    # --- Training Arguments ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"] and torch.cuda.is_available(),
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=config["save_steps"],
        save_total_limit=2, # Saves the best and the latest checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # Use eval_split F1 for best model
        greater_is_better=True,
        report_to=[config["report_to"]] if config.get("report_to") else [],
        run_name=run_name,
        label_names=["labels"], # Specify label column name
        remove_unused_columns=False, # Keep all columns tokenized earlier
        ddp_find_unused_parameters=False,
    )
    logger.info(f"FP16 enabled: {training_args.fp16}")


    # --- Callbacks ---
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,       # Stop if metric doesn't improve for 3 evaluations
        early_stopping_threshold=0.0,    # Minimum improvement considered
    )

    # --- Trainer Initialization ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # --- Save Final Model & Metrics ---
    logger.info("Saving best model and tokenizer...")
    trainer.save_model(config["output_dir"]) # Saves the best model due to load_best_model_at_end=True
    trainer.save_state() # Saves trainer state, including metrics

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate on sanity set
    sanity_metrics = trainer.evaluate(eval_dataset=sanity_dataset, metric_key_prefix="final_sanity")
    trainer.log_metrics("final_sanity", sanity_metrics)
    trainer.save_metrics("final_sanity", sanity_metrics)

    logger.info(f"Best model saved to {config['output_dir']}")
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a teacher model on SST-2.")
    parser.add_argument(
        "config_path", type=str, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config_path) 