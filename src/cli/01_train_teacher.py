import typer
import os
import yaml
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from transformers import DataCollatorWithPadding, IntervalStrategy, TrainingArguments, Trainer

from src.models import build_teacher
from src.data import ClassificationDataModule


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def compute_metrics(p):
    """Computes metrics for HF Trainer."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary') # Assuming binary
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


@app.command()
def main(config_path: Path = type.Argument(..., help="Path to YAML config")):
    cfg = yaml.safe_load(config_path.read_text())

    # --- SETUP W&B ---
    run_name = cfg['training'].get("run_name", f"teacher_train_{Path(cfg['training']['output_dir']).name}")
    report_to = cfg['training'].get("report_to", "none") # Default to no reporting
    if report_to == "wandb":
        project_name = cfg['training'].get("wandb_project", "senti_synth_teacher")
        os.environ.pop("WANDB_DISABLED", None) # Ensure it's enabled if requested
        os.environ["WANDB_PROJECT"] = project_name
        logger.info(f"Reporting to W&B project: {project_name}")
    else:
        os.environ["WANDB_DISABLED"] = "true" # Explicitly disable
        logger.info("W&B reporting disabled.")

    # --- BUILD MODEL ---
    model, tokenizer = build_teacher(cfg['model'])
    
    # --- SETUP DATA ---
    data_module = ClassificationDataModule(cfg['data'], tokenizer)
    data_module.setup()
    train_dataset = data_module.get_train_dataset()
    eval_dataset = data_module.get_eval_dataset()

    # --- SETUP TRAINER ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args_dict = {
        "output_dir": cfg['training']['output_dir'],
        "overwrite_output_dir": cfg['training'].get("overwrite_output_dir", True),
        "do_train": True,
        "do_eval": eval_dataset is not None, # Only do eval if eval_dataset exists
        "per_device_train_batch_size": cfg['training'].get("per_device_train_batch_size", 8),
        "per_device_eval_batch_size": cfg['training'].get("per_device_eval_batch_size", 16),
        "gradient_accumulation_steps": cfg['training'].get("gradient_accumulation_steps", 1),
        "num_train_epochs": cfg['training'].get("num_train_epochs", 3),
        "learning_rate": cfg['training'].get("learning_rate", 5e-5),
        "warmup_ratio": cfg['training'].get("warmup_ratio", 0.1),
        "fp16": cfg['training'].get("fp16", torch.cuda.is_available()), # Enable FP16 if available by default
        "logging_dir": cfg['training'].get("logging_dir", f"{cfg['training']['output_dir']}/logs"),
        "logging_steps": cfg['training'].get("logging_steps", 100),
        "eval_strategy": IntervalStrategy.STEPS if eval_dataset is not None else IntervalStrategy.NO,
        "eval_steps": cfg['training'].get("eval_steps", 500),
        "save_strategy": IntervalStrategy.STEPS,
        "save_steps": cfg['training'].get("save_steps", 500),
        "save_total_limit": cfg['training'].get("save_total_limit", 2),
        "load_best_model_at_end": cfg['training'].get("load_best_model_at_end", eval_dataset is not None), # Only if eval is done
        "metric_for_best_model": cfg['training'].get("metric_for_best_model", "eval_f1" if eval_dataset else None),
        "greater_is_better": cfg['training'].get("greater_is_better", True),
        "report_to": [report_to] if report_to != "none" else [],
        "run_name": run_name,
        "label_names": ["labels"], # Standard practice
        "remove_unused_columns": False, # We already removed them in data module
        "ddp_find_unused_parameters": cfg['training'].get("ddp_find_unused_parameters", False),
    }

    training_args = TrainingArguments(**training_args_dict)
    logger.info(f"Training arguments: {training_args}. FP16 Enabled: {training_args.fp16}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    # --- TRAIN ---
    logger.info("Training model...")
    train_result = trainer.train()
    logger.info(f"Training results: {train_result}")

    # Save final model & metrics
    logger.info(f"Saving best model to {training_args.output_dir}")
    trainer.save_model() # Saves the best model due to load_best_model_at_end=True
    trainer.save_state()

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate on test set if available
    test_dataset = data_module.get_test_dataset()
    if test_dataset and cfg['training'].get("do_test_eval", True):
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info(f"Test set evaluation complete: {test_metrics}")


    logger.info("Script finished successfully.")


if __name__ == "__main__":
    app()