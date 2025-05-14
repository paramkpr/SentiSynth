#!/usr/bin/env python
"""
05_train_student.py
-------------------

Train a *student* sentiment‑classification model (e.g. TinyBERT) on either

  • 1 000 “real” SST‑2 rows **only**, **or**
  • the same 1 000 real rows **plus** 20 000 synthetic rows,

depending on the `dataset_path` you pass in your YAML file.

Usage
-----
    python 05_train_student.py configs/student_real_1k.yaml
    python 05_train_student.py configs/student_mix.yaml
"""

import logging
from pathlib import Path
import typer, yaml, torch
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    IntervalStrategy,
)

# ------------- project helpers
from src.models import build_teacher  
from src.training.student_weighted_soft_trainer import WeightedSoftTrainer as Trainer
from src.data   import StudentDataModule
from src.utils.wandb_setup import setup_wandb
from src.utils.metrics     import compute_metrics     # accuracy / F1, etc.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer()

# ────────────────────────────────────────────────────────────────────────────
@app.command()
def main(config_path: Path = typer.Argument(..., help="Path to YAML config")):
    # ----------------------------------------------------------------- CONFIG
    cfg = yaml.safe_load(config_path.read_text())

    # ------------------------------------------------------ optional W&B init
    run_name, report_to = setup_wandb(cfg)

    # ---------------------------------------------------------------  MODEL
    # build_student() should return (model, tokenizer).  If you haven’t
    # implemented it yet, just import `build_teacher` and point `cfg["model"]`
    # at a tiny‑BERT checkpoint. 
    model, tokenizer = build_teacher(cfg["model"])

    # ---------------------------------------------------------------  DATA
    dm = StudentDataModule(cfg["data"], tokenizer)
    dm.setup()

    train_ds = dm.get_train_dataset()
    eval_ds  = dm.get_eval_dataset()      # usually “val” split (can be None)

    # ----------------------------------------------------- DATA COLLATOR
    data_collator = DataCollatorWithPadding(tokenizer)

    # ------------------------------------------------ TRAINING ARGUMENTS
    tr_args = TrainingArguments(
        output_dir           = cfg["training"]["output_dir"],
        overwrite_output_dir = cfg["training"].get("overwrite_output_dir", True),
        do_train             = True,
        do_eval              = eval_ds is not None,
        per_device_train_batch_size = cfg["training"].get("per_device_train_batch_size", 32),
        per_device_eval_batch_size  = cfg["training"].get("per_device_eval_batch_size", 64),
        gradient_accumulation_steps = cfg["training"].get("gradient_accumulation_steps", 1),
        num_train_epochs            = cfg["training"].get("num_train_epochs", 3),
        learning_rate               = cfg["training"].get("learning_rate", 3e-5),
        warmup_ratio                = cfg["training"].get("warmup_ratio", 0.1),
        fp16                        = cfg["training"].get("fp16", torch.cuda.is_available()),
        logging_dir                 = cfg["training"].get(
                                          "logging_dir",
                                          f"{cfg['training']['output_dir']}/logs"),
        logging_steps               = cfg["training"].get("logging_steps", 100),
        eval_strategy               = (
            IntervalStrategy.STEPS if eval_ds is not None else IntervalStrategy.NO
        ),
        eval_steps                  = cfg["training"].get("eval_steps", 500),
        save_strategy               = IntervalStrategy.STEPS,
        save_steps                  = cfg["training"].get("save_steps", 500),
        save_total_limit            = cfg["training"].get("save_total_limit", 2),
        load_best_model_at_end      = cfg["training"].get(
                                          "load_best_model_at_end", eval_ds is not None),
        metric_for_best_model       = cfg["training"].get("metric_for_best_model", "eval_f1"),
        greater_is_better           = cfg["training"].get("greater_is_better", True),
        report_to                   = [report_to] if report_to != "none" else [],
        run_name                    = run_name,
        label_names                 = ["labels"],
        remove_unused_columns       = False,           # DM already stripes columns
        ddp_find_unused_parameters  = cfg["training"].get("ddp_find_unused_parameters", False),
    )
    logger.info("TrainingArguments ready (fp16=%s).", tr_args.fp16)

    # ------------------------------------------------------------- TRAINER
    trainer = Trainer(
        model           = model,
        args            = tr_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        alpha           = cfg["training"].get("alpha", 0.5),
        temperature     = cfg["training"].get("temperature", 2.0),
    )

    # --------------------------------------------------------------- TRAIN
    logger.info("🚀  Starting training …")
    train_result = trainer.train()
    logger.info("✅  Training finished.")

    # ------------------------------------------------- SAVE & LOG METRICS
    trainer.save_model()      # saves best model if load_best_model_at_end=True
    trainer.save_state()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # ------------------------------------------------------------ TEST
    test_ds = dm.get_sanity_dataset()
    if test_ds and cfg["training"].get("do_test_eval", True):
        logger.info("🧪  Running test evaluation …")
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    logger.info("🎉  Script completed successfully.")


if __name__ == "__main__":
    app()
