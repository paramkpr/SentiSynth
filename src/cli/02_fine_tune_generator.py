# generator_finetune.py
import math
import logging
from pathlib import Path

import typer
import yaml
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

from src.data import GeneratorDataModule
from src.models import build_generator
from src.utils.wandb_setup import setup_wandb
from src.utils.metrics import perplexity_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()




@app.command()
def main(
    config_path: Path = typer.Argument(..., help="Path to YAML config"),
):
    # ------------------------------------------------------------------ CONFIG
    cfg = yaml.safe_load(config_path.read_text())

    # ------------------------------ W&B (optional – falls back to “none”)
    run_name, report_to = setup_wandb(cfg)

    # ------------------------------------------------------- MODEL & TOKENISER
    model, tokenizer = build_generator(cfg["model"])

    # ➡️ 1.  Add sentiment‑prefix tokens so the model always sees them first
    special_tokens = {"additional_special_tokens": ["<POS>", "<NEG>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    def freeze_lower_layers(m, keep_last=4):
        total = len(m.transformer.h)
        cutoff = total - keep_last
        for i, block in enumerate(m.transformer.h):
            if i < cutoff:
                for p in block.parameters():
                    p.requires_grad = False

    freeze_lower_layers(model, keep_last=cfg["model"].get("unfrozen_layers", 4))

    # -----------------------------------------------------------------  DATA
    dm = GeneratorDataModule(cfg["data"], tokenizer)
    dm.setup()
    train_ds = dm.get_train_dataset()
    eval_ds = dm.get_eval_dataset()

    # ---------------------------------- DATALOADER COLLATOR (causal‑LM, no MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ------------------------------------------------ TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        overwrite_output_dir=cfg["training"].get("overwrite_output_dir", True),
        do_train=True,
        do_eval=eval_ds is not None,
        per_device_train_batch_size=cfg["training"].get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg["training"].get("num_train_epochs", 3),
        learning_rate=cfg["training"].get("learning_rate", 5e-5),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        fp16=cfg["training"].get("fp16", torch.cuda.is_available()),
        logging_dir=cfg["training"].get("logging_dir", f"{cfg['training']['output_dir']}/logs"),
        logging_steps=cfg["training"].get("logging_steps", 100),
        eval_strategy=(
            IntervalStrategy.STEPS if eval_ds is not None else IntervalStrategy.NO
        ),
        eval_steps=cfg["training"].get("eval_steps", 500),
        save_strategy=IntervalStrategy.STEPS,
        save_steps=cfg["training"].get("save_steps", 500),
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        load_best_model_at_end=cfg["training"].get(
            "load_best_model_at_end", eval_ds is not None
        ),
        metric_for_best_model=cfg["training"].get(
            "metric_for_best_model", "eval_loss" if eval_ds else None
        ),
        greater_is_better=False,  # lower perplexity is better
        report_to=[report_to] if report_to != "none" else [],
        run_name=run_name,
        remove_unused_columns=False,
        ddp_find_unused_parameters=cfg["training"].get("ddp_find_unused_parameters", False),
    )
    logger.info("Training args created (fp16=%s).", training_args.fp16)

    # --------------------------------------------------------------- TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=perplexity_metrics if eval_ds is not None else None,
    )

    # ------------------------------------------------- TRAINING LOOP
    logger.info("🚀  Starting training …")
    train_result = trainer.train()
    logger.info("✅  Training finished")

    # --------------------------- SAVE FINALISED CHECKPOINT & METRICS
    trainer.save_model()          # saves best if load_best_model_at_end=True
    trainer.save_state()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # ------------------------------------------ OPTIONAL TEST EVALUATION
    test_ds = dm.get_sanity_dataset()
    if test_ds and cfg["training"].get("do_test_eval", True):
        logger.info("🧪  Running test evaluation …")
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    logger.info("🎉  Script completed successfully.")


if __name__ == "__main__":
    app()
