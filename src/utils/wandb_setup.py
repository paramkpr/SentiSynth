import os
import logging
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_wandb(cfg: dict):
    """Setup WANDB for logging."""
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
    return run_name, report_to