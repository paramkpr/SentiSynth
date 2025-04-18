import os
import logging

logger = logging.getLogger(__name__)


def setup_wandb(cfg: dict):
    """Setup WANDB for logging."""
    project_name = cfg.get("project_name", "senti-synth-teacher")
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_PROJECT"] = project_name
    logger.info(f"Reporting to W&B project: {project_name}")
