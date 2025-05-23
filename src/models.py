"""
Model factory for the teacher model.
"""
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

def build_teacher(cfg: dict):
    """
    Builds and returns the teacher model and tokenizer using Hugging Face.

    Args:
        cfg (dict): Configuration dictionary for the model, expecting keys like:
            - model_name (str): Hugging Face model identifier.
            - num_labels (int): Number of classification labels.

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = cfg.get("model_name", "microsoft/deberta-v3-base")
    num_labels = cfg.get("num_labels", 2)
    use_fast_tokenizer = cfg.get("use_fast_tokenizer", True)

    logger.info(f"Loading teacher model: {model_name} with {num_labels} labels.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    logger.info(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)

    return model, tokenizer


def build_generator(cfg: dict):
    """
    Builds and returns the generator model and tokenizer using Hugging Face.

    Args:
        cfg (dict): Configuration dictionary for the model, expecting keys like:

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = cfg.get("model_name", "gpt2")
    use_fast_tokenizer = cfg.get("use_fast_tokenizer", True)

    logger.info(f"Loading generator model: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer