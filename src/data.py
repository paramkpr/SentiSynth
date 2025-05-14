import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, GPT2Tokenizer
logger = logging.getLogger(__name__)


class ClassificationDataModule:
    """
    Data module for classification tasks. Handles standard text classification setup.
    Used by Teacher (training/eval) and Student (eval). 
    """
    def __init__(self, cfg: dict, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset_path = cfg.get("dataset_path", None)
        self.max_len = cfg.get("max_len", 128)
        
        self.tokenized_datasets = None

        self.text_column  = cfg.get("text_column",  "text")
        self.label_column = cfg.get("label_column", "labels")

        self.required_splits = ["train", "val", "sanity", "test"]
        self.required_columns = [self.text_column, self.label_column]

        self.pos_token    = cfg.get("pos_token", "<POS>")
        self.neg_token    = cfg.get("neg_token", "<NEG>")

    def _load_clean_dataset(self) -> DatasetDict:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        dataset = load_from_disk(self.dataset_path)

        missing_splits = [s for s in self.required_splits if s not in dataset]
        missing_cols = [c for c in self.required_columns if c not in dataset["train"].column_names]
        
        if missing_splits:
            raise ValueError(f"Dataset missing splits: {missing_splits}")
        if missing_cols:
            raise ValueError(f"Dataset missing columns: {missing_cols}")
                
        return dataset
    
    def _tokenize_function(self, batch: Dict[str, List]):
        tokenised = self.tokenizer(
            batch[self.text_column],
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )
        # 🡒  keep labels so Trainer can build loss
        tokenised["labels"] = batch[self.label_column]
        return tokenised

    
    def setup(self):
        """Loads and tokenizes the dataset."""
        if self.tokenized_datasets:
            return
        
        raw_datasets = self._load_clean_dataset()
        self.tokenized_datasets = raw_datasets.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[c for c in raw_datasets["train"].column_names if c not in
                             ["input_ids", "attention_mask", "labels"]]
        )
        
        logger.info(f"Loaded and tokenized datasets with max length: {self.max_len}")
        logger.info(f"Columns in tokenized datasets: {self.tokenized_datasets['train'].column_names}")

    def get_train_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["train"]
    
    def get_eval_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["val"]
    
    def get_sanity_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["sanity"]
    
    def get_test_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["test"]
    


class GeneratorDataModule:
    """
    Data module for generative fine-tuning tasks.
    Handles text generation setup.
    """
    def __init__(self, cfg: dict, tokenizer: GPT2Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset_path = cfg.get("dataset_path", None)

        self.max_len = cfg.get("max_len", 32)

        self.tokenized_datasets = None

        self.required_splits = ["train", "val", "sanity", "test"]
        self.text_column = "text"

    def _load_clean_dataset(self) -> DatasetDict:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        dataset = load_from_disk(self.dataset_path)
        
        missing_splits = [s for s in self.required_splits if s not in dataset]
        if missing_splits:
            raise ValueError(f"Dataset missing splits: {missing_splits}")
        
        return dataset
    
    def _tokenize_function(self, examples):
        """Tokenization function for map."""
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            padding=False,
            max_length=self.max_len
        )
    
    def setup(self):
        """Loads and tokenizes the dataset."""
        if self.tokenized_datasets:
            return
        
        raw_datasets = self._load_clean_dataset()


        self.tokenized_datasets = raw_datasets.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[c for c in raw_datasets["train"].column_names if c not in
                             ["input_ids", "attention_mask", "labels"]]
        )

        logger.info(f"Loaded and tokenized datasets with max length: {self.max_len}")
        logger.info(f"Columns in tokenized datasets: {self.tokenized_datasets[self.required_splits[0]].column_names}")


    def get_train_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["train"]
    
    def get_eval_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["val"]

    def get_sanity_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["sanity"]

    def get_test_dataset(self):
        if not self.tokenized_datasets: self.setup()  # noqa: E701
        return self.tokenized_datasets["test"]


class StudentDataModule:
    """
    Dataset helper for *classification* models.
    • NO <POS>/<NEG> prefixes
    • Keeps the label column so Trainer can compute loss
    """

    def __init__(self, cfg: Dict, tokenizer: AutoTokenizer):
        self.cfg       = cfg
        self.tokenizer = tokenizer

        self.dataset_path = Path(cfg["dataset_path"]).expanduser()
        self.max_len      = cfg.get("max_len", 128)

        self.text_column  = cfg.get("text_column",  "text")
        self.label_column = cfg.get("label_column", "labels")

        self.required_splits  = ["train", "val", "sanity", "test"]
        self.required_columns = [self.text_column, self.label_column]

        self.dataset:   Optional[DatasetDict] = None
        self.tokenized: Optional[DatasetDict] = None

    # ------------------------------------------------------------------ public
    def setup(self):
        self.dataset = self._load_clean_dataset()

        # --> keep labels; drop only raw text (we add tokenised ids instead)
        self.tokenized = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[self.text_column],  # keep 'labels'!
            desc="Tokenising dataset",
            load_from_cache_file=True,
        )

    def get_train_dataset(self):
        return self.tokenized["train"]

    def get_eval_dataset(self):
        return self.tokenized.get("val", None)

    def get_sanity_dataset(self):
        return self.tokenized.get("sanity", None)

    # ---------------------------------------------------------------- internal
    def _load_clean_dataset(self) -> DatasetDict:
        logger.info("Loading dataset from %s", self.dataset_path)
        ds = load_from_disk(str(self.dataset_path))

        missing = [s for s in self.required_splits if s not in ds]
        if missing:
            raise ValueError(f"Dataset missing splits {missing}")

        missing_cols = [
            c for c in self.required_columns if c not in ds["train"].column_names
        ]
        if missing_cols:
            raise ValueError(f"Dataset missing columns {missing_cols}")

        return ds

    def _tokenize_function(self, batch: Dict[str, List]):
        tokenised = self.tokenizer(
            batch[self.text_column],
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )
        # 🡒  keep labels so Trainer can build loss
        tokenised["labels"] = batch[self.label_column]
        return tokenised