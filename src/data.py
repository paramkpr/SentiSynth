import logging
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

        self.required_splits = ["train", "val", "sanity", "test"]
        self.required_columns = ["text", "labels"]

        self.label_column = cfg.get("label_column", "label")
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
    
    def _tokenize_function(self, examples):
        """
        Add sentiment prefix, THEN run GPT‑tokeniser.
        examples is a batch dict with keys like 'text' and 'label'.
        """
        texts = []
        labels = examples.get(self.label_column, None)

        # Build prefixed text list
        if labels is not None:
            # Supervised split (train/val/test)
            for t, lab in zip(examples[self.text_column], labels):
                prefix = self.pos_token if lab == 1 else self.neg_token
                eos    = self.tokenizer.eos_token
                texts.append(f"{prefix} {t} {eos}")
        else:
            # Unlabelled split (e.g. sanity set) – leave as‑is
            texts = examples[self.text_column]

        return self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_len,
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
        
