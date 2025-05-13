#!/usr/bin/env python
"""
Quick demo: load a fine‑tuned DeBERTa‑v3 model and run one inference.
Install deps first:
    pip install "transformers>=4.40" torch --upgrade
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# -------------------------------------------------------------------
# 1️⃣  where did Trainer save your model?
# -------------------------------------------------------------------
MODEL_DIR = "runs/teacher/deberta_v3_base/"  # <= change me!

# -------------------------------------------------------------------
# 2️⃣  easiest: high‑level pipeline
# -------------------------------------------------------------------
device = "mps"
clf = pipeline(
    task="text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=device,
)

example = "This movie was absolutely trash!"
print("\nPipeline result:")
print(clf(example))                 # [{'label': 'POSITIVE', 'score': 0.97}]

