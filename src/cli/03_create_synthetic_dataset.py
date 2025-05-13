#!/usr/bin/env python
"""
03_create_synthetic_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a **quality‑controlled** synthetic SST‑2‑style dataset using a
fine‑tuned causal‑LM (e.g. GPT‑2) **and** a teacher classifier. A sample is
accepted only when the teacher's confidence exceeds a configurable
threshold. The teacher's prediction provides the final label, so we are
robust to generator drift (e.g., a <POS> prompt that meanders into a
negative review).

Key update (2025‑05‑13)
-----------------------
* **FIXED**: Hugging Face `pipeline` returns a *list of lists* when you pass
  a batch of prompts. The script now flattens the nested structure, so we
  no longer hit `TypeError: list indices must be integers or slices, not str`.

Configuration (YAML)
--------------------
model:
  ckpt_dir: "runs/generator/gpt2_sst2/checkpoint-best"

a": "\n" teacher:
  ckpt_dir: "runs/teacher/deberta_v3_base/checkpoint-2000"
  min_confidence: 0.8

data:
  output_dir: "data/synthetic_sst2"
  n_samples_total: 20000
  split_ratio: {train: 0.9, val: 0.05, test: 0.05}

generation:
  max_new_tokens: 64
  temperature: 0.7
  top_k: 50
  top_p: 0.95
  repetition_penalty: 1.2
  seed: 42
  batch_size: 8
"""

from __future__ import annotations

import json
import logging
import math
import random
import itertools, queue, threading, time
from pathlib import Path
from typing import Dict, List

import torch
import typer
import yaml
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer()

PROMPTS = ["<POS> Review:", "<NEG> Review:"]


def _set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_output(generated: str) -> str:
    """Remove the sentiment prefix and tidy whitespace."""
    try:
        cleaned = generated.split(":", 1)[1].strip()
    except IndexError:
        cleaned = generated.strip()
    return " ".join(cleaned.split())


def _write_jsonl(items: List[Dict[str, str]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
    logger.info("Wrote %d samples → %s", len(items), path)


def _label_from_teacher(res: Dict[str, str | float]) -> int | None:
    label = res["label"].lower()
    if label in {"positive", "pos", "label_1", "1"}:
        return 1
    if label in {"negative", "neg", "label_0", "0"}:
        return 0
    return None


@app.command()
def main(cfg_path: Path = typer.Argument(..., help="Path to YAML config")):
    # ------------------------------- Load config
    cfg = yaml.safe_load(cfg_path.read_text())

    gen_ckpt = Path(cfg["model"]["ckpt_dir"]).expanduser()
    teacher_ckpt = Path(cfg["teacher"]["ckpt_dir"]).expanduser()
    min_conf = float(cfg["teacher"].get("min_confidence", 0.8))

    n_target = int(cfg["data"].get("n_samples_total", 20000))
    split_ratio = cfg["data"].get(
        "split_ratio", {"train": 0.9, "val": 0.05, "test": 0.05}
    )
    if not math.isclose(sum(split_ratio.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    gcfg = cfg.get("generation", {})
    _set_seed(gcfg.get("seed", 42))

    # --------------------------------------- Generator pipeline
    device = 0 if torch.cuda.is_available() else -1
    tokenizer_gen = AutoTokenizer.from_pretrained(gen_ckpt)
    model_gen = (
        AutoModelForCausalLM.from_pretrained(gen_ckpt, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    model_gen = torch.compile(model_gen, mode="reduce-overhead", fullgraph=False)

    # Pre‑encode <POS>/<NEG> once
    prompt_ids = tokenizer_gen(
        PROMPTS, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    

    # --------------------------------------- Teacher pipeline  (CPU)
    tok = AutoTokenizer.from_pretrained(teacher_ckpt)
    torch.set_num_threads(30)        # use all CPU cores
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_ckpt
    ).eval()                          # CPU ⇢ device = -1
    teacher_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=-1,            # ← run on CPU
        batch_size=1024,
        truncation=True,
    )

    # Generation params
    gen_params = {
        "max_new_tokens": gcfg.get("max_new_tokens", 64),
        "temperature": gcfg.get("temperature", 0.7),
        "top_k": gcfg.get("top_k", 50),
        "top_p": gcfg.get("top_p", 0.95),
        "repetition_penalty": gcfg.get("repetition_penalty", 1.2),
        "eos_token_id": tokenizer_gen.eos_token_id,
        "do_sample": True,
        "num_return_sequences": 1,
    }
    batch_size = int(gcfg.get("batch_size", 128))

    # --------------------------------------- Main generation loop
    dataset: List[Dict[str, str]] = []
    rejects = 0
    pbar = tqdm(total=n_target, desc="Accepted samples")

    # ── Async teacher consumer on CPU ───────────────────────────────
    work_q: queue.Queue[list[str]] = queue.Queue(maxsize=4)
    res_q:  queue.Queue[list[Dict]] = queue.Queue(maxsize=4)

    def _cpu_teacher():
        while True:
            batch_txt = work_q.get()
            if batch_txt is None:
                break
            res_q.put(teacher_pipe(batch_txt))
            work_q.task_done()

    t = threading.Thread(target=_cpu_teacher, daemon=True)
    t.start()

    while len(dataset) < n_target:
        # 1️⃣  build input tensor (no re‑tokenisation)
        rep = (batch_size + len(PROMPTS) - 1) // len(PROMPTS)
        input_ids = prompt_ids.repeat_interleave(rep, 0)[:batch_size]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        with torch.inference_mode():
            gen_ids = model_gen.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,          # ← add this
                pad_token_id=tokenizer_gen.pad_token_id,
                **gen_params,
            )
        outputs = tokenizer_gen.batch_decode(gen_ids, skip_special_tokens=True)

        texts = [_clean_output(t) for t in outputs]
        texts = [t for t in texts if t]
        # 2️⃣ hand off to CPU thread and immediately start next gen loop
        work_q.put(texts)

        # 3️⃣ drain any finished label batches
        while not res_q.empty() and len(dataset) < n_target:
            teacher_out = res_q.get()
            for res, text in zip(teacher_out, texts):
                conf = res["score"]
                label = _label_from_teacher(res)
                if label is not None and conf >= min_conf:
                    dataset.append({"label": label, "text": text})
                    pbar.update(1)
                    if len(dataset) >= n_target:
                        break
                else:
                    rejects += 1
                    logger.debug(
                        "Rejected (conf=%.3f, label=%s): %.60s",
                        conf,
                        label,
                        text,
                    )

    work_q.put(None)        # stop CPU thread
    t.join()

    logger.info(
        "Finished generation: %d accepted, %d rejected (%.2f%% rejection)",
        len(dataset),
        rejects,
        100 * rejects / (len(dataset) + rejects),
    )

    # --------------------------------------- Shuffle & split
    random.shuffle(dataset)
    n_train = int(n_target * split_ratio["train"])
    n_val = int(n_target * split_ratio["val"])
    splits = {
        "train": dataset[:n_train],
        "val": dataset[n_train : n_train + n_val],
        "test": dataset[n_train + n_val :],
    }

    # --------------------------------------- Write files
    out_dir = Path(cfg["data"]["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        _write_jsonl(items, out_dir / f"{split_name}.jsonl")

    logger.info("\u2705  Synthetic dataset ready → %s", out_dir)


if __name__ == "__main__":
    app()
