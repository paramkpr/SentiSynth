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
import re
import logging
import math
import random
import itertools, queue, threading, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import typer
import yaml
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

from src.utils.prompts import POS_PROMPTS, NEG_PROMPTS, PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer()




def _set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_output(generated: str) -> str:
    """Remove the sentiment prefix and tidy whitespace."""
    _NON_ALPHA_RE  = re.compile(r"[^A-Za-z'\s]")
    _STRAY_S_RE    = re.compile(r"(?<![A-Za-z])'s\b")   # 's not preceded by a letter
    _MULTI_SPACE_RE = re.compile(r"\s+")

    try:
        cleaned = generated.split(":", 1)[1].strip()
    except IndexError:
        cleaned = generated.strip()
    cleaned = " ".join(cleaned.split())

    cleaned = _STRAY_S_RE.sub("", cleaned)         # 1. kill orphan 's
    cleaned = _NON_ALPHA_RE.sub(" ", cleaned)      # 2. strip non‑alpha chars
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)    # 3. tidy spacing
    return cleaned.strip()


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
        device=device,            
        batch_size=128,
        truncation=True,
    )
    beta        = cfg["teacher"].get("beta", 0.3)
    temperature = cfg["teacher"].get("temperature", 2.0)

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

    # ── Async teacher (CPU) ─────────────────────────────────────────
    work_q: queue.Queue[list[str]]           = queue.Queue(maxsize=30)
    res_q:  queue.Queue[tuple[list[str],     # original cleaned texts
                              list[list[Dict[str, float]]]]] = queue.Queue()

    def _cuda_teacher():
        while True:
            batch_txt = work_q.get()
            if batch_txt is None:
                break
            # Single forward pass → list‑of‑lists (both labels)
            scores = teacher_pipe(batch_txt, top_k=None)
            res_q.put((batch_txt, scores))
            work_q.task_done()

    t = threading.Thread(target=_cuda_teacher, daemon=True)
    t.start()

    target_pos = n_target // 2
    target_neg = n_target - target_pos          # works for odd n_target, too
    pos_needed, neg_needed = target_pos, target_neg

    while pos_needed > 0 or neg_needed > 0:
        # ───────────────────────────────────────────────────────── PROMPTS
        # How many of each sentiment to put into *this* batch?
        n_pos = min(batch_size // 2, pos_needed)
        n_neg = min(batch_size - n_pos, neg_needed)

        # Build prompt list & shuffle so the generator sees mixed order
        prompts = (
            random.choices(POS_PROMPTS, k=n_pos) +
            random.choices(NEG_PROMPTS, k=n_neg)
        )
        random.shuffle(prompts)

        # 1️⃣  tokenize prompts once (no repeat_interleave gymnastics)
        enc = tokenizer_gen(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False,
            add_special_tokens=False,
        ).to(device)

        # 2️⃣  generate
        with torch.inference_mode():
            gen_ids = model_gen.generate(
                **enc,
                pad_token_id=tokenizer_gen.pad_token_id,
                **gen_params,
            )

        batch_out = tokenizer_gen.batch_decode(gen_ids, skip_special_tokens=True)

        # Hand the whole cleaned batch to the CPU worker
        cleaned = [t for t in map(_clean_output, batch_out) if t]
        if cleaned:
            work_q.put(cleaned)

        # ── Consume any finished teacher results ───────────────────
        while not res_q.empty():
            texts, scores_list = res_q.get()
            for text, scores in zip(texts, scores_list):
                # scores == [{'label':'LABEL_0','score':…}, {'label':'LABEL_1','score':…}]
                prob_neg = next(d["score"] for d in scores if "LABEL_0" in d["label"].upper())
                prob_pos = next(d["score"] for d in scores if "LABEL_1" in d["label"].upper())
                conf, label = (prob_pos, 1) if prob_pos >= prob_neg else (prob_neg, 0)

                if conf < min_conf:
                    rejects += 1
                    continue

                soft_scaled = F.softmax(
                    torch.tensor([prob_neg, prob_pos]) / temperature, dim=-1
                ).tolist()

                sample = {
                    "text":        text,
                    "labels":      label,
                    "soft_labels": soft_scaled,
                    "weights":     beta,
                    "is_synth":    1,
                }

                dataset.append(sample)
                if label == 1:
                    pos_needed -= 1
                else:
                    neg_needed -= 1
                pbar.update(1)

            res_q.task_done()

        if len(dataset) >= n_target:
            break
        
    pbar.close()

    print("loop done")

    work_q.put(None)        # signal worker to stop
    t.join()

    print("thread joined")

    # Flush anything still waiting
    while not res_q.empty():
        texts, scores_list = res_q.get()
        res_q.task_done()

    print("res_q flushed")

    logger.info(
        "\n\nFinished generation: %d accepted, %d rejected (%.2f%% rejection)",
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

    # Print the first 10 samples of each split
    for split_name, items in splits.items():
        print(f"\n\n{'='*50}")
        print(f"First 10 samples of {split_name}:")
        print(f"{'='*50}\n")
        print(f"{'#':<4} {'Text':<50} {'Label':<6} {'Soft Labels':<25} {'Weight':<8} {'Synth'}")
        print(f"{'-'*4} {'-'*50} {'-'*6} {'-'*25} {'-'*8} {'-'*5}")
        for i, item in enumerate(items[:10]):
            print(f"{i+1:<4} {item['text'][:50]:<50} {item['labels']:<6} {str(item['soft_labels']):<25} {item['weights']:<8.2f} {item['is_synth']}")
        print(f"\n{'='*50}\n")

    # --------------------------------------- Write files
    out_dir = Path(cfg["data"]["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        _write_jsonl(items, out_dir / f"{split_name}.jsonl")

    logger.info("\u2705  Synthetic dataset ready → %s", out_dir)


if __name__ == "__main__":
    app()
