#!/usr/bin/env python
"""
Build two DatasetDicts for student‑model experiments:

• real_1k/   – 1 000‑row sample of the original SST‑2
• mix_20k/   – same 1 000 real rows  +  20 000 synthetic rows

Both folders keep the original val / test / sanity splits unchanged.
"""
import random
from pathlib import Path
import typer

import torch
import torch.nn.functional as F
from datasets import (
    load_from_disk,
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    ClassLabel,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

app = typer.Typer()


@app.command()
def main(
    real_ds_path: Path = typer.Argument(
        ..., help="Path to original SST‑2 DatasetDict (with train / val / test / sanity)"
    ),
    synth_dir: Path = typer.Argument(
        ..., help="Folder containing train/val/test JSONL produced earlier"
    ),
    out_root: Path = typer.Argument(
        "data/", help="Parent directory where new DatasetDicts are written"
    ),
    real_sample: int = typer.Option(1_000, help="# real rows to keep in each run"),
    teacher_ckpt: Path = typer.Option(..., help="Checkpoint dir for the teacher"),
    temperature: float = typer.Option(2.0, help="Soft‑label temperature"),
    seed: int = 42,
):
    random.seed(seed)

    # ---------------------------------------------------------------------
    # 1.  Teacher pipeline (single instantiation)
    # ---------------------------------------------------------------------
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(teacher_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_ckpt).eval()
    teacher = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=device,         # GPU iff available
        batch_size=128,
        top_k=None,            # → list‑of‑lists with both classes
        truncation=True,
        max_length=512,
    )

    def _extract_probs(scores):
        """Return (p_neg, p_pos) from pipeline output."""
        p_neg = next(
            d["score"]
            for d in scores
            if "NEG" in d["label"].upper() or "LABEL_0" in d["label"].upper()
        )
        p_pos = next(
            d["score"]
            for d in scores
            if "POS" in d["label"].upper() or "LABEL_1" in d["label"].upper()
        )
        return p_neg, p_pos

    # ---------------------------------------------------------------------
    # 2.  Load real SST‑2 and synthetic JSONL
    # ---------------------------------------------------------------------
    real = load_from_disk(real_ds_path)  # splits: train / val / sanity / test
    for split in ("train", "val", "test", "sanity"):
        if "idx" in real[split].column_names:
            real[split] = real[split].remove_columns("idx")

    synth = load_dataset(
        "json",
        data_files={
            "train": str(synth_dir / "train.jsonl"),
            "val": str(synth_dir / "val.jsonl"),
            "test": str(synth_dir / "test.jsonl"),
        },
    )

    # Harmonise column names
    if "label" in real["train"].column_names:
        real = real.rename_column("label", "labels")
    if "label" in synth["train"].column_names:
        synth = synth.rename_column("label", "labels")

    # Cast to ClassLabel so everything lines up
    label_cls = ClassLabel(names=["negative", "positive"])
    for ds in (real, synth):
        for split in ds:
            ds[split] = ds[split].cast_column("labels", label_cls)

    # ---------------------------------------------------------------------
    # 3.  Augment REAL data with weights + soft labels (batched)
    # ---------------------------------------------------------------------
    def augment_real_batch(batch):
        scores_batch = teacher(batch["text"])
        soft_batch = []
        for scores in scores_batch:
            p_neg, p_pos = _extract_probs(scores)
            scaled = F.softmax(
                torch.tensor([p_neg, p_pos]) / temperature, dim=-1
            ).tolist()
            soft_batch.append(scaled)

        n = len(batch["text"])
        return {
            "weights": [1.0] * n,
            "is_synth": [0] * n,
            "soft_labels": soft_batch,
        }

    real = real.map(
        augment_real_batch,
        batched=True,
        desc="🏷️  Adding weights & soft labels to REAL data",
        load_from_cache_file=False,
    )

    # ---------------------------------------------------------------------
    # 4.  Build real‑only (1 k) DatasetDict
    # ---------------------------------------------------------------------
    real_1k_train = real["train"].shuffle(seed=seed).select(range(real_sample))
    real_1k = DatasetDict(
        train=real_1k_train,
        val=real["val"],
        sanity=real["sanity"],
        test=real["test"],
    )
    out1 = out_root / "real_1k"
    real_1k.save_to_disk(out1)
    print(f"✅  saved {out1}")

    # ---------------------------------------------------------------------
    # 5.  Build mixed DatasetDict (1 k real + synth)
    # ---------------------------------------------------------------------
    mix = DatasetDict(
        train=concatenate_datasets([real_1k_train, synth["train"]]),
        val=concatenate_datasets([real["val"], synth["val"]]),
        sanity=real["sanity"],
        test=concatenate_datasets([real["test"], synth["test"]]),
    )
    out2 = out_root / "mix_20k_real_1k_beta_0.3"
    mix.save_to_disk(out2)
    print(f"✅  saved {out2}")


if __name__ == "__main__":
    typer.run(main)
