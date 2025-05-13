#!/usr/bin/env python
"""
Build two DatasetDicts for student‑model experiments:

• real_1k/   – 1 000‑row sample of the original SST‑2
• mix_20k/   – same 1 000 real rows  +  20 000 synthetic rows

Both folders keep the original val / test / sanity splits unchanged.
"""
import random, typer
from pathlib import Path
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict

app = typer.Typer()

@app.command()
def main(
    real_ds_path:  Path = typer.Argument(..., help="path to original SST‑2 DatasetDict"),
    synth_dir:     Path = typer.Argument(..., help="folder with train/val/test JSONL produced earlier"),
    out_root:      Path = typer.Argument("data/", help="where to write the new DatasetDicts"),
    real_sample:   int  = typer.Option(1_000, help="# real rows to keep"),
    seed:          int  = 42,
):
    random.seed(seed)

    # ------------------------- load original SST‑2
    real = load_from_disk(real_ds_path)          # splits: train / val / sanity / test
    assert "text" in real["train"].column_names and "label" in real["train"].column_names

    # ------------------------- load synthetic JSONL
    synth = load_dataset(
        "json",
        data_files={
            "train": synth_dir / "train.jsonl",
            "val":   synth_dir / "val.jsonl",
            "test":  synth_dir / "test.jsonl",
        },
    )

    # harmonise column name → the datamodule expects “labels”
    synth = synth.rename_column("label", "labels")
    real  = real.rename_column("label", "labels") if "label" in real["train"].column_names else real

    # ------------------------- 1 000‑row real‑only set
    real_1k_train = real["train"].shuffle(seed=seed).select(range(real_sample))
    real_1k = DatasetDict(
        train   = real_1k_train,
        val     = real["val"],
        sanity  = real["sanity"],
        test    = real["test"],
    )
    out1 = out_root / "real_1k"
    real_1k.save_to_disk(out1)
    print(f"✅  saved {out1}")

    # ------------------------- 20 k synth  + 1 k real
    mix_train  = concatenate_datasets([real_1k_train, synth["train"]])
    mix_val    = concatenate_datasets([real["val"],     synth["val"]])
    mix_test   = concatenate_datasets([real["test"],    synth["test"]])

    mix = DatasetDict(
        train   = mix_train,
        val     = mix_val,
        sanity  = real["sanity"],
        test    = mix_test,
    )
    out2 = out_root / "mix_20k_real_1k"
    mix.save_to_disk(out2)
    print(f"✅  saved {out2}")

if __name__ == "__main__":
    typer.run(main)
