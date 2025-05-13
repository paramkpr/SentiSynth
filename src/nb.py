#!/usr/bin/env python
import random, torch, typer
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main(
    ckpt_dir: Path = typer.Argument(..., help="Path to fine‑tuned checkpoint"),
    n: int = typer.Option(10, help="How many reviews to generate"),
):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model     = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    special_tokens = {"additional_special_tokens": ["<POS>", "<NEG>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    device    = 0 if torch.cuda.is_available() else -1

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    prompts = ["<POS>", "<NEG>"]
    for i in range(n):
        prompt = random.choice(prompts)
        out = gen(
            prompt,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].strip()
        print(f"{i+1:02d}. {out}")

if __name__ == "__main__":
    typer.run(main)
