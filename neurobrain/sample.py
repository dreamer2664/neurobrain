"""Sample from a trained checkpoint. No pretrained weights: pass your own .pt."""

from __future__ import annotations

import argparse
import os

import torch

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer


def load(ckpt: str, tok_path: str = "data/tokenizer.json"):
    tok = BPETokenizer.load(tok_path)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, ck.get("step", -1)


def complete(model, tok, prompt: str, n: int = 200, temperature: float = 0.8, top_k: int = 40) -> str:
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=n, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_best.pt")
    ap.add_argument("--prompt", default="# a tiny helper that retries a call three times\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--out", default="samples/lm_sample.txt")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[sample] no checkpoint at {args.ckpt} yet")
        return
    model, tok, step = load(args.ckpt)
    text = complete(model, tok, args.prompt, args.tokens, args.temperature, args.top_k)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# checkpoint step {step}\n# prompt:\n{args.prompt}\n# ---- continuation ----\n{text}\n")
    print(text)


if __name__ == "__main__":
    main()
