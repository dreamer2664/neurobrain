# neurobrain

A neural brain trained **completely from scratch** — no pretrained weights, no
downloaded models, no local LLMs. Just a transformer, a CNN, a byte-level BPE
tokenizer and a lot of Python source code + synthetic desktop-control data.

**Two skills:**

1. **Programming** — a 4.25M-parameter decoder-only transformer learns Python by
   next-token prediction on real open-source source code (CPython stdlib,
   installed libraries, and released sdist tarballs).
2. **Desktop control** — a from-scratch CNN (`GripNet`) looks at a 128×128
   screenshot and learns to point at the right widget for a natural-language
   instruction (instruction-conditioned visual grounding). The same task is also
   serialized into a text protocol and mixed into the language brain's corpus,
   so the transformer learns the shape of screen→action transcripts.

```
                    ┌──────────────────────────────┐
   python source ──▶│  byte-level BPE (2048 tok)   │──▶ uint16 token stream
   screen protocol ─┤  learned from scratch        │        │
                    └──────────────────────────────┘        ▼
                                              ┌──────────────────────────┐
                                              │ transformer (6L, d=224)  │──▶ lm_best.pt
                                              └──────────────────────────┘
   synthetic desktop ──▶ 128×128 renders ──▶ ┌──────────────────────────┐
   instruction text   ──▶ char encoder    ──▶│ GripNet CNN + text query │──▶ grip_best.pt
                                              └──────────────────────────┘
```

## What is (and isn't) here

- **Is**: everything written from scratch in this repo — tokenizer, transformer,
  training loop, synthetic desktop environment, vision model.
- **Isn't**: any pretrained checkpoint, API call, or external model. `pip install
  torch` is used only as the tensor/autograd library.

## Repo layout

| path | what |
|---|---|
| `neurobrain/tokenizer.py` | byte-level BPE trainer + encoder, from scratch |
| `neurobrain/model.py` | GPT: RMSNorm, RoPE, causal attention, SwiGLU |
| `neurobrain/data.py` | memmapped token streams, mixed batching |
| `neurobrain/train_lm.py` | training loop (AdamW, cosine LR, eval, checkpointing) |
| `neurobrain/ui_world.py` | synthetic desktop renderer + text protocol + labels |
| `neurobrain/gripnet.py` | instruction-conditioned CNN for screen pointing |
| `neurobrain/train_grip.py` | GripNet training loop |
| `neurobrain/sample.py` | sample code from a checkpoint |
| `scripts/build_corpus.py` | corpus construction + tokenizer training |
| `scripts/run_forever.sh` | continuous train → sample → push driver |
| `reports/` | training logs (jsonl), eval metrics, corpus stats |
| `samples/` | generated text and prediction images from checkpoints |

## Training data

- **Code**: real Python from CPython's stdlib, the installed site-packages, and
  sdist releases of flask / django / sqlalchemy / requests / fastapi / pytest and
  friends. Deduplicated, split 97/3 train/val.
- **Desktop**: fully synthetic. A random desktop layout (windows, buttons,
  fields, checkboxes, tabs, icons, menus, sliders) rendered to 128×128, with a
  randomly chosen instruction and the correct click target. Labels are encoded
  as unique glyph chips so that finding a widget is a real visual search task.
  Coordinates in the text protocol are in 1920-scale screen pixels, while the
  vision model works in 128-scale pixels.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/build_corpus.py --max-code-bytes 160000000 --vocab-size 2048
python -m neurobrain.train_lm --resume --steps 40000 --batch 12 --accum 2 --ctx 256
python -m neurobrain.train_grip --steps 4000 --batch 32
python -m neurobrain.sample --ckpt checkpoints/lm_best.pt --prompt "# quicksort\n"
```

Checkpoints live in `checkpoints/` and are git-ignored (they are rebuilt by the
trainer); logs and metrics are committed continuously by `scripts/push.sh`.

## Status

Training is running continuously on a 2-core / 2 GB CPU sandbox (the two models
take turns, see `scripts/train_all.sh`). See
`reports/lm_log.jsonl`, `reports/grip_log.jsonl` and the latest commit messages
for live progress, and `samples/` for fresh generations.

| | value |
|---|---|
| tokenizer | 2048-token byte-level BPE, learned from scratch (2.9 bytes/token) |
| language model | 4.25M params, 6 layers, d=224, ctx=256, ~6.5k tokens/s on 2 CPU cores |
| corpus | see `reports/corpus_report.json` |
| GripNet | 4-block CNN + char encoder + dot-product attention over 8×8 grid |

Nothing here is a product; it is a small brain being grown in public.
