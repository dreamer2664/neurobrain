"""Checkpoint directories per specs/model_io.md.

Layout:
    checkpoints/<name>/
      config.json      architecture + step + stub flag + sha256 of the rest
      weights.pt       fp32 state_dict (source of truth)
      tokenizer.json   exact tokenizer used for training
      card.md          what it is / is not

`config.json` additionally carries `weights_sha256` (spec-change published by
agent 01, 2026-10-22) so `brainbox doctor`/load can verify integrity.

A "stub" checkpoint is a freshly-initialised, seeded model with the real
architecture and the real tokenizer. It exists so the runtime can be built
and measured before a trained `.pt` is published; it is flagged `stub: true`
in config.json, in card.md and in every report that used it. It is never
presented as trained.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

# Real LM dimensions, from neurobrain/train_lm.py (GPTConfig there) and
# reports/lm_summary.json (4,247,264 params at step 4652).
REAL_LM_DIMS = {"n_layer": 6, "n_head": 7, "d_model": 224, "ctx": 256}


class CheckpointError(RuntimeError):
    pass


@dataclass
class CheckpointInfo:
    path: str
    kind: str = "lm"
    step: int = -1
    stub: bool = False
    params: int = 0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "step": self.step,
                "stub": self.stub, "params": self.params}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_tokenizer_path() -> str | None:
    candidates = [
        os.environ.get("BRAINBOX_TOKENIZER"),
        os.path.join(os.getcwd(), "data", "tokenizer.json"),
        os.path.join(repo_root(), "data", "tokenizer.json"),
        os.path.expanduser("~/neurobrain/data/tokenizer.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _gpt_config(cfg: dict):
    from neurobrain.model import GPTConfig

    fields = set(GPTConfig.__dataclass_fields__)
    return GPTConfig(**{k: v for k, v in cfg.items() if k in fields})


def write_checkpoint_dir(out_dir: str, model, tokenizer_path: str, step: int,
                         stub: bool, card: str | None = None) -> CheckpointInfo:
    import torch

    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "weights.pt")
    torch.save(model.state_dict(), weights_path)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    shutil.copyfile(tokenizer_path, tok_path)

    config = dict(model.cfg.to_dict())
    config.update({
        "kind": "lm",
        "step": int(step),
        "stub": bool(stub),
        "params": model.num_params(),
        "tokenizer_sha256": sha256_file(tok_path),
        "weights_sha256": sha256_file(weights_path),
    })
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "card.md"), "w") as f:
        f.write(card or _default_card(config))
    return CheckpointInfo(path=out_dir, step=int(step), stub=bool(stub),
                          params=config["params"], config=config)


def _default_card(config: dict) -> str:
    kind = "STUB (fresh random init, NOT trained)" if config.get("stub") else "trained"
    return (f"# lm checkpoint — {kind}\n\n"
            f"- step: {config.get('step')}\n"
            f"- params: {config.get('params')}\n"
            f"- arch: n_layer={config.get('n_layer')} n_head={config.get('n_head')} "
            f"d_model={config.get('d_model')} ctx={config.get('ctx')} "
            f"vocab={config.get('vocab_size')}\n"
            f"- tokenizer_sha256: {config.get('tokenizer_sha256')}\n")


def read_config(ckdir: str) -> dict:
    cfg_path = os.path.join(ckdir, "config.json")
    if not os.path.exists(cfg_path):
        raise CheckpointError(f"no config.json in {ckdir}")
    with open(cfg_path) as f:
        return json.load(f)


def inspect_checkpoint_dir(ckdir: str) -> dict:
    """Cheap doctor-grade inspection: config + file presence, no model build."""
    cfg = read_config(ckdir)
    files = {}
    for name in ("weights.pt", "tokenizer.json", "card.md"):
        p = os.path.join(ckdir, name)
        files[name] = os.path.getsize(p) if os.path.exists(p) else None
    return {"path": ckdir, "kind": cfg.get("kind", "lm"), "step": cfg.get("step", -1),
            "stub": bool(cfg.get("stub", False)), "params": cfg.get("params", 0),
            "vocab_size": cfg.get("vocab_size"), "files": files, "config": cfg}


def load_checkpoint_dir(ckdir: str, verify: bool = True):
    """Load (model, tokenizer, info) from a checkpoint directory, sha-verified."""
    import torch

    from neurobrain.model import GPT
    from neurobrain.tokenizer import BPETokenizer

    cfg = read_config(ckdir)
    tok_path = os.path.join(ckdir, "tokenizer.json")
    weights_path = os.path.join(ckdir, "weights.pt")
    for p in (tok_path, weights_path):
        if not os.path.exists(p):
            raise CheckpointError(f"missing {p}")
    if verify:
        if cfg.get("tokenizer_sha256") and sha256_file(tok_path) != cfg["tokenizer_sha256"]:
            raise CheckpointError(f"tokenizer sha256 mismatch in {ckdir}")
        if cfg.get("weights_sha256") and sha256_file(weights_path) != cfg["weights_sha256"]:
            raise CheckpointError(f"weights sha256 mismatch in {ckdir}")

    tok = BPETokenizer.load(tok_path)
    gpt_cfg = _gpt_config(cfg)
    if tok.vocab_size != gpt_cfg.vocab_size:
        raise CheckpointError(
            f"tokenizer vocab {tok.vocab_size} != config vocab {gpt_cfg.vocab_size}")
    model = GPT(gpt_cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    info = CheckpointInfo(path=ckdir, kind=cfg.get("kind", "lm"),
                          step=int(cfg.get("step", -1)),
                          stub=bool(cfg.get("stub", False)),
                          params=model.num_params(), config=cfg)
    return model, tok, info


def load_legacy_pt(pt_path: str, tokenizer_path: str):
    """Load the trainer's bare .pt format: {"model", "cfg", "step", ...}."""
    import torch

    from neurobrain.model import GPT
    from neurobrain.tokenizer import BPETokenizer

    tok = BPETokenizer.load(tokenizer_path)
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = GPT(_gpt_config(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    info = CheckpointInfo(path=pt_path, step=int(ck.get("step", -1)), stub=False,
                          params=model.num_params(), config=dict(ck["cfg"]))
    return model, tok, info


def migrate_legacy(pt_path: str, tokenizer_path: str, out_dir: str) -> CheckpointInfo:
    model, _tok, info = load_legacy_pt(pt_path, tokenizer_path)
    card = (f"# lm checkpoint — migrated from legacy .pt\n\n"
            f"- source: {os.path.basename(pt_path)}\n"
            f"- step: {info.step}\n- params: {info.params}\n")
    return write_checkpoint_dir(out_dir, model, tokenizer_path,
                                step=info.step, stub=False, card=card)


def build_stub_model(tokenizer, seed: int = 0, config_override: dict | None = None):
    """Fresh seeded GPT with the real dims (or a test override). Not trained."""
    import torch

    from neurobrain.model import GPT, GPTConfig

    dims = dict(REAL_LM_DIMS)
    if config_override:
        dims.update(config_override)
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, **dims))
    model.eval()
    return model


def make_stub(out_dir: str, tokenizer_path: str, seed: int = 0,
              config_override: dict | None = None) -> CheckpointInfo:
    from neurobrain.tokenizer import BPETokenizer

    tok = BPETokenizer.load(tokenizer_path)
    model = build_stub_model(tok, seed=seed, config_override=config_override)
    card = ("# lm checkpoint — STUB (fresh random init, NOT trained)\n\n"
            f"Seeded ({seed}) random initialisation with the real architecture "
            "and the real tokenizer, so brainbox can be built and measured "
            "before a trained lm_best.pt is published. Output text is noise.\n")
    return write_checkpoint_dir(out_dir, model, tokenizer_path,
                                step=0, stub=True, card=card)


def find_checkpoint() -> tuple[str, str] | None:
    """Search for a loadable LM checkpoint. Returns (path, 'dir'|'legacy')."""
    env = os.environ.get("BRAINBOX_CHECKPOINT")
    if env and os.path.exists(env):
        return env, "dir" if os.path.isdir(env) else "legacy"
    roots = [os.path.join(os.getcwd(), "checkpoints"),
             os.path.join(repo_root(), "checkpoints"),
             os.path.expanduser("~/.brainbox/checkpoints")]
    best: tuple[int, int, str] | None = None  # (not_stub, step, path)
    legacy = None
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                try:
                    cfg = read_config(p)
                except Exception:
                    continue
                key = (0 if cfg.get("stub") else 1, int(cfg.get("step", 0)), p)
                if best is None or key > best:
                    best = key
            elif name.endswith(".pt") and legacy is None:
                legacy = p
    if best is not None:
        return best[2], "dir"
    if legacy is not None:
        return legacy, "legacy"
    return None
