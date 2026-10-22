"""int8 weight-only quantization (M7).

Per-output-row symmetric int8 for every >=2-D float tensor (Linear, Conv2d,
Embedding weights); norms/biases/scalars stay fp32. At runtime the quantized
modules dequantize their weight per forward, so resident weight memory drops
~4x at the cost of a small transient fp32 copy (largest layer here is
0.6 MB). The on-disk sidecar is `weights.int8.pt` in the checkpoint
directory; `weights.pt` (fp32) remains the source of truth.

Format: {name: {"q": int8 tensor, "scale": fp32 broadcast tensor}} for
quantized entries, {name: {"fp32": tensor}} for the rest — loadable with
weights_only=True.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- tensor level
def _quantize_tensor(w: torch.Tensor):
    flat = w.detach().float().reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-12)
    q = torch.round(flat / scale).clamp(-127, 127).to(torch.int8)
    bshape = (w.shape[0],) + (1,) * (w.ndim - 1)
    return q.reshape(w.shape), scale.reshape(bshape).contiguous()


def _dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def quantize_state_dict(sd: dict) -> dict:
    out = {}
    for name, t in sd.items():
        if t.is_floating_point() and t.ndim >= 2:
            q, scale = _quantize_tensor(t)
            out[name] = {"q": q, "scale": scale}
        else:
            out[name] = {"fp32": t.detach().clone()}
    return out


def dequantize_state_dict(qsd: dict) -> dict:
    out = {}
    for name, e in qsd.items():
        out[name] = _dequantize_tensor(e["q"], e["scale"]) if "q" in e else e["fp32"]
    return out


# --------------------------------------------------------------- module level
class QuantLinear(nn.Module):
    def __init__(self, q, scale, bias=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, shared: dict):
        key = lin.weight.data_ptr()
        if key not in shared:
            shared[key] = _quantize_tensor(lin.weight)
        q, scale = shared[key]
        return cls(q, scale, lin.bias)

    def forward(self, x):
        return F.linear(x, _dequantize_tensor(self.q, self.scale), self.bias)


class QuantConv2d(nn.Module):
    def __init__(self, q, scale, bias, stride, padding, dilation, groups):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.stride, self.padding = stride, padding
        self.dilation, self.groups = dilation, groups

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, shared: dict):
        key = conv.weight.data_ptr()
        if key not in shared:
            shared[key] = _quantize_tensor(conv.weight)
        q, scale = shared[key]
        return cls(q, scale, conv.bias, conv.stride, conv.padding,
                   conv.dilation, conv.groups)

    def forward(self, x):
        return F.conv2d(x, _dequantize_tensor(self.q, self.scale), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class QuantEmbedding(nn.Module):
    def __init__(self, q, scale, padding_idx=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.padding_idx = padding_idx

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, shared: dict):
        key = emb.weight.data_ptr()
        if key not in shared:
            shared[key] = _quantize_tensor(emb.weight)
        q, scale = shared[key]
        return cls(q, scale, emb.padding_idx)

    def forward(self, idx):
        return F.embedding(idx, _dequantize_tensor(self.q, self.scale),
                           padding_idx=self.padding_idx)


def _iter_parents(model: nn.Module):
    for parent_name, parent in list(model.named_modules()):
        for attr, child in list(parent.named_children()):
            yield parent, attr, child


@torch.no_grad()
def quantize_model(model: nn.Module) -> nn.Module:
    """In-place: swap Linear/Conv2d/Embedding for int8 dequant-on-forward
    versions. Tied weights share one quantized storage."""
    shared: dict = {}
    for parent, attr, child in _iter_parents(model):
        if isinstance(child, nn.Linear):
            setattr(parent, attr, QuantLinear.from_linear(child, shared))
        elif isinstance(child, nn.Conv2d):
            setattr(parent, attr, QuantConv2d.from_conv(child, shared))
        elif isinstance(child, nn.Embedding):
            setattr(parent, attr, QuantEmbedding.from_embedding(child, shared))
    model.eval()
    return model


@torch.no_grad()
def load_quantized_into(model: nn.Module, qsd: dict) -> nn.Module:
    """Load an int8 sidecar dict into a fresh model without ever
    materializing the full fp32 state dict."""
    model.load_state_dict({k: e["fp32"] for k, e in qsd.items() if "fp32" in e},
                          strict=False, assign=True)
    prefix_of = {id(m): name for name, m in model.named_modules()}
    for parent, attr, child in _iter_parents(model):
        prefix = prefix_of[id(child)]
        wname = f"{prefix}.weight"
        entry = qsd.get(wname)
        if entry is None or "q" not in entry:
            continue
        bias_entry = qsd.get(f"{prefix}.bias")
        bias = bias_entry.get("fp32") if bias_entry else None
        if isinstance(child, nn.Linear):
            setattr(parent, attr, QuantLinear(entry["q"], entry["scale"], bias))
        elif isinstance(child, nn.Conv2d):
            setattr(parent, attr, QuantConv2d(
                entry["q"], entry["scale"], bias, child.stride, child.padding,
                child.dilation, child.groups))
        elif isinstance(child, nn.Embedding):
            setattr(parent, attr, QuantEmbedding(entry["q"], entry["scale"],
                                                 child.padding_idx))
    model.eval()
    return model


# ------------------------------------------------------------ checkpoint io
def write_int8_sidecar(ckdir: str, model: nn.Module) -> str:
    from .checkpoints import read_config, sha256_file

    qsd = quantize_state_dict(model.state_dict())
    path = os.path.join(ckdir, "weights.int8.pt")
    torch.save(qsd, path)
    cfg = read_config(ckdir)
    cfg["weights_int8_sha256"] = sha256_file(path)
    cfg["quantized"] = "int8-weight-only per-row symmetric (sidecar)"
    with open(os.path.join(ckdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def load_lm_int8_dir(ckdir: str):
    """Load an LM checkpoint from weights.int8.pt only. The model is built
    on the meta device and materialized by the sidecar, so the fp32
    weights are never allocated or read. Returns (model, tokenizer, info).
    """
    from .checkpoints import (CheckpointError, CheckpointInfo, _gpt_config,
                              read_config, sha256_file)

    cfg = read_config(ckdir)
    tok_path = os.path.join(ckdir, "tokenizer.json")
    if cfg.get("tokenizer_sha256") and sha256_file(tok_path) != cfg["tokenizer_sha256"]:
        raise CheckpointError(f"tokenizer sha256 mismatch in {ckdir}")
    qsd = load_int8_sidecar(ckdir)  # verifies its own sha

    from neurobrain.model import GPT, build_rope
    from neurobrain.tokenizer import BPETokenizer

    tok = BPETokenizer.load(tok_path)
    gpt_cfg = _gpt_config(cfg)
    if tok.vocab_size != gpt_cfg.vocab_size:
        raise CheckpointError("tokenizer/config vocab mismatch")
    with torch.device("meta"):
        model = GPT(gpt_cfg)
    load_quantized_into(model, qsd)
    for mod in model.modules():  # rebuild non-persistent RoPE buffers
        if hasattr(mod, "cos") and getattr(mod, "cos", None) is not None \
                and mod.cos.is_meta:
            cos, sin = build_rope(mod.head_dim, gpt_cfg.ctx)
            mod.cos, mod.sin = cos, sin
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    meta_left += [n for n, b in model.named_buffers() if b.is_meta]
    if meta_left:
        raise CheckpointError(f"int8 load left meta tensors: {meta_left}")
    model.eval()
    info = CheckpointInfo(path=ckdir, kind=cfg.get("kind", "lm"),
                          step=int(cfg.get("step", -1)),
                          stub=bool(cfg.get("stub", False)),
                          params=int(cfg.get("params", 0)), config=cfg)
    return model, tok, info


def load_int8_sidecar(ckdir: str) -> dict:
    from .checkpoints import CheckpointError, read_config, sha256_file

    path = os.path.join(ckdir, "weights.int8.pt")
    if not os.path.exists(path):
        raise CheckpointError(f"no weights.int8.pt in {ckdir}")
    cfg = read_config(ckdir)
    want = cfg.get("weights_int8_sha256")
    if want and sha256_file(path) != want:
        raise CheckpointError(f"weights.int8 sha256 mismatch in {ckdir}")
    return torch.load(path, map_location="cpu", weights_only=True)
