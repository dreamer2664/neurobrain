"""A small decoder-only Transformer, written from scratch.

No pretrained weights, no external model code: RMSNorm, rotary position
embeddings, causal self-attention via scaled_dot_product_attention and a
SwiGLU feed-forward block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    ctx: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def build_rope(head_dim: int, ctx: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(ctx).float()
    freqs = torch.outer(t, inv)          # (ctx, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    # x: (B, H, T, D); offset = absolute position of the first token in x
    # (needed by the KV-cache decode path, where x holds only new tokens)
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[2]
    c, s = cos[offset:offset + T][None, None], sin[offset:offset + T][None, None]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        cos, sin = build_rope(self.head_dim, cfg.ctx)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_cached(self, x: torch.Tensor, past: tuple | None,
                       pos0: int) -> tuple[torch.Tensor, tuple]:
        """Incremental attention for generation. `past` is (k, v) of shape
        (B, H, S, hd) already rotated at their absolute positions; `pos0` is
        the absolute position of x's first token. Returns (out, new_past)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin, offset=pos0)
        k = apply_rope(k, self.cos, self.sin, offset=pos0)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        S = k.shape[2]
        if T == S:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # decode step(s): each query at absolute pos p attends keys <= p
            mask = torch.ones(T, S, dtype=torch.bool,
                              device=q.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 4.0):
        super().__init__()
        hidden = int(2 * (mult * d) / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

    def forward_cached(self, x: torch.Tensor, past: tuple | None,
                       pos0: int) -> tuple[torch.Tensor, tuple]:
        # eval-only path (no dropout): used by GPT.forward_cached
        a, kv = self.attn.forward_cached(self.norm1(x), past, pos0)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        # scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.ctx, f"sequence length {T} > ctx {self.cfg.ctx}"
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, idx: torch.Tensor,
                       cache: list | None = None) -> tuple[torch.Tensor, list]:
        """Forward over only the NEW tokens in idx, reusing `cache`
        (per-block (k, v) tuples). Returns (logits, new_cache). Requires
        eval mode semantics (no dropout); callers must keep total length
        <= cfg.ctx since cached keys cannot slide out of the window."""
        B, T = idx.shape
        pos0 = 0 if cache is None else cache[0][0].shape[2]
        assert pos0 + T <= self.cfg.ctx, "cached forward would exceed ctx"
        x = self.tok_emb(idx)
        new_cache = []
        for i, blk in enumerate(self.blocks):
            past = None if cache is None else cache[i]
            x, kv = blk.forward_cached(x, past, pos0)
            new_cache.append(kv)
        x = self.norm_f(x)
        return self.lm_head(x), new_cache

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, eos_id: int | None = None) -> torch.Tensor:
        if idx.shape[1] + max_new_tokens > self.cfg.ctx:
            # prompt + budget overflows the window: fall back to the sliding
            # (cacheless) path, which re-runs the full forward per token
            return self._generate_slow(idx, max_new_tokens, temperature, top_k, eos_id)
        cache = None
        new = idx
        out = idx
        for _ in range(max_new_tokens):
            logits, cache = self.forward_cached(new, cache)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, nxt), dim=1)
            new = nxt
            if eos_id is not None and (nxt == eos_id).all():
                break
        return out

    @torch.no_grad()
    def _generate_slow(self, idx: torch.Tensor, max_new_tokens: int,
                       temperature: float = 0.8, top_k: int | None = 50,
                       eos_id: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.ctx:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
