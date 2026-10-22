"""EyesGrounder: Two-Stage Contrast-Invariant Visual UI Grounder (from scratch, CPU-first).

Stage 1 (`Stage1CoarseNet`):
  - Zero-param local contrast & edge stem (`[R, G, B, signed_norm, |norm|, |dx|, |dy|]`)
    for out-of-distribution theme/palette invariance.
  - Shift-invariant multi-scale 1D CharConv + 2-head attentive text encoder.
  - Anti-aliased FPN (`128 -> 64 -> 32 <-> 16`) + CoordConv + FiLM + 4-head cosine matcher.
  - Predicts 32x32 coarse heatmap, query-conditioned sub-cell offsets, widget box size, and action.

Stage 2 (`Stage2RefineNet`):
  - Extracts native-resolution crops (`crop_frac=80/256` -> `64x64`) around top-K Stage-1 peaks.
  - Verifies/re-ranks candidates by reading crisp 8-14px text/icons at native resolution
    and predicts sub-pixel refined `(x, y)` + `(w, h)` via spatial soft-argmax + residual offset.

Recovery-conditioned grounding (`Deliverable 4`):
  - `ground(png, instruction, exclude_box=None)` masks out candidate cells inside `exclude_box`
    so rung 2 of the drive retry ladder cleanly jumps to the next-best candidate widget.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .ui_world import ACTION_CLASSES, CANVAS as COARSE_SIZE, encode_task


def extract_visual_channels(rgb: torch.Tensor) -> torch.Tensor:
    """Augment normalized RGB `[-1,1]` `(B,3,H,W)` -> `(B,7,H,W)` with polarity-invariant contrast & edges."""
    lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    mean = F.avg_pool2d(lum, kernel_size=5, stride=1, padding=2)
    diff = lum - mean
    var = F.avg_pool2d(diff * diff, kernel_size=5, stride=1, padding=2)
    std = torch.sqrt(var + 1e-4)
    c_signed = (diff / (std + 0.08)).clamp(-3.0, 3.0) / 3.0
    c_abs = c_signed.abs()

    dx = torch.zeros_like(lum)
    dy = torch.zeros_like(lum)
    dx[:, :, :, 1:] = (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()
    dy[:, :, 1:, :] = (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()
    return torch.cat([rgb, c_signed, c_abs, dx, dy], dim=1)


def _coord_grid(batch_size: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(batch_size, 1, h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(batch_size, 1, h, w)
    return torch.cat([xs, ys], dim=1)


class ConvBlockAA(nn.Module):
    """ResNet-D anti-aliased residual block (`stride=2` conv + `AvgPool2d(2)` skip, 4x faster on CPU)."""

    def __init__(self, cin: int, cout: int, stride: int | tuple[int, int] = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        if stride == 2 or (isinstance(stride, tuple) and stride != (1, 1)):
            self.skip = nn.Sequential(nn.AvgPool2d(stride), nn.Conv2d(cin, cout, 1, bias=False))
        elif cin != cout:
            self.skip = nn.Conv2d(cin, cout, 1, bias=False)
        else:
            self.skip = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class CharConvTextEncoder(nn.Module):
    """Shift-invariant multi-scale character n-gram encoder with multi-head attention pooling."""

    def __init__(self, char_vocab: int = 40, d_emb: int = 48, d_out: int = 128, n_attn_heads: int = 2):
        super().__init__()
        self.n_attn_heads = n_attn_heads
        self.emb = nn.Embedding(char_vocab, d_emb, padding_idx=0)
        self.k1 = nn.Conv1d(d_emb, 32, kernel_size=1)
        self.k3 = nn.Conv1d(d_emb, 48, kernel_size=3, padding=1)
        self.k5 = nn.Conv1d(d_emb, 48, kernel_size=5, padding=2)
        c_mid = 32 + 48 + 48  # 128
        self.refine = nn.Sequential(
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1),
        )
        self.attn = nn.Conv1d(c_mid, n_attn_heads, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Linear(c_mid * (n_attn_heads + 1), d_out),
            nn.SiLU(inplace=True),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        # task: (B, L)
        mask = (task != 0).unsqueeze(1).float()  # (B, 1, L)
        e = self.emb(task).transpose(1, 2) * mask  # (B, d_emb, L)
        h = F.silu(torch.cat([self.k1(e), self.k3(e), self.k5(e)], dim=1)) * mask
        h = F.silu(h + self.refine(h)) * mask  # (B, 128, L)

        logits = self.attn(h)  # (B, H, L)
        logits = logits.masked_fill(mask == 0, -1e4)
        weights = F.softmax(logits, dim=-1)  # (B, H, L)
        heads = [(h * weights[:, hi : hi + 1, :]).sum(dim=-1) for hi in range(self.n_attn_heads)]
        h_masked = h.masked_fill(mask == 0, -1e4)
        h_max = h_masked.max(dim=-1).values
        h_max = torch.where(mask.sum(dim=-1) > 0, h_max, torch.zeros_like(h_max))

        return self.proj(torch.cat([*heads, h_max], dim=1))


class Stage1CoarseNet(nn.Module):
    """Coarse full-screen UI grounder (`128x128` -> `32x32` grid + sub-cell offsets + box size + action)."""

    def __init__(self, char_vocab: int = 40, d_text: int = 112, grid: int = 32):
        super().__init__()
        self.char_vocab = char_vocab
        self.d_text = d_text
        self.grid = grid

        self.text = CharConvTextEncoder(char_vocab=char_vocab, d_emb=48, d_out=d_text)
        self.s_64 = ConvBlockAA(7, 32, stride=2)     # 128 -> 64
        self.s_32 = ConvBlockAA(32, 64, stride=2)    # 64 -> 32
        self.s_16 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlockAA(64, 96, stride=1),           # 32 -> 16 context
        )
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(64 + 96 + 2, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )

        # Multiplicative FiLM + spatial shift block (links left-aligned label to widget center)
        self.film = nn.Linear(d_text, 192)
        self.post_film = ConvBlockAA(96, 96, stride=1)

        # 4-head cosine cross-correlation
        self.n_heads = 4
        self.v_proj = nn.Conv2d(96, 96, 1, bias=False)
        self.q_proj = nn.Linear(d_text, 96, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.5))

        self.score_head = nn.Sequential(
            nn.Conv2d(96 + self.n_heads, 48, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 1, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 2, 1),
            nn.Tanh(),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 2, 1),
            nn.Sigmoid(),
        )
        self.act_head = nn.Sequential(
            nn.Linear(96 + d_text, 80),
            nn.SiLU(inplace=True),
            nn.Linear(80, 4),
        )

    def encode_instruction(self, task: torch.Tensor) -> torch.Tensor:
        return self.text(task)

    def encode_vision(self, img: torch.Tensor) -> torch.Tensor:
        ch = extract_visual_channels(img)
        f64 = self.s_64(ch)
        f32 = self.s_32(f64)
        f16 = self.s_16(f32)
        f16_up = F.interpolate(f16, size=f32.shape[-2:], mode="bilinear", align_corners=False)
        coords = _coord_grid(img.shape[0], f32.shape[2], f32.shape[3], img.device, img.dtype)
        return self.fpn_fuse(torch.cat([f32, f16_up, coords], dim=1))  # (B, 96, Gh, Gw)

    def match_features(self, fpn: torch.Tensor, q: torch.Tensor):
        B, C, Gh, Gw = fpn.shape
        gamma, beta = self.film(q).chunk(2, dim=1)
        f_mod = self.post_film(fpn * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None])

        vh = self.v_proj(fpn).view(B, self.n_heads, C // self.n_heads, Gh, Gw)
        qh = self.q_proj(q).view(B, self.n_heads, C // self.n_heads, 1, 1)
        vh = F.normalize(vh, dim=2, eps=1e-6)
        qh = F.normalize(qh, dim=2, eps=1e-6)
        cos_sim = (vh * qh).sum(dim=2) * self.logit_scale.clamp(0.5, 10.0)

        scores = self.score_head(torch.cat([f_mod, cos_sim], dim=1)).flatten(1)
        offsets = self.offset_head(f_mod).flatten(2).transpose(1, 2)
        box_wh = self.box_head(f_mod).flatten(2).transpose(1, 2)
        pooled = F.adaptive_avg_pool2d(f_mod, 1).flatten(1)
        act_logits = self.act_head(torch.cat([pooled, q], dim=1))
        return scores, offsets, box_wh, act_logits

    def forward_with_query(self, img: torch.Tensor, q: torch.Tensor):
        fpn = self.encode_vision(img)
        return self.match_features(fpn, q)

    def forward(self, img: torch.Tensor, task: torch.Tensor):
        q = self.text(task)
        scores, offsets, box_wh, act_logits = self.forward_with_query(img, q)
        return scores, offsets, box_wh, act_logits, q

    def click_xy(self, scores: torch.Tensor, offsets: torch.Tensor, width: float = 256.0, height: float | None = None) -> torch.Tensor:
        if height is None:
            height = width
        B, HW = scores.shape
        g = int(round(HW ** 0.5))
        cell = scores.argmax(dim=1)
        cy = (cell // g).float()
        cx = (cell % g).float()
        off = offsets[torch.arange(B, device=scores.device), cell]
        x = (cx + 0.5 + 0.45 * off[:, 0]) * (width / g)
        y = (cy + 0.5 + 0.45 * off[:, 1]) * (height / g)
        return torch.stack([x.clamp(0, width - 1), y.clamp(0, height - 1)], dim=1)

    def loss(
        self,
        scores: torch.Tensor,
        offsets: torch.Tensor,
        box_wh: torch.Tensor,
        act_logits: torch.Tensor,
        target_hm: torch.Tensor,
        target_off: torch.Tensor,
        action: torch.Tensor,
        target_wh: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        hm_flat = target_hm.flatten(1).float()
        hm_flat = hm_flat / hm_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(scores, dim=1)
        soft = -(hm_flat * logp).sum(dim=1).mean()
        cell = hm_flat.argmax(dim=1).long()
        hard = F.cross_entropy(scores, cell)
        ce_loss = 0.5 * soft + 0.5 * hard

        idx = torch.arange(offsets.shape[0], device=offsets.device)
        off_loss = F.smooth_l1_loss(offsets[idx, cell], target_off, beta=0.15)
        act_loss = F.cross_entropy(act_logits, action)
        wh_loss = torch.tensor(0.0, device=offsets.device)
        if target_wh is not None:
            wh_loss = F.smooth_l1_loss(box_wh[idx, cell], target_wh, beta=0.05)

        total = ce_loss + 0.8 * off_loss + 0.4 * wh_loss + 0.2 * act_loss
        return total, {
            "ce": float(ce_loss.item()),
            "off": float(off_loss.item()),
            "wh": float(wh_loss.item()),
            "act": float(act_loss.item()),
        }


class Stage2RefineNet(nn.Module):
    """Candidate-pinned 32x160 horizontal CRNN (`CNN + Vertical-Glyph-Collapse + BiGRU`) verifier & pointer."""

    CROP_H = 32
    CROP_W = 160
    ANCHOR_FRAC_X = 0.70         # Candidate cx is ALWAYS pinned at col 112 ([-112px .. +48px] covers wide & centered!)
    SPAN_H_FRAC = 32.0 / 256.0   # 1:1 native vertical pixels on 256x256
    SPAN_W_FRAC = 160.0 / 256.0  # 1:1 native horizontal pixels on 256x256
    MAX_DX_PX = 14.0
    MAX_DY_PX = 12.0

    def __init__(self, d_text: int = 112):
        super().__init__()
        self.d_text = d_text
        self.text_enc = CharConvTextEncoder(d_out=d_text)

        # 7 contrast/RGB channels + 3 candidate-anchor channels (u, v, beacon) -> 10 channels
        # Stride=1 native stroke stem preserves 1-px TrueType character strokes!
        self.stem1 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.SiLU(inplace=True),
        )
        self.c16 = ConvBlockAA(20, 40, stride=2)   # 32x160 -> 16x80
        self.c8 = ConvBlockAA(40, 72, stride=2)    # 16x80  -> 8x40
        self.c8_ctx = ConvBlockAA(72, 72, stride=1)

        # Full-height (8x1) vertical glyph+border collapse -> 40-column horizontal sequence!
        self.v_collapse = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=(8, 1), bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        # Candidate-relative column positional embedding (col 28 == candidate cx!)
        self.col_emb = nn.Parameter(torch.randn(1, 96, 40) * 0.02)
        # Horizontal BiGRU reads sequential characters AND propagates left-aligned labels up to widget borders!
        self.bigru = nn.GRU(input_size=96, hidden_size=80, num_layers=1, batch_first=True, bidirectional=True)
        d_seq = 160

        self.n_heads = 4
        self.head_dim = 20  # 4 * 20 = 80
        self.k_proj = nn.Linear(d_seq, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_seq, d_text, bias=False)
        self.q_proj = nn.Linear(d_text, self.n_heads * self.head_dim, bias=False)
        self.attn_temp = nn.Parameter(torch.tensor(4.5))
        self.col_attn_bias = nn.Parameter(torch.zeros(1, 40))

        # Anchor descriptor right around col 28 (cols 25..31 = cx +- 12px)
        self.anc_proj = nn.Sequential(
            nn.Linear(d_seq * 2, d_text),
            nn.SiLU(inplace=True),
            nn.LayerNorm(d_text),
        )

        # Pairwise verification MLP takes [anc_vec, v_attn, q, anc_vec*q, v_attn*q, head_stats (8)]
        self.verify_head = nn.Sequential(
            nn.Linear(d_text * 5 + self.n_heads * 2, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.sub_off = nn.Sequential(
            nn.Linear(d_text * 3, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),  # (delta_u, delta_v, w_frac, h_frac)
        )

    def encode_instruction(self, task: torch.Tensor) -> torch.Tensor:
        return self.text_enc(task)

    def encode_crops(self, crops: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode `(K, 6, 32, 160)` candidate-pinned strips ONCE per screen."""
        K = crops.shape[0]
        rgb = crops[:, :3]
        anc = crops[:, 3:6]
        ch = extract_visual_channels(rgb)
        x = torch.cat([ch, anc], dim=1)  # (K, 10, 32, 160)

        f8 = self.c8_ctx(self.c8(self.c16(self.stem1(x))))  # (K, 72, 8, 40)
        seq = (self.v_collapse(f8).squeeze(2) + self.col_emb).transpose(1, 2)  # (K, 40, 96)
        h_seq, _ = self.bigru(seq)  # (K, 40, 160)

        anc_mean = h_seq[:, 25:32, :].mean(dim=1)
        anc_fovea = h_seq[:, 21:35, :].max(dim=1).values
        anc_vec = self.anc_proj(torch.cat([anc_mean, anc_fovea], dim=1))  # (K, d_text)

        k_map = F.normalize(
            self.k_proj(h_seq).view(K, 40, self.n_heads, self.head_dim).permute(0, 2, 3, 1),
            dim=2,
            eps=1e-6,
        )  # (K, H, Dh, 40)
        v_map = self.v_proj(h_seq).transpose(1, 2).contiguous()  # (K, d_text, 40)
        return {"k_map": k_map, "v_map": v_map, "anc_vec": anc_vec}

    def score_matrix(self, crop_repr: dict[str, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        """Compute `(Q, K)` verification logits matching every instruction `q[i]` against every crop `k`."""
        k_map = crop_repr["k_map"]      # (K, H, Dh, 40)
        v_map = crop_repr["v_map"]      # (K, D, 40)
        anc_vec = crop_repr["anc_vec"]  # (K, D)

        Q = q.shape[0]
        K = k_map.shape[0]
        q_heads = F.normalize(self.q_proj(q).view(Q, self.n_heads, self.head_dim), dim=2, eps=1e-6)
        sim_h = torch.einsum("qhd,khdp->qkhp", q_heads, k_map)  # (Q, K, H, 40)
        temp = self.attn_temp.clamp(1.0, 12.0)
        attn = F.softmax(sim_h.mean(dim=2) * temp + self.col_attn_bias[None, :, :], dim=-1)
        v_attn = torch.einsum("qkp,kdp->qkd", attn, v_map)      # (Q, K, D)

        sim_max = sim_h.max(dim=-1).values                       # (Q, K, H)
        sim_anc = sim_h[:, :, :, 24:33].mean(dim=-1)             # (Q, K, H)

        q_exp = q[:, None, :].expand(Q, K, -1)
        a_exp = anc_vec[None, :, :].expand(Q, K, -1)
        cat = torch.cat([a_exp, v_attn, q_exp, a_exp * q_exp, v_attn * q_exp, sim_max, sim_anc], dim=-1)
        return self.verify_head(cat).squeeze(-1)  # (Q, K)

    def encode_and_offset(
        self, crops: torch.Tensor, q: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Single-pass strip encoding + diagonal offset/wh prediction for fast matrix training."""
        crop_repr = self.encode_crops(crops)
        k_map = crop_repr["k_map"]
        v_map = crop_repr["v_map"]
        anc_vec = crop_repr["anc_vec"]
        q_heads = F.normalize(self.q_proj(q).view(crops.shape[0], self.n_heads, self.head_dim), dim=2, eps=1e-6)
        sim_h = torch.einsum("khd,khdp->khp", q_heads, k_map)
        temp = self.attn_temp.clamp(1.0, 12.0)
        attn = F.softmax(sim_h.mean(dim=1) * temp + self.col_attn_bias, dim=-1)
        v_attn = torch.einsum("kp,kdp->kd", attn, v_map)
        sub = self.sub_off(torch.cat([anc_vec, v_attn, q], dim=1))
        return crop_repr, torch.tanh(sub[:, :2]), torch.sigmoid(sub[:, 2:4])

    def forward(
        self, crops: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score paired `(K, 6, 32, 160)` candidate strips and `(K, d_text)` queries + predict residual offsets."""
        crop_repr = self.encode_crops(crops)
        k_map = crop_repr["k_map"]
        v_map = crop_repr["v_map"]
        anc_vec = crop_repr["anc_vec"]

        K = crops.shape[0]
        q_heads = F.normalize(self.q_proj(q).view(K, self.n_heads, self.head_dim), dim=2, eps=1e-6)
        sim_h = torch.einsum("khd,khdp->khp", q_heads, k_map)
        temp = self.attn_temp.clamp(1.0, 12.0)
        attn = F.softmax(sim_h.mean(dim=1) * temp + self.col_attn_bias, dim=-1)
        v_attn = torch.einsum("kp,kdp->kd", attn, v_map)

        sim_max = sim_h.max(dim=-1).values
        sim_anc = sim_h[:, :, 24:33].mean(dim=-1)

        cat = torch.cat([anc_vec, v_attn, q, anc_vec * q, v_attn * q, sim_max, sim_anc], dim=-1)
        verify_logit = self.verify_head(cat).squeeze(-1)

        sub = self.sub_off(torch.cat([anc_vec, v_attn, q], dim=1))
        delta_uv = torch.tanh(sub[:, :2])
        fine_wh = torch.sigmoid(sub[:, 2:4])
        return verify_logit, delta_uv, fine_wh


def extract_crop_tensor(
    img_rgb_uint8: np.ndarray,
    cx: float,
    cy: float,
    span_w: float,
    span_h: float,
    out_h: int = Stage2RefineNet.CROP_H,
    out_w: int = Stage2RefineNet.CROP_W,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Extract candidate-pinned `32x160` strip (`cx` ALWAYS at `0.70*span_w`, border-replicated at screen edges)."""
    H, W = img_rgb_uint8.shape[:2]
    x0 = float(cx) - Stage2RefineNet.ANCHOR_FRAC_X * float(span_w)
    y0 = float(cy) - 0.50 * float(span_h)
    rw = float(span_w)
    rh = float(span_h)

    xs = x0 + (np.arange(out_w, dtype=np.float32) + 0.5) * (rw / out_w)
    ys = y0 + (np.arange(out_h, dtype=np.float32) + 0.5) * (rh / out_h)
    xi = np.clip(np.floor(xs).astype(np.int64), 0, W - 1)
    yi = np.clip(np.floor(ys).astype(np.int64), 0, H - 1)
    rgb_f = img_rgb_uint8[np.ix_(yi, xi)].astype(np.float32) / 127.5 - 1.0

    u = ((xs - float(cx)) / max(1.0, 0.5 * rw))[None, :]
    v = ((ys - float(cy)) / max(1.0, 0.5 * rh))[:, None]
    u_map = np.broadcast_to(u, (out_h, out_w))
    v_map = np.broadcast_to(v, (out_h, out_w))
    gx = (xs - float(cx)) / max(2.0, 0.14 * rw)
    gy = (ys - float(cy)) / max(2.0, 0.35 * rh)
    beacon = np.exp(-(gx[None, :] ** 2) - (gy[:, None] ** 2)).astype(np.float32)

    arr6 = np.concatenate([rgb_f, u_map[:, :, None], v_map[:, :, None], beacon[:, :, None]], axis=2)
    return arr6, (x0, y0, rw, rh)


def _normalize_exclude_box(exclude_box: Any) -> tuple[float, float, float, float] | None:
    if exclude_box is None:
        return None
    if isinstance(exclude_box, dict):
        return (
            float(exclude_box["x"]),
            float(exclude_box["y"]),
            float(exclude_box["w"]),
            float(exclude_box["h"]),
        )
    if len(exclude_box) == 4:
        return (float(exclude_box[0]), float(exclude_box[1]), float(exclude_box[2]), float(exclude_box[3]))
    raise ValueError(f"Unsupported exclude_box: {exclude_box}")


def topk_grid_candidates(
    scores_1d: torch.Tensor,
    offsets_2d: torch.Tensor,
    box_wh_2d: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    k: int = 7,
    exclude_box: Any = None,
    nms_radius_cells: int = 1,
) -> list[dict[str, Any]]:
    """Extract top-K spatially distinct candidate peaks from a single image's coarse grid."""
    hw = scores_1d.shape[0]
    g = int(round(hw ** 0.5))
    logp = F.log_softmax(scores_1d, dim=0).detach().cpu().numpy().copy()
    offs = offsets_2d.detach().cpu().numpy()
    bwh = box_wh_2d.detach().cpu().numpy()

    ex = _normalize_exclude_box(exclude_box)
    cell_w = canvas_w / g
    cell_h = canvas_h / g

    if ex is not None:
        ex_x, ex_y, ex_w, ex_h = ex
        for c_idx in range(hw):
            gy, gx = divmod(c_idx, g)
            px = (gx + 0.5 + 0.45 * offs[c_idx, 0]) * cell_w
            py = (gy + 0.5 + 0.45 * offs[c_idx, 1]) * cell_h
            if ex_x <= px <= ex_x + ex_w and ex_y <= py <= ex_y + ex_h:
                logp[c_idx] = -1e9

    cands: list[dict[str, Any]] = []
    work = logp.reshape(g, g).copy()
    for rank in range(k):
        flat_idx = int(work.argmax())
        best_val = float(work.flat[flat_idx])
        if best_val <= -1e8 and cands:
            break
        gy, gx = divmod(flat_idx, g)
        px = float(np.clip((gx + 0.5 + 0.45 * offs[flat_idx, 0]) * cell_w, 0.0, canvas_w - 1.0))
        py = float(np.clip((gy + 0.5 + 0.45 * offs[flat_idx, 1]) * cell_h, 0.0, canvas_h - 1.0))
        est_w = float(np.clip(bwh[flat_idx, 0] * canvas_w, 12.0, canvas_w * 0.6))
        est_h = float(np.clip(bwh[flat_idx, 1] * canvas_h, 10.0, canvas_h * 0.4))
        cands.append({
            "rank": rank,
            "cell": flat_idx,
            "stage1_logp": best_val,
            "x": px,
            "y": py,
            "w": est_w,
            "h": est_h,
        })
        y0, y1 = max(0, gy - nms_radius_cells), min(g, gy + nms_radius_cells + 1)
        x0, x1 = max(0, gx - nms_radius_cells), min(g, gx + nms_radius_cells + 1)
        work[y0:y1, x0:x1] = -1e9

    return cands


class EyesGrounder:
    """Unified runtime grounder supporting Stage-1, Stage-2 strip refiner, multi-scale, and recovery `exclude_box`."""

    def __init__(
        self,
        stage1: Stage1CoarseNet,
        stage2: Stage2RefineNet | None = None,
        top_k: int = 7,
        nms_radius_cells: int = 1,
        stage1_weight: float | None = None,
        stage2_weight: float = 1.0,
    ):
        self.stage1 = stage1.eval()
        self.stage2 = stage2.eval() if stage2 is not None else None
        self.top_k = top_k
        self.nms_radius_cells = nms_radius_cells
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight

    @staticmethod
    def _to_rgb_array(png_or_img: bytes | str | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(png_or_img, np.ndarray):
            return png_or_img
        if isinstance(png_or_img, Image.Image):
            return np.asarray(png_or_img.convert("RGB"), dtype=np.uint8)
        if isinstance(png_or_img, (bytes, bytearray)):
            return np.asarray(Image.open(io.BytesIO(png_or_img)).convert("RGB"), dtype=np.uint8)
        if isinstance(png_or_img, str):
            return np.asarray(Image.open(png_or_img).convert("RGB"), dtype=np.uint8)
        raise TypeError(f"Unsupported image input: {type(png_or_img)}")

    def ground(
        self,
        png: bytes | str | Image.Image | np.ndarray,
        instruction: str,
        exclude_box: Any = None,
        use_stage2: bool = True,
        multi_scale: bool = False,
    ) -> dict[str, Any]:
        rgb = self._to_rgb_array(png)
        H, W = rgb.shape[:2]
        ex = _normalize_exclude_box(exclude_box)

        with torch.no_grad():
            t = torch.tensor([encode_task(instruction)], dtype=torch.long)
            q1 = self.stage1.encode_instruction(t)

            im_pil = Image.fromarray(rgb)
            im128 = np.asarray(im_pil.resize((COARSE_SIZE, COARSE_SIZE), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
            x128 = torch.from_numpy(im128).permute(2, 0, 1).unsqueeze(0)
            s128, o128, wh128, act_logits = self.stage1.forward_with_query(x128, q1)

            cands = topk_grid_candidates(
                s128[0],
                o128[0],
                wh128[0],
                canvas_w=W,
                canvas_h=H,
                k=self.top_k,
                exclude_box=ex,
                nms_radius_cells=self.nms_radius_cells,
            )

            if multi_scale:
                im160 = np.asarray(im_pil.resize((160, 160), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
                x160 = torch.from_numpy(im160).permute(2, 0, 1).unsqueeze(0)
                s160, o160, wh160, act160 = self.stage1.forward_with_query(x160, q1)
                act_logits = 0.5 * (act_logits + act160)
                extra = topk_grid_candidates(
                    s160[0],
                    o160[0],
                    wh160[0],
                    canvas_w=W,
                    canvas_h=H,
                    k=3,
                    exclude_box=ex,
                    nms_radius_cells=self.nms_radius_cells,
                )
                min_dist = max(7.0, 0.03 * max(W, H))
                for ec in extra:
                    if all(((ec["x"] - c["x"]) ** 2 + (ec["y"] - c["y"]) ** 2) ** 0.5 > min_dist for c in cands):
                        cands.append(ec)

            act_idx = int(act_logits[0].argmax().item())
            action = ACTION_CLASSES[act_idx]

            best = cands[0]
            px, py = best["x"], best["y"]
            bw, bh = best["w"], best["h"]
            conf = float(np.exp(np.clip(best["stage1_logp"], -20.0, 0.0)))

            if use_stage2 and self.stage2 is not None and len(cands) > 0:
                span_w = max(96.0, Stage2RefineNet.SPAN_W_FRAC * W)
                span_h = max(24.0, Stage2RefineNet.SPAN_H_FRAC * H)
                crop_arrs = []
                crop_rects = []
                for c in cands:
                    arr, rect = extract_crop_tensor(rgb, c["x"], c["y"], span_w, span_h)
                    crop_arrs.append(arr)
                    crop_rects.append(rect)
                crop_batch = torch.from_numpy(np.stack(crop_arrs, axis=0)).permute(0, 3, 1, 2)
                q2 = self.stage2.encode_instruction(t)
                q2_rep = q2.expand(len(cands), -1)
                v_logits, delta_uv, f_wh = self.stage2(crop_batch, q2_rep)

                best_score = -1e9
                for idx_c, c in enumerate(cands):
                    _, _, rw, rh = crop_rects[idx_c]
                    max_dx = Stage2RefineNet.MAX_DX_PX * (rw / float(Stage2RefineNet.CROP_W))
                    max_dy = Stage2RefineNet.MAX_DY_PX * (rh / float(Stage2RefineNet.CROP_H))
                    fx = float(np.clip(c["x"] + max_dx * float(delta_uv[idx_c, 0].item()), 0.0, W - 1.0))
                    fy = float(np.clip(c["y"] + max_dy * float(delta_uv[idx_c, 1].item()), 0.0, H - 1.0))

                    # Enforce exclude_box strictly even after Stage-2 refinement shift!
                    if ex is not None and (ex[0] <= fx <= ex[0] + ex[2] and ex[1] <= fy <= ex[1] + ex[3]):
                        continue

                    s1_w = (
                        float(self.stage1_weight)
                        if self.stage1_weight is not None
                        else float(getattr(self.stage2, "stage1_alpha", torch.tensor(0.45)).item())
                    )
                    comb = s1_w * c["stage1_logp"] + self.stage2_weight * float(v_logits[idx_c].item())
                    if comb > best_score:
                        best_score = comb
                        px, py = fx, fy
                        bw = float(np.clip(float(f_wh[idx_c, 0].item()) * W, 16.0, W * 0.92))
                        bh = float(np.clip(float(f_wh[idx_c, 1].item()) * H, 12.0, H * 0.25))
                        conf = float(1.0 / (1.0 + np.exp(-np.clip(float(v_logits[idx_c].item()), -15.0, 15.0))))

        ix = int(round(np.clip(px, 0, W - 1)))
        iy = int(round(np.clip(py, 0, H - 1)))
        bx = int(round(np.clip(px - bw / 2.0, 0, W - 1)))
        by = int(round(np.clip(py - bh / 2.0, 0, H - 1)))
        return {
            "x": ix,
            "y": iy,
            "fx": round(float(px), 2),
            "fy": round(float(py), 2),
            "action": action,
            "action_id": act_idx,
            "confidence": round(conf, 4),
            "box": [bx, by, int(round(bw)), int(round(bh))],
        }

