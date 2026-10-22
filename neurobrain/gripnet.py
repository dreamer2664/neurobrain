"""GripNet: a small from-scratch CNN that looks at a screen and points at things.

Architecture (coarse-to-fine pointing):

    image 128x128 --> conv encoder --> 8x8 feature grid
    instruction     --> char encoder --> query vector q

    coarse head:  q . features  -> 8x8 attention heatmap   (which cell)
    offset head:  conv(features) -> 8x8 x 2               (where inside the cell)

The coarse grid gives instruction-conditioned visual search; the sub-cell
offsets recover pixel-level precision that a pure 8x8 heatmap cannot reach.
A pooled head also predicts the action type (click / type / toggle / drag).
Everything is random-initialised: no pretrained vision backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class GripNet(nn.Module):
    def __init__(self, char_vocab: int = 40, max_len: int = 48, d: int = 64, grid: int = 8):
        super().__init__()
        self.grid = grid
        self.vision = nn.Sequential(
            ConvBlock(3, 32),     # 128 -> 64
            ConvBlock(32, 64),    # 64 -> 32
            ConvBlock(64, 96),    # 32 -> 16
            ConvBlock(96, 128),   # 16 -> 8
        )
        self.text = nn.Sequential(
            nn.Embedding(char_vocab, 32, padding_idx=0),
            nn.Flatten(start_dim=1),
            nn.Linear(max_len * 32, 96),
            nn.SiLU(),
            nn.Linear(96, d),
            nn.LayerNorm(d),
        )
        self.proj = nn.Conv2d(128, d, 1)
        self.offset_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 2, 1), nn.Tanh()
        )
        self.act_head = nn.Sequential(nn.Linear(128 + d, 96), nn.SiLU(), nn.Linear(96, 4))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, img: torch.Tensor, task: torch.Tensor):
        f = self.vision(img)                       # (B,128,8,8)
        feat = self.proj(f)                        # (B,d,8,8)
        B, d, H, W = feat.shape
        q = self.text(task).unsqueeze(1)           # (B,1,d)
        keys = feat.flatten(2).transpose(1, 2)     # (B,HW,d)
        scores = torch.bmm(keys, q.transpose(1, 2)).squeeze(-1) / (d ** 0.5)
        scores = scores * self.temperature.clamp(0.25, 4.0)
        pool = F.adaptive_avg_pool2d(f, 1).flatten(1)
        act_logits = self.act_head(torch.cat([pool, q.squeeze(1)], dim=1))
        offsets = self.offset_head(f).flatten(2).transpose(1, 2)  # (B, HW, 2) in [-1,1]
        return scores, offsets, act_logits

    def click_xy(self, scores: torch.Tensor, offsets: torch.Tensor, size: int = 128) -> torch.Tensor:
        """argmax cell + learned sub-cell offset -> pixel coordinates."""
        B = scores.shape[0]
        cell = scores.argmax(dim=1)
        cell_size = size / self.grid
        cy = (cell // self.grid).float()
        cx = (cell % self.grid).float()
        off = offsets[torch.arange(B), cell]           # (B,2)
        x = (cx + 0.5 + 0.45 * off[:, 0]) * cell_size
        y = (cy + 0.5 + 0.45 * off[:, 1]) * cell_size
        return torch.stack([x.clamp(0, size - 1), y.clamp(0, size - 1)], dim=1)

    def loss(self, scores, offsets, act_logits, target_hm, target_off, action,
             kl_weight: float = 1.0):
        logp = F.log_softmax(scores, dim=1)
        soft = -(target_hm * logp).sum(1).mean()
        hard = F.cross_entropy(scores, target_hm.argmax(1).long())
        cell = target_hm.argmax(1).long()
        off_pred = offsets[torch.arange(offsets.shape[0]), cell]
        off_loss = F.l1_loss(off_pred, target_off)
        act = F.cross_entropy(act_logits, action)
        total = kl_weight * (0.5 * soft + 0.5 * hard) + 0.5 * off_loss + 0.2 * act
        return total, {
            "ce": (0.5 * soft + 0.5 * hard).item(),
            "off": off_loss.item(),
            "act": act.item(),
        }
