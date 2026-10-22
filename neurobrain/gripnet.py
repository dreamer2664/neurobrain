"""GripNet: a small from-scratch CNN that looks at a screen and points at things.

Image encoder (4 strided conv blocks) -> 8x8 feature grid. The instruction is
encoded character-by-character and used as a query against every grid cell, so
the network learns instruction-conditioned visual search, not a fixed map. The
soft-argmax over the resulting heatmap is the click location; a pooled head
also predicts the action type (click / type / toggle / drag).
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
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        act_logits = self.act_head(torch.cat([pooled, q.squeeze(1)], dim=1))
        return scores, act_logits

    def click_xy(self, scores: torch.Tensor, size: int = 128) -> torch.Tensor:
        """Soft-argmax click location in pixels."""
        B = scores.shape[0]
        hm = scores.view(B, 1, self.grid, self.grid)
        hm = F.softmax(hm.view(B, -1), dim=1).view(B, 1, self.grid, self.grid)
        up = F.interpolate(hm, size=(size, size), mode="bilinear", align_corners=False)
        up = up.view(B, -1)
        idx = up.argmax(dim=1)
        y = (idx // size).float()
        x = (idx % size).float()
        return torch.stack([x, y], dim=1)

    def loss(self, scores, act_logits, target_hm, action, kl_weight: float = 1.0):
        logp = F.log_softmax(scores, dim=1)
        soft = -(target_hm * logp).sum(1).mean()
        hard = F.cross_entropy(scores, target_hm.argmax(1).long())
        act = F.cross_entropy(act_logits, action)
        return kl_weight * (0.5 * soft + 0.5 * hard) + 0.2 * act, {
            "ce": (0.5 * soft + 0.5 * hard).item(),
            "act": act.item(),
        }
