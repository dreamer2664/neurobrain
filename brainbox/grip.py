"""GripNet grounder hosting: screenshot + instruction -> click point.

Preprocessing matches training exactly (neurobrain/ui_world.py: 128x128,
pixels /127.5 - 1, `encode_task`) and the reference loop in
scripts/e2e_demo.py. Coordinates are returned in the *input image's* pixel
space so the caller can act on the screenshot it sent.
"""

from __future__ import annotations

import io
import os
import time

from . import checkpoints as ckmod


class Grounder:
    def __init__(self, model=None, step: int = -1, source: str = "", stub: bool = False):
        self.model = model
        self.step = step
        self.source = source
        self.stub = stub

    # ------------------------------------------------------------------ load
    @classmethod
    def from_legacy(cls, path: str) -> "Grounder":
        import torch

        from neurobrain.gripnet import GripNet

        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = GripNet()  # defaults are the trained config (specs/model_io.md)
        model.load_state_dict(ck["model"])
        model.eval()
        return cls(model=model, step=int(ck.get("step", -1)), source=path)

    @classmethod
    def from_dir(cls, path: str) -> "Grounder":
        import torch

        from neurobrain.gripnet import GripNet

        cfg = ckmod.read_config(path)
        if cfg.get("kind") != "grip":
            raise ckmod.CheckpointError(f"{path} is not a grip checkpoint")
        weights = os.path.join(path, "weights.pt")
        if cfg.get("weights_sha256") and ckmod.sha256_file(weights) != cfg["weights_sha256"]:
            raise ckmod.CheckpointError(f"weights sha256 mismatch in {path}")
        model = GripNet(char_vocab=cfg.get("char_vocab", 40),
                        max_len=cfg.get("max_len", 48),
                        d=cfg.get("d", 64), grid=cfg.get("grid", 32))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        return cls(model=model, step=int(cfg.get("step", -1)), source=path,
                   stub=bool(cfg.get("stub", False)))

    @classmethod
    def find(cls, explicit: str | None = None) -> "Grounder":
        path = explicit or os.environ.get("BRAINBOX_GRIP_CHECKPOINT")
        if path:
            if os.path.isdir(path):
                return cls.from_dir(path)
            if os.path.exists(path):
                return cls.from_legacy(path)
            raise ckmod.CheckpointError(f"grip checkpoint not found: {path}")
        roots = [os.path.join(os.getcwd(), "checkpoints"),
                 os.path.join(ckmod.repo_root(), "checkpoints"),
                 os.path.expanduser("~/.brainbox/checkpoints")]
        for root in roots:
            d = os.path.join(root, "grip")
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                return cls.from_dir(d)
            pt = os.path.join(root, "grip_best.pt")
            if os.path.exists(pt):
                return cls.from_legacy(pt)
        raise ckmod.CheckpointError(
            "no grip checkpoint found (checkpoints/grip/ or grip_best.pt)")

    def describe(self) -> dict:
        return {"step": self.step, "source": self.source, "stub": self.stub,
                "params": sum(p.numel() for p in self.model.parameters())
                if self.model is not None else 0}

    # ------------------------------------------------------------------ run
    def ground(self, png_bytes: bytes, instruction: str) -> dict:
        """Spec shape: {"x": int, "y": int, "action": str} (image pixels)."""
        full = self.ground_full(png_bytes, instruction)
        return {"x": full["x"], "y": full["y"], "action": full["action"]}

    def ground_full(self, png_bytes: bytes, instruction: str) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        from neurobrain.ui_world import ACTION_CLASSES, CANVAS, encode_task

        if self.model is None:
            raise ckmod.CheckpointError("grounder not loaded")
        self.model.eval()  # BatchNorm: never depend on the caller's mode
        t0 = time.time()
        try:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"invalid image: {e}") from e
        w, h = img.size
        small = img.resize((CANVAS, CANVAS), Image.BILINEAR)
        x = np.asarray(small, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None]
        task = torch.tensor([encode_task(instruction)], dtype=torch.long)
        with torch.no_grad():
            scores, offsets, act_logits = self.model(x, task)
            px, py = self.model.click_xy(scores, offsets, CANVAS)[0].tolist()
            action = ACTION_CLASSES[int(act_logits.argmax(dim=1)[0])]
        return {"x": min(max(int(round(px / CANVAS * w)), 0), w - 1),
                "y": min(max(int(round(py / CANVAS * h)), 0), h - 1),
                "action": action, "x128": float(px), "y128": float(py),
                "w": w, "h": h, "step": self.step,
                "ms": int((time.time() - t0) * 1000)}
