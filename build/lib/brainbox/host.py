"""BrainHost: owns the loaded model(s) and serves generation/scoring.

One model execution at a time (a lock serialises callers; the M1 HTTP layer
queues in front of this). Every entry point checks the RAM governor first.
"""

from __future__ import annotations

import os
import threading
import time

from . import checkpoints as ckmod
from .governor import Governor, rss_mb


class BrainHost:
    def __init__(self, checkpoint: str | None = None, threads: int = 2,
                 governor: Governor | None = None,
                 tokenizer_path: str | None = None, stub_seed: int = 0,
                 grounder=None, grip_checkpoint: str | None = None):
        self.checkpoint_arg = checkpoint
        self.threads = threads
        self.governor = governor or Governor()
        self.tokenizer_path = tokenizer_path
        self.stub_seed = stub_seed
        self.grounder = grounder
        self.grip_checkpoint_arg = grip_checkpoint
        self.model = None
        self.tok = None
        self.info: ckmod.CheckpointInfo | None = None
        self.boot_s: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def boot(self) -> dict:
        import torch

        t0 = time.time()
        torch.set_num_threads(self.threads)
        torch.set_grad_enabled(False)
        tok_path = self.tokenizer_path or ckmod.default_tokenizer_path()

        if self.checkpoint_arg:
            path = self.checkpoint_arg
            if os.path.isdir(path):
                self.model, self.tok, self.info = ckmod.load_checkpoint_dir(path)
            elif os.path.exists(path):
                if not tok_path:
                    raise ckmod.CheckpointError("no tokenizer.json found")
                self.model, self.tok, self.info = ckmod.load_legacy_pt(path, tok_path)
            else:
                raise ckmod.CheckpointError(f"checkpoint not found: {path}")
        else:
            found = ckmod.find_checkpoint()
            if found and found[1] == "dir":
                self.model, self.tok, self.info = ckmod.load_checkpoint_dir(found[0])
            elif found and found[1] == "legacy":
                if not tok_path:
                    raise ckmod.CheckpointError("no tokenizer.json found")
                self.model, self.tok, self.info = ckmod.load_legacy_pt(found[0], tok_path)
            else:
                # Nothing trained is published yet: in-memory stub, real arch +
                # real tokenizer, honestly flagged. Never blocks on other agents.
                if not tok_path:
                    raise ckmod.CheckpointError(
                        "no checkpoint and no tokenizer.json found")
                from neurobrain.tokenizer import BPETokenizer

                self.tok = BPETokenizer.load(tok_path)
                self.model = ckmod.build_stub_model(self.tok, seed=self.stub_seed)
                self.info = ckmod.CheckpointInfo(
                    path="(in-memory stub)", step=0, stub=True,
                    params=self.model.num_params(),
                    config=self.model.cfg.to_dict())

        self.boot_s = time.time()
        return self.describe()

    def describe(self) -> dict:
        assert self.info is not None
        d = {**self.info.to_dict(), "boot_s": self.boot_s,
             "threads": self.threads,
             "vocab_size": self.tok.vocab_size if self.tok else None}
        if self.grounder is not None:
            d["grip"] = self.grounder.describe()
        return d

    def boot_grip(self) -> dict:
        """Load the GripNet grounder (also lazy from ground())."""
        if self.grounder is None:
            from .grip import Grounder

            self.grounder = Grounder.find(self.grip_checkpoint_arg)
        return self.grounder.describe()

    def _ready(self):
        if self.model is None:
            self.boot()
        self.governor.check_or_raise()

    # ------------------------------------------------------------ inference
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.8,
                 top_k: int = 40, seed: int | None = None) -> dict:
        import torch

        with self._lock:
            self._ready()
            if seed is not None:
                torch.manual_seed(seed)
            ids = self.tok.encode(prompt) or self.tok.encode("\n")
            x = torch.tensor([ids], dtype=torch.long)
            t0 = time.time()
            out = self.model.generate(x, max_new_tokens=max_tokens,
                                      temperature=temperature, top_k=top_k)
            ms = int((time.time() - t0) * 1000)
            full = out[0].tolist()
            return {"text": self.tok.decode(full),
                    "completion": self.tok.decode(full[len(ids):]),
                    "tokens": len(full) - len(ids), "ms": ms}

    def score(self, prompt: str, top_k: int = 5) -> dict:
        """Top-k next-token logprobs for the prompt. Spec: {tokens, logprobs}."""
        import torch
        import torch.nn.functional as F

        with self._lock:
            self._ready()
            ids = self.tok.encode(prompt) or self.tok.encode("\n")
            x = torch.tensor([ids[-self.model.cfg.ctx:]], dtype=torch.long)
            with torch.no_grad():  # worker thread: boot's set_grad_enabled is main-thread only
                logits, _ = self.model(x)
            logp = F.log_softmax(logits[0, -1, :].float(), dim=-1)
            vals, idxs = torch.topk(logp, min(top_k, logp.shape[-1]))
            return {"tokens": [self.tok.decode([int(i)]) for i in idxs],
                    "token_ids": [int(i) for i in idxs],
                    "logprobs": [float(v) for v in vals]}

    def ground(self, png_bytes: bytes, instruction: str) -> dict:
        """Spec shape {"x","y","action"} in the input image's pixels."""
        with self._lock:
            self._ready()
            if self.grounder is None:
                self.boot_grip()
            return self.grounder.ground(png_bytes, instruction)

    def health(self) -> dict:
        return {"ok": self.model is not None, "rss_mb": int(rss_mb()),
                "step": self.info.step if self.info else -1}
