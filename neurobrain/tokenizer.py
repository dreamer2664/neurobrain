"""Byte-level BPE tokenizer, implemented from scratch (no external tokenizer libs).

The vocabulary starts as the 256 raw byte values, then merges are learned by
counting adjacent symbol pairs over pre-tokenized text. Encoding applies the
learned merges greedily by rank, and caches results per pre-token, which makes
it fast enough to tokenize tens of megabytes on a couple of CPU cores.

Special tokens (``<|endoftext|>``, ``<|code|>``, ``<|ui|>`` ...) live above the
BPE range and are split off before BPE is applied.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Iterable

# GPT-2 style pre-tokenization: words, numbers, punctuation runs, whitespace runs.
PRETOKEN_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|code|>", "<|ui|>", "<|pad|>"]


class BPETokenizer:
    def __init__(self, vocab: list[bytes], merges: list[tuple[int, int]],
                 specials: list[str] | None = None):
        self.specials = list(specials or [])
        base = list(vocab)
        spec_bytes = [s.encode("utf-8") for s in self.specials]
        if spec_bytes and base[-len(spec_bytes):] == spec_bytes:
            base = base[:-len(spec_bytes)]
        self.base_vocab = base
        self.vocab = base + spec_bytes
        self.merges = list(merges)
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._special_re = None
        self._ranks: dict[tuple[int, int], int] = {}
        if self.specials:
            pat = "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True))
            self._special_re = re.compile(pat)
        self._rebuild_ranks()

    # ------------------------------------------------------------------ io
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_special(self) -> int:
        return len(self.specials)

    @property
    def n_bpe(self) -> int:
        return len(self.base_vocab)

    def special_id(self, tok: str) -> int:
        return self.n_bpe + self.specials.index(tok)

    def _rebuild_ranks(self) -> None:
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "vocab": [b.decode("latin-1") for b in self.vocab],
            "merges": self.merges,
            "specials": self.specials,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            vocab=[b.encode("latin-1") for b in payload["vocab"]],
            merges=[tuple(m) for m in payload["merges"]],
            specials=payload.get("specials", []),
        )

    # -------------------------------------------------------------- encode
    def _encode_pretoken(self, chunk: bytes) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        ranks = self._ranks
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = self.merges[best_rank]
            ids = ids[:best_i] + [256 + best_rank] + ids[best_i + 2:]
        out = tuple(ids)
        if len(self._cache) < 2_000_000:
            self._cache[chunk] = out
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if self._special_re is None or not allowed_special:
            return self._encode_text(text)
        out: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            out.extend(self._encode_text(text[pos:m.start()]))
            out.append(self.special_id(m.group()))
            pos = m.end()
        out.extend(self._encode_text(text[pos:]))
        return out

    def _encode_text(self, text: str) -> list[int]:
        out: list[int] = []
        for m in PRETOKEN_RE.finditer(text):
            out.extend(self._encode_pretoken(m.group().encode("utf-8")))
        return out

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    # -------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        n_bpe = self.n_bpe
        chunks = []
        for i in ids:
            if i >= n_bpe:
                if not skip_special:
                    chunks.append(self.specials[i - n_bpe].encode("utf-8"))
                continue
            chunks.append(self.vocab[i])
        return b"".join(chunks).decode("utf-8", errors="replace")


def train_bpe(texts: Iterable[str], vocab_size: int, specials: list[str] | None = None,
              verbose: bool = True) -> BPETokenizer:
    """Learn a byte-level BPE vocabulary over ``texts``.

    Incremental implementation: we keep a pair -> count index over unique
    pre-tokens weighted by frequency, and after every merge only re-scan the
    words that contained the merged pair.
    """
    specials = list(specials or SPECIAL_TOKENS)
    counts: Counter[str] = Counter()
    n_chars = 0
    for t in texts:
        n_chars += len(t)
        counts.update(m.group() for m in PRETOKEN_RE.finditer(t))

    words: list[list[int]] = [list(w.encode("utf-8")) for w in counts]
    freqs: list[int] = list(counts.values())

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_where[p].add(wi)

    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    merges: list[tuple[int, int]] = []
    n_merges = vocab_size - 256
    if verbose:
        print(f"[bpe] {n_chars/1e6:.1f}M chars, {len(words):,} unique pre-tokens, "
              f"{n_merges} merges to learn")

    for step in range(n_merges):
        if not pair_counts:
            break
        (a, b), cnt = max(pair_counts.items(), key=lambda kv: kv[1])
        if cnt < 2:
            break
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append((a, b))

        affected = list(pair_where.pop((a, b), ()))
        for wi in affected:
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                else:
                    pair_where[p].discard(wi)
            nw: list[int] = []
            i = 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[wi] = nw
            for p in zip(nw, nw[1:]):
                pair_counts[p] += f
                pair_where[p].add(wi)

        if verbose and (step + 1) % 512 == 0:
            print(f"[bpe] merge {step+1}/{n_merges}  best_count={cnt}  vocab={len(vocab)}")

    return BPETokenizer(vocab, merges, specials)
