"""Vectorised scan of a uint16 token stream.

Stats and verify must judge the bytes on disk, not the generator's bookkeeping,
but decoding 100M tokens through Python to look for ``<|retry|>`` is far too
slow. So markers are searched as **token id sequences** with numpy, in chunks,
and the episode invariants are derived from sorted position arrays.

Everything here is O(tokens) time and O(chunk) memory: a 350 MB stream is
scanned in bounded RAM.
"""

from __future__ import annotations

import numpy as np

from . import drive_rules

CHUNK = 8 << 20  # tokens per pass; 16 MB as uint16


def marker_patterns(tok) -> dict[str, list[int]]:
    return {k: list(tok.encode(v)) for k, v in drive_rules.MARKERS.items()}


def eot_id(tok) -> int:
    return tok.encode("<|endoftext|>")[0]


def _iter_chunks(arr: np.ndarray, size: int = CHUNK):
    n = len(arr)
    for s in range(0, n, size):
        yield s, min(n, s + size)


def find_positions(arr: np.ndarray, patterns: dict[str, list[int]],
                   names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Exact positions of each marker's token sequence, chunked.

    All the ``<|...|>`` markers share their first token, so a token -> marker
    lookup would collide. Candidates are therefore grouped by first token and
    every marker in the group is tested against the group's candidates.
    """
    names = names or list(patterns)
    groups: dict[int, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(patterns[name][0], []).append(i)
    first_tokens = sorted(groups)
    table = np.full(65536, -1, dtype=np.int16)  # token id -> group id
    for g, t in enumerate(first_tokens):
        table[t] = g
    group_members = [groups[t] for t in first_tokens]
    maxlen = max(len(patterns[n]) for n in names)
    hits: dict[str, list[np.ndarray]] = {n: [] for n in names}

    for s, e in _iter_chunks(arr):
        hi = min(len(arr), e + maxlen)  # overlap so multi-token markers survive
        chunk = np.asarray(arr[s:hi])
        gid = table[chunk]
        cand = np.flatnonzero(gid >= 0)
        if cand.size == 0:
            continue
        keep = cand < (e - s)
        cand, gid_c = cand[keep], gid[cand[keep]]
        for g, members in enumerate(group_members):
            gc = cand[gid_c == g]
            if gc.size == 0:
                continue
            for i in members:
                name = names[i]
                pat = patterns[name]
                # a candidate too close to the end of the buffer has no room
                # for the rest of the token sequence
                gc_ok = gc[gc + len(pat) <= chunk.size]
                if gc_ok.size == 0:
                    continue
                ok = np.ones(gc_ok.shape, dtype=bool)
                for k in range(1, len(pat)):
                    ok &= chunk[gc_ok + k] == pat[k]
                p = gc_ok[ok]
                if p.size:
                    hits[name].append(p.astype(np.int64) + s)
    return {n: (np.sort(np.concatenate(v)) if v else np.empty(0, np.int64))
            for n, v in hits.items()}


def token_classes(tok, arr: np.ndarray) -> dict[str, int]:
    """Coarse class balance of every token in the stream (lookup table + bincount)."""
    n_bpe = tok.n_bpe
    table = np.zeros(65536, dtype=np.uint8)
    names = ["special", "alpha", "digit", "space", "punct", "byte"]
    vocab = tok.vocab
    for i, b in enumerate(vocab):
        if i >= n_bpe:
            table[i] = 0
            continue
        if not b:
            table[i] = 4
            continue
        c = b[0]
        if 65 <= c <= 90 or 97 <= c <= 122:
            table[i] = 1
        elif 48 <= c <= 57:
            table[i] = 2
        elif c in b" \t\n\r":
            table[i] = 3
        elif 32 <= c < 127:
            table[i] = 4
        else:
            table[i] = 5
    counts = np.zeros(6, dtype=np.int64)
    for s, e in _iter_chunks(arr):
        counts += np.bincount(table[np.asarray(arr[s:e])], minlength=6)
    total = int(counts.sum()) or 1
    return {names[i]: int(counts[i]) for i in range(6)} | {
        "_share": {names[i]: round(float(counts[i]) / total, 5) for i in range(6)}
    }


def episode_bounds(eots: np.ndarray, n_tokens: int) -> np.ndarray:
    """[start, end) pairs for each EOT-terminated episode."""
    if eots.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    starts = np.concatenate([[0], eots[:-1] + 1])
    ends = eots + 1
    return np.stack([starts, ends], axis=1)


def scan(arr: np.ndarray, tok, sample_docs: int | None = None) -> dict:
    """Drive invariants + throughput numbers for a token stream of episodes."""
    patterns = marker_patterns(tok)
    eot = eot_id(tok)
    eots = []
    for s, e in _iter_chunks(arr, 32 << 20):
        chunk = np.asarray(arr[s:e])
        eots.append(np.flatnonzero(chunk == eot).astype(np.int64) + s)
    eots_arr = np.concatenate(eots) if eots else np.empty(0, np.int64)
    bounds = episode_bounds(eots_arr, len(arr))
    n_eps = len(bounds)
    if n_eps == 0:
        return {"episodes": 0, "error": "no episodes (no <|endoftext|> boundary)"}

    pos = find_positions(arr, patterns)

    def count_in(name: str) -> np.ndarray:
        p = pos.get(name, np.empty(0, np.int64))
        if p.size == 0:
            return np.zeros(n_eps, dtype=np.int64)
        lo = np.searchsorted(p, bounds[:, 0], side="left")
        hi = np.searchsorted(p, bounds[:, 1], side="left")
        return hi - lo  # per-episode counts

    # counts per episode
    n_done = count_in("done")
    n_retry = count_in("retry")
    n_err = count_in("err")
    n_code_marker = count_in("task")
    n_plan = count_in("plan")
    mix = terminal_mix(arr, tok, pos["done"])

    # I2: every retry must be followed by an attempt before the episode's EOT
    retry_pos = pos.get("retry", np.empty(0, np.int64))
    acts = np.sort(np.concatenate([pos.get(k, np.empty(0, np.int64)) for k in ("act", "try", "step")]))
    bad_retries = 0
    if retry_pos.size:
        j = np.searchsorted(acts, retry_pos, side="right")
        has_next = j < acts.size
        next_act = np.where(has_next, acts[np.clip(j, 0, acts.size - 1)], np.iinfo(np.int64).max)
        k = np.searchsorted(eots_arr, retry_pos, side="right")
        ep_end = eots_arr[np.clip(k, 0, eots_arr.size - 1)]
        bad_retries = int(np.sum(~((j < acts.size) & (next_act < ep_end))))

    n_with_retry = int(np.sum(n_retry > 0))
    n_with_failure = int(np.sum((n_retry > 0) | (n_err > 0)))

    # kind mix, from the first marker of each episode is expensive; use content:
    # code episodes carry <|task|>, plan episodes carry <|plan|>, the rest are ui
    code = n_code_marker > 0
    plan = (~code) & (n_plan > 0)
    ui = ~code & ~plan
    total = n_eps or 1

    stats = {
        "episodes": int(n_eps),
        "tokens": int(len(arr)),
        "tokens_per_episode": round(len(arr) / total, 1),
        "terminal_rate": round(float(np.mean(n_done == 1)), 6),
        "terminal_over": int(np.sum(n_done > 1)),
        "terminal_missing": int(np.sum(n_done == 0)),
        "retry_after_failure_rate": round(1.0 - bad_retries / max(1, retry_pos.size), 6),
        "retries": int(retry_pos.size),
        "bad_retries": bad_retries,
        "mean_retries": round(float(n_retry.mean()), 3),
        "failure_fraction": round(n_with_failure / total, 6),
        "retry_fraction": round(n_with_retry / total, 6),
        # desktop episodes are the ones with a screen; code episodes have no
        # scan by design, so the gate applies to ui + plan only
        "scan_fraction": round(float(np.mean(count_in("scan")[np.asarray(ui | plan)] > 0))
                               if (ui | plan).any() else 1.0, 4),
        "mix": {"ui": round(float(ui.sum()) / total, 4),
                "code": round(float(code.sum()) / total, 4),
                "plan": round(float(plan.sum()) / total, 4)},
        "retry_hist": _hist(n_retry, 8),
        "terminal_mix": {k: round(v / total, 4) for k, v in mix.items()},
        "failures": {"terminal": [], "retry_gap": []},
    }
    if stats["terminal_rate"] < 1.0:
        miss = np.flatnonzero(n_done != 1)[:5]
        stats["failures"]["terminal"] = miss.tolist()
    if bad_retries:
        stats["failures"]["retry_gap"] = sample_bad_retry_episodes(
            retry_pos, acts, eots_arr, bounds, limit=5)
    return stats


def sample_bad_retry_episodes(retry_pos, acts, eots, bounds, limit=5) -> list[int]:
    j = np.searchsorted(acts, retry_pos, side="right")
    has_next = j < acts.size
    next_act = np.where(has_next, acts[np.clip(j, 0, acts.size - 1)], np.iinfo(np.int64).max)
    k = np.searchsorted(eots, retry_pos, side="right")
    ep_end = eots[np.clip(k, 0, eots.size - 1)]
    bad = retry_pos[~((j < acts.size) & (next_act < ep_end))]
    out = []
    for p in bad[:limit]:
        ep = int(np.searchsorted(bounds[:, 0], p, side="right") - 1)
        if 0 <= ep < len(bounds):
            out.append(ep)
    return out


OUTCOMES = list(drive_rules.OUTCOMES)


def terminal_mix(arr: np.ndarray, tok, done_pos: np.ndarray) -> dict:
    """Classify what follows each terminal marker.

    Marker-string counting cannot do this: "failed" also appears in the code
    retry text ("attempt 1 failed, patch the logic and rerun"). So the outcome
    is matched as a token sequence *immediately after* each <|done|>, longest
    match first.
    """
    counts = {k: 0 for k in OUTCOMES}
    counts["other"] = 0
    if done_pos.size == 0:
        return counts
    done_pat = list(tok.encode(drive_rules.TERMINAL))
    after = done_pos + len(done_pat)
    todo = np.ones(done_pos.size, dtype=bool)
    for name in sorted(OUTCOMES, key=lambda n: -len(tok.encode(n))):
        pat = tok.encode(name)
        ok = todo.copy()
        for k, tid in enumerate(pat):
            idx = after + k
            valid = idx < len(arr)
            ok &= valid
            ok[valid] &= np.asarray(arr[idx[valid]]) == tid
        counts[name] = int(ok.sum())
        todo &= ~ok
    counts["other"] = int(todo.sum())
    return counts


def _hist(x: np.ndarray, nbins: int) -> list[int]:
    x = np.asarray(x)
    if x.size == 0:
        return [0] * nbins
    b = np.clip(x, 0, nbins - 1).astype(np.int64)
    return np.bincount(b, minlength=nbins).tolist()[:nbins]
