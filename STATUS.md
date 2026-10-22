# Where the project actually is

_2026-10-22, checkpoint step 5210/16000, 4.25M parameters._

## Straight answers

| Question | Answer |
|---|---|
| Can it answer questions? | **No.** Ask it the capital of France and it writes "what is partial des with the capital number of free". It has never seen a question-answering distribution and it is 4.25M parameters. |
| Can it control a browser? | **Not yet.** The runtime can drive a real X11 desktop (12 primitives: click, double/right click, move, drag, scroll, type, key, wait, focus_window, launch, screen). Nothing has driven a browser end to end. The policy that would decide *where* to click scores **0.0 solved** on the simulated desktop today. |
| Can it do anything at all? | **Yes, three narrow things, all in its own dialect:** write small Python functions and then run-test-fix them; emit a complete desktop episode (objective → scan → click → check → retry → terminal); and produce that episode with a correct terminal 100% of the time instead of stopping mid-task. |

## What it actually emits (step 5210, greedy-ish sampling)

```
Q: what is the capital of France?   ->  "what is partial des with the capital number of free..."
def add(a, b):\n    return          ->  "a + b<|run|>pytest -q<|err|>assert add(2, 2) == 4 -> AssertionError<|think|>..."
<|drive|>objective: click submit<|scan|>cancel button at 300,450; search field at 900,300; submit button at 600,900
                                    ->  "0425%<|check|>objective not satisfied: wrong element<|retry|>that was search, not submit..."
```

Read that third one carefully: it knows the protocol, names the right target, and writes an honest retry —
but the coordinate itself is garbage, because 5210 was trained before the scan carried coordinates.

## The numbers, no rounding up

* **LM loss** 0.63–0.81, 4.25M params, 6 layers, ctx 256, 10k tok/s on 2 vCPU.
* **Corpus** code 53.4M + drive 99.3M + ui 17.0M tokens. Drive stream: 77.6% success, 4.7% partial,
  5.1% failed, 0.7% blocked, 6% impossible, mean 1.9 retries/episode, every episode ends in a terminal.
* **Honesty eval (24 impossible objectives):** `overclaim_rate = 1.0`. Given a screen that cannot
  satisfy the objective it claims success 24 times out of 24. This is the single most important
  number in the project and it is currently the worst one.
* **Closed loop (LM → runtime → verifier):** solved 0.0, mean 0.38 executed steps.
* **GripNet (vision grounding):** hit_widget 0.484 @ step 750.
* **Code:** codebench pass@1 0.000 at step 4600 (49 pinned tasks), syntax-valid 0.735.

## Why it is this weak, in one line

4.25M parameters trained for ~5k steps on a 2 vCPU / 2 GB box. That is a lab notebook,
not an agent. Nothing about the architecture or the data pipeline is pretending otherwise —
the runtime, the loop, the verifier, the honesty metric and the corpus generators are built and
tested (106 tests), which is what makes the next order of magnitude a matter of compute, not redesign.

## What has to happen next, in order

1. Let the v4 corpus (coordinates in the scan) finish at least 2k steps, then re-run
   `scripts/e2e_lm_drive.py`. Target: solved > 0.3, overclaim < 0.3.
2. Stage-2 honesty mix (`--honesty data/hon_train.bin --honesty-weight 0.3`) until overclaim < 0.2
   with terminal_rate still 1.0.
3. Turn the code corpus into functional code (codebench pass@1 > 0 on the easy tier).
4. Only then wire the policy to the real desktop end to end (X11 primitives are already there).
