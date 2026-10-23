# Where the project actually is

_2026-10-23, 05:25Z. Checkpoint step 18,600 of 21,000, 4.25M parameters, trained
from scratch on this box. Nothing pretrained is anywhere in the tree._

## The two bars

| bar | target | last measured | verdict |
|---|---|---|---|
| 50 basic questions | 90% | **98%** (49/50) | **passed** |
| open chrome, research, report in docs | 95% | **30%** (6/20, max 12 actions) | **not passed** |

The second number is the honest one, measured by the runtime executing every
click. It was 0% a few hours ago and every one of those failures was traced to a
defect in the world or in the corpus, not to the model's capacity. The three that
mattered most are now fixed and pinned by tests (see `tests/test_web_world.py`):

1. the docs screen did not show the document's contents, so a paste changed
   nothing the policy could see and it pasted seven times without saving;
2. a wrong search result left the results page byte-identical, so the policy had
   no evidence it had erred and re-clicked the same row;
3. the corpus never put a `<|scan|>` between paste and save, while the runtime
   always appends one — the training pattern was never the pattern at run time.

What is left is one skill: matching the objective's wording ("the boiling point
of water") to the one row on the results page that names it. The policy clicks
the right *kind* of element in the right sequence — chrome, article, copy, docs,
paste, save, and it recovers from a wrong article with an honest `<|retry|>` and
a click on back — but it chooses the row by position, not by reading it. That is
the difference between 30% and 95%.

## What it can do right now

* answer basic factual questions at 98% on the trained curriculum (capital,
  currency, continent, language, chemical symbol, atomic number, misc, code);
  17% on facts it has never been trained on, reported separately and never
  counted in the bar above;
* run the whole research-and-report episode in the mock desktop: launch chrome,
  open an article, copy the answer, open docs, paste, save — verified by the
  runtime's own state, not by what the model says;
* emit an honest retry when an action did not do what it intended, and go back
  rather than clicking the same dead element forever;
* gripper: 0.88 hit-widget on the UI screenshot grounder (0.44M params).

## What it cannot do yet

* read a results page and pick the row that answers the objective (see above);
* write code that passes a test suite: val loss 2.44 on code, codebench pass@1 0;
  the corpus is there, the model is 4.25M parameters and it shows.
* anything on a real desktop: the primitive layer exists (12 primitives, X11),
  the policy has only ever been evaluated in the mock world.

## The shape of the thing

* one 4.25M-parameter transformer, 6 layers, 7 heads, d=224, context 256,
  vocab 2052, its own byte-level BPE trained on its own corpora;
* one text protocol for everything: `<|drive|>objective … <|scan|>elements …
  <|step|>n<|act|>click(x,y)<|obs|>… <|check|>… <|retry|>… <|done|>…`;
* five data domains, all generated here: code 53M tokens, desktop drive 98M,
  UI 17M, facts 0.7M, mock-web 30M;
* a runtime that executes the click and writes the observation. The model's own
  claims are never evidence: a `<|done|>success` the world does not confirm is
  recorded as an overclaim and the loop keeps working.

## Files that matter

* `scripts/demo_brain.py` — one command, both bars, transcripts included
* `scripts/e2e_web_task.py` — the web bar, runtime-verified
* `scripts/qa_eval.py` — the question bar plus the unseen-fact probe
* `scripts/web_oracle.py` — label-driven oracle; 100% on every task, which is how
  we know a policy score means something
* `tests/test_web_world.py` — the invariants above, as tests
