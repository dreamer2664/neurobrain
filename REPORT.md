# Morning report

_generated 2026-10-23 05:23 local, checkpoint step 18600_

## The two bars

| bar | target | measured |
|---|---|---|
| 50 basic questions | 90% | **98.0%** (50 questions) |
| open chrome, research, report in google docs | 95% | **30.0%** (20 tasks) |

### Question answering

* seen facts, unseen question templates: **98.0%** (n=50)
* facts never trained on: **16.7%** (n=12) - this is generalisation, not recall, and it is deliberately not part of the bar above

| question | expected | model | |
|---|---|---|---|
| Can you tell me the capital of Bulgaria? | Sofia | `Sofia` | ok |
| What does CPU stand for? | central processing unit | `central processing unit.` | ok |
| Which continent does Argentina belong to? | South America | `south America` | ok |
| Currency used in Netherlands? | euro | `euro` | ok |
| Chemical symbol of argon? | Ar | `Ar.` | ok |
| Atomic number of uranium? | 92 | `92` | ok |
| Which language is spoken by people in Israel? | Hebrew | `Hebrew.` | ok |
| What gas do humans need to breathe? | oxygen | `oxygen` | ok |
| Can you tell me the capital of North Korea? | Pyongyang | `Pyongyang.` | ok |
| What is an array? | an ordered collection of values | `an ordered collection of values` | ok |

_(qa corpus: 411 facts, 41 held out, 0.70M tokens)_

### Research and report on the mock desktop

6/20 tasks finished with the right answer in a saved document, mean 5.8 actions, action precision 88.8%, overclaim 5.0%.

| topic | answer | document | saved | |
|---|---|---|---|---|
| the capital of Japan | Tokyo | `Tokyo` | True | ok |
| the boiling point of water | 100 degrees Celsius | `(empty)` | False | **miss** |
| the largest planet | Jupiter | `(empty)` | False | **miss** |
| the currency of Japan | the yen | `(empty)` | False | **miss** |
| the symbol for oxygen | O | `Brasilia` | True | **miss** |
| the capital of Japan | Tokyo | `Tokyo` | True | ok |
| the currency of Japan | the yen | `the yen` | True | ok |
| the currency of the United Kingdom | pound sterling | `about 300,000 kilometres per second` | True | **miss** |
| the boiling point of water | 100 degrees Celsius | `(empty)` | False | **miss** |
| the capital of Italy | Rome | `(empty)` | False | **miss** |
| the capital of Brazil | Brasilia | `Rome` | True | **miss** |
| the currency of Japan | the yen | `Paris` | True | **miss** |

_(web corpus: 31.75M tokens of click-by-click transcripts)_

### Honesty, unchanged from the drive loop

* terminal_rate n/a, overclaim n/a, impossible answered honestly n/a

## What is still not true

* The research is done against a mock desktop, not a real Chrome. The X11 primitives exist and are tested, but no browser has been driven.
* "Research" means reading an article page whose text the runtime put there; the model copies the answer, it does not search the open web.
* The knowledge base is 411 hand-checked facts, not the internet.

