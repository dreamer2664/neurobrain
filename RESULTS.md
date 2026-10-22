# live results

_updated 2026-10-22 17:16 UTC_

## corpus

| stream | tokens |
|---|---|
| code_train | 53.28M |
| code_val | 1.82M |
| ui_train | 17.16M |
| ui_val | 0.52M |
| vocab | 2052 |
| code bytes/token | 2.908 |

## language brain

- step: **800**
- held-out loss: code **3.174** (ppl 23.9), desktop **0.795** (ppl 2.2)
- syntax-valid generations: **4/10** (40%) at step 200
- desktop protocol: format 100%, action type 58%, click inside target widget 6%

![curves](samples/training_curves.png)

## gripnet (screen pointing)

- step: **750**
- click lands on the instructed widget: **31.6%**
- click within 6 px of widget centre: 9.0%
- action-type accuracy: 100.0%

![grip](samples/grip_predictions.png)

