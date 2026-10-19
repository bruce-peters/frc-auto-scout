## No hindsight: every match uses only what a live system would have (2026-10-18)

**Requirement (Bruce).** The system has to work at a new event with zero context, live, match by match.
Nothing may come from the event's later matches or from per-event hand work. Matches already played at the
event may be used as they happen, and fixed models trained once on past data are fine.

**What used hindsight, and what changed** (`pipeline.NO_HINDSIGHT = True`, the default):

| piece | before | now |
| --- | --- | --- |
| DINO robot recognizer (`w_dino`) | probe trained on all the event's other matches | `probe/causal`: each match's probe is trained only on the event's earlier matches (`build_dino_probe.py probe --causal`); the first matches get no DINO vote |
| credit model's fuel-OPR prior | OPR over all the event's quals | `pipeline.prior_opr`: quals played before the match, shrunk toward each team's fuel OPR at its earlier events of the season (`autoscout/season.py`, `SEASON_PRIOR`) |
| credit model | fitted on hindsight inputs | refit on no-hindsight inputs (a fixed model, trained once on 5026's Chezy scouting) |
| end/main camera priors | built from matches as they are processed | unchanged: in order when run live |

**Cost of removing hindsight, and what the season prior buys back.** Nested leave-one-match-out CV against
5026's scouting (105 alliances, 54 Chezy quals) at λ 3e-5, plus held-out identity over all 131 matches:

| run | held-out identity | best-estimate fuel MAE | top scorer | video-only fuel MAE |
| --- | ---: | ---: | ---: | ---: |
| hindsight (DINO + OPR from the whole event), v6 | 97.4 % | 43.6 | 72 % | 46.9 |
| no hindsight, in-event OPR only, v7 | 96.6 % | 44.2 | 70 % | 48.0 |
| **no hindsight + season-to-date prior, v8 (adopted)** | 96.6 % | **41.9** | **75 %** | 48.0 |

* **What hindsight was worth:** about 0.8 points of identity and 0.6-1.1 balls per robot-match. The DINO
  probe accounts for most of it.
* **The season prior** more than recovers that. Public fuel OPR from each team's earlier 2026 events gives a
  prior from qm1, and the model now leans on it more (OPR power 0.29 → 0.55).
* **Caveat:** Chezy is an offseason event, so every team has earlier events. In week 1 of a new season
  there is no season history, and the prior falls back to the in-event OPR (v7 row).

**Update (later on 2026-10-18).** The fuel-OPR / season prior was taken out of the credit split altogether:
per-team numbers now come from each match's own video only (`data/exp_season/RESULTS_section.md`, "Removed
from the default"). The no-hindsight requirement now applies to the DINO probe and the camera priors.
