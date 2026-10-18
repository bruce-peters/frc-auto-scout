## "Is it shooting" signals in the credit model (2026-10-18, built, not adopted)

**What.** Two per-robot shooting signals, found by the failure finder's launch research
(`data/exp_launch/RESULTS.md`):
* **DINO "is shooting":** a ridge model on the robot crop's DINOv2 embedding, centred per robot and view
  within the match. It is trained once on 5026's Chezy scout taps (`models/shooting_ridge.npz`).
* **Raw hopper drain:** the robot's FUEL-yellow box fraction falling.

They enter a version-2 credit model as + g · log1p(window mean) per robot (`autoscout/shooting.py`,
`attribution._model_weights`; `tests/test_core.py` checks the port against the dataset code). The deployed
model has no such terms, so they are off.

**Results** (no hindsight, season prior on, 54 scouted Chezy quals):

| check | season prior only | + shooting signals |
| --- | ---: | ---: |
| nested CV fuel MAE (λ 3e-5) | 41.9 | 41.1 (paired −0.8, 95 % CI −2.2 .. +0.7) |
| nested CV share error | 0.126 | 0.124 (−0.002, −0.006 .. +0.002) |
| full pipeline, best-estimate MAE (in-sample: both models fitted on these matches) | 40.9 | 39.4 (−1.5, −2.9 .. −0.0) |
| 254 bursts where scouts say ≥ 60 % 254 but credit is > 0.3 lower | 63 jumps, 152 balls | 67 jumps, 153 balls |

* **On its own** (in-event OPR prior) the signal was worth −2.75 (CI −5.2 .. −0.4). Most of that overlaps with
  the season-to-date prior: both mainly tell the model which robot is the shooter.
* **Caveat:** about 15 specs were tried on these matches, so the CIs are optimistic.
* **Not adopted:** the honest gain is within noise, it does not fix 254's misassigned bursts, and it needs
  the DINO embedding step at every new match. Turn it on with
  `data/exp_attrmodel/tools/export_model.py ds.pkl out.json --spec "lin/all+dist+mlprior+fx:dinoc+fx:drainraw"`
  (`data/exp_attrmodel/attr_model_v8_shoot.json` is that export).
