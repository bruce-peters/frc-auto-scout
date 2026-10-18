## Learned per-team credit split, fitted to 5026's human scouting

**Question.** `attribution.split_increments` shares each scoreboard jump among the alliance's three robots
with the hand-tuned weight `zone * (s + PRIOR*vis + HIDDEN*(1-vis))`. Can a small model fitted to FRC 5026's
scouting of the Chezy Champs quals do better, under honest cross-validation, than that rule and than a
0.7 video + 0.3 fuel-OPR blend of the per-team shares? Does it still hold up at the other events?

**Answer.** Yes, on two of the four scouting metrics. It is the pipeline default since finalize_v4
(`models/attr_model.json`); `result.json` keeps the hand-tuned rule's split as `fuel_video`.

* The learned model cuts the share error from 0.150 (blend) to 0.134 and the team-match fuel error from 49.1
  to 44.5. Both differences have 95 % bootstrap intervals that exclude zero.
* Correlation (0.824 vs 0.817) and top-scorer agreement (65 % vs 66 %) are tied with the blend.
* On the other four events (no scouting), it matches the blend's share correlation with an independent half
  of fuel OPR (0.696 vs 0.696; the current rule gets 0.635) and its top-scorer agreement (60 % vs 60 %).
  Its fuel error against that OPR half is larger than the blend's (39.2 vs 34.9; current rule 40.6). That
  reference has no per-match variation, so it rewards shares squeezed toward OPR, which the scouting shows
  is wrong: the blend's shares have slope 0.50 against scouting, the model's 0.65, ideal 1.
* Nearly all of the gain comes from the OPR prior applied at the match level, together with re-tuning the
  rule's constants. None of the extra video features helped: splitting launch evidence by link kind
  (box / proj / proj_hidden) or by camera view, coverage-based pseudo-shots, or a recall-corrected
  (Gamma-Poisson) rate. A robot's distance to its hub helps a little.

### Data (`tools/build_ds.py`)

* **Source.** After finalize_v3 (`logs/finalize_v3.log` "done in 5437s"), `output/*/result.json` and
  `dets.json` were copied to `final_snapshot/`. `final_snapshot.md5` holds checksums, verified against
  `output/` after finalize. Nothing in `output/` was written.
* **TeamSignals.** These are rebuilt from `result.json` (shots, team paths, overlay counter) exactly as
  `pipeline.analyse` builds them. Re-running the current rule on them reproduces the stored per-team fuel
  to within 3 balls for 256 of the 260 overlay alliances. The other 4 are off by ≤ 4 balls (≤ 2 % of
  share).
* **Rows.** Each scoreboard jump is one row. Each of the three robots has these features:
  * kernel-weighted launch evidence `s`, also split by link kind and by view (own end / other end / main);
  * kernel-weighted launch counts toward / away from the hub;
  * visibility: from the track, per view from `dets.json`, and exclusive coverage (own end / other view
    only / no box);
  * zone fraction;
  * mean distance to its own hub;
  * launches in a ±10 s window and over the whole match.
* **Per alliance.** Fuel-OPR shares (event quals), and each team's typical video share in its other matches
  at the event.
* **Targets.**
  * Match level: scouted shots / alliance scouted shots, averaged over each robot's scouts. Fitting and
    scoring use this target.
  * Per jump: kernel-weighted scouted shot times (`time_valid` rows, tap − 0.8 s, timer = auto start +
    t). Used only as an optional auxiliary loss.
* **Size.** 104 fully scouted alliances in 54 cc quals, 88 of them with usable timestamps for all three
  robots.
* **Noise floor.** On the 90 alliances with two scouts on every robot, the two scouts' shares differ by
  TV 0.128 (correlation 0.843).

### Model (`tools/fit_cv.py`)

For each jump the weights are normalised over the alliance's three robots, so the alliance total stays
exact:

    w_r = (zone_r + 0.001)^gz * (s_r + prior*vis_r + hidden*(1 - vis_r))^b * exp(g_dist * (dist_r - 3.5 m) / 2 m)

* `dist_r` is the robot's mean distance to its own hub over the 0.8–5.5 s window. The term is dropped when
  the position is unknown.
* After all jumps are summed, each team's match total is multiplied by `opr_share_r^g_opr` (floored at
  0.02). The alliance is then renormalised to the same total, and every period's credit is scaled by the
  same per-team factor.
* The old rule is the special case `b = gz = 1`, `prior = 0.12`, `hidden = 2`, `g_dist = g_opr = 0`.
* **Fitting.** Positive constants are fitted in log space by L-BFGS. The loss is the cross-entropy of the
  match share against the scouted share, plus an L2 pull toward the old rule.
* **Cross-validation.** Leave-one-match-out: both alliances of a match are held out together, 54 folds.
  Inside each training fold, a grouped 5-fold CV picks the L2 strength from {0, 3e-5, 3e-4, 3e-3, 3e-2},
  so the held-out numbers are honest. Bootstrap intervals resample matches (2000 draws).

### Results: held-out scouting metrics (104 alliances, 54 matches)

| method | share error (TV) | share corr | top scorer | team-match fuel MAE | slope vs scouting |
| --- | ---: | ---: | ---: | ---: | ---: |
| even 1/3 | 0.251 | – | 38 % | 81.2 | 0 |
| fuel-OPR split | 0.190 | 0.605 | 61 % | 57.7 | 0.38 |
| video only (current rule, stored results) | 0.153 | 0.776 | 62 % | 52.1 | 0.55 |
| 0.7 video + 0.3 OPR (the bar) | 0.150 | 0.817 | 66 % | 49.1 | 0.50 |
| match-level video^b · OPR^g (2 params, LOMO) | 0.141 | 0.804 | 67 % | 47.5 | 0.66 |
| learned evidence only, no OPR (`lin/all+dist`) | 0.148 | 0.797 | 59 % | 50.2 | 0.57 |
| learned evidence + OPR prior on every jump (`lin/all+opr`) | 0.140 | 0.814 | 65 % | 46.2 | 0.62 |
| learned evidence + match-level OPR prior (`lin/all+mlprior`) | 0.136 | 0.824 | 65 % | 45.0 | 0.65 |
| **+ distance to hub (`lin/all+dist+mlprior`, adopted)** | **0.134** | **0.824** | **65 %** | **44.5** | 0.65 |
| own-end-view launches only (`lin/view+dist+mlprior`, not adopted) | 0.131 | 0.836 | 66 % | 43.7 | 0.67 |

* **TV** is half the L1 distance between predicted and scouted shares per alliance. **Fuel MAE** is
  |share difference| × the TBA alliance total, per team-match. **Slope** is the regression of predicted on
  scouted share deviations from 1/3.
* **Adopted model minus the blend (95 % bootstrap intervals):**
  * TV −0.026 … −0.005;
  * fuel MAE −7.5 … −1.3;
  * correlation −0.021 … +0.031;
  * top scorer −8 … +6 points.
* **Learned evidence without OPR** is not distinguishable from the current rule or from the blend
  (TV −0.012 … +0.009 against the blend). The video features alone carry no more signal.
* **The own-end-view variant** drove the weight of main-view and other-end launches to zero
  (`exp(-26)`, `exp(-9)`) with no L2 shrinkage. It transfers slightly worse: video-only share
  correlation with fuel OPR on the other events is 0.712, against 0.724 for the current rule. It would also
  need per-shot views in TeamSignals, so it was not adopted.
* **Also tried on the pre-finalize snapshot, none better than the plain form:**
  * per-kind gains (box / proj / proj_hidden);
  * per-view gains;
  * pseudo-shots by exclusive coverage;
  * Gamma-Poisson rate `(s + α)/(own + r_oth·other + r_none·none)`;
  * a ±10 s launch window;
  * match-level launch rate as a prior or as a scale on the pseudo-shots;
  * a separate auto temperature;
  * per-period instead of match-level OPR prior (TV 0.148 vs 0.146 on the dev snapshot);
  * the team's typical video share in its other matches (−0.002 TV, needs other matches' outputs);
  * the per-jump scouted-timing loss (worse TV: scout timestamps are too noisy at the jump level).

**What the fit says** (all 104 alliances, L2 3e-5, `attr_model.json`):

* `prior` 0.12 → 0.002: a robot in plain view that launches nothing gets essentially no credit.
* `hidden` 2.0 → 6.2 and `b` 1 → 0.60: evidence is compressed. A robot with 4× the launch evidence gets
  about 2.3× the weight, and a robot no camera sees keeps a real share. This fits launch detection
  saturating in volleys and noisy links.
* `gz` 0.76: the teleop zone gate is a little softer.
* `g_dist` −0.65 per 2 m: a robot 2 m further from its hub than another gets 0.52× its weight.
* `g_opr` 0.37: each team's match total is multiplied by its OPR share^0.37.

### Transfer to the other events (`tools/transfer.py`, `transfer_final.json`)

* **Fit.** The models are fitted on all 104 scouted Chezy alliances and applied to the 120 overlay
  alliances of cagle, cancmp, casnv and joh.
* **Video-only models** are scored against the full-event fuel OPR, the historical metric.
* **Models that use OPR** are scored split-half: the prior is the OPR from the event's odd-numbered quals and
  the reference is the OPR from its even-numbered quals, then the other way round, both pooled. All
  OPR-using methods get the same half-OPR input.

| other events (120 alliances) | share corr | top scorer | fuel MAE vs OPR |
| --- | ---: | ---: | ---: |
| video only, current rule, vs full OPR | 0.724 | 60 % | 35.8 |
| learned evidence (no OPR), vs full OPR | 0.722 | 59 % | 36.2 |
| current rule, vs half OPR | 0.635 | 54 % | 40.6 |
| OPR prior alone, vs other half | 0.654 | 56 % | 35.1 |
| 0.7 video + 0.3 OPR, vs other half | 0.696 | 60 % | 34.9 |
| **learned model, vs other half** | **0.696** | **60 %** | 39.2 |

* **Per event, share corr of learned model vs blend (split-half):**

  | event | learned model | blend |
  | --- | ---: | ---: |
  | cagle | 0.827 | 0.821 |
  | cancmp | 0.651 | 0.635 |
  | casnv | 0.689 | 0.694 |
  | joh | 0.400 | 0.390 |

  At joh, half-event OPR alone is nearly useless (0.126). There, OPR^0.37 at the match level hurts less
  than the 30 % linear blend.
* **No sign of over-fitting to Chezy.** The learned video constants leave the other events' correlation
  with OPR unchanged, and the full model matches the blend there.
* **The higher fuel MAE vs OPR is expected.** The model keeps more per-match variation than the blend, and
  an OPR reference treats all of it as error.

### Implementation (default since finalize_v4)

**`autoscout/attribution.py`**

* `ATTR_MODEL` defaults to `models/attr_model.json` when that file exists. `AUTOSCOUT_ATTR_MODEL` overrides
  it with another path, or `none` for the hand-tuned rule. `load_model()` caches the JSON. The unit tests
  pin the rule (`tests/conftest.py`).
* `TeamSignals.dist_t` / `dist_v` / `dist_mean()`.
* `_model_weights()` gives the per-jump weights above. It also honours the absent / parked flags
  (`absent`, `prior_off`, `no_hidden`) on the pseudo-shot term, the same way the rule does. The model was
  fitted with those flags off.
* `_opr_factors()` applies the match-level prior. It rescales the per-period credit and the `fuel_log`
  entries, so the alliance total is unchanged.
* `split_increments(..., opr=None)` takes an optional `{team: fuel OPR}`.

**`autoscout/pipeline.py`**

* The zone block fills the distance-to-hub samples when a model is loaded (`_track_position` on the same
  0.5 s grid).
* `_attr_opr(info)` passes the event's fuel OPR. It is cached per event, skipped if TBA fails, and only
  computed when the model has `g_opr`.
* `result.json` gets `"credit_model"` when a model is in use.

**Weights.** `data/exp_attrmodel/attr_model.json` holds the readable constants, the fit metadata and the
cross-validated metrics.

**Tests.** `tests/test_core.py::test_learned_credit_model_matches_rule_form_and_opr_prior_keeps_total`:
* a JSON with the rule's own constants reproduces the rule;
* the OPR prior keeps the alliance total, and the `fuel_log` still sums to it.

`pytest tests -q`: 41 passed.

**To turn it off.** In PowerShell:

    $env:AUTOSCOUT_ATTR_MODEL = "none"
    .venv/Scripts/python scripts/finalize.py

The OPR prior needs the event's TBA match list, which is cached in `data/cache/tba`.

**Parity** (`tools/parity.py`, `parity.log`).

* **The attribution code.** `split_increments` with the model, fed TeamSignals rebuilt from all 259 stored
  overlay alliances, reproduces the cross-validated offline model exactly (max share difference 4e-16).
* **The pipeline wiring.** `pipeline.analyse` was run in memory on 2026cc_qm15 and qm41. With the model on,
  shares match the offline model computed from the same run to ≤ 0.003; that gap comes from path rounding
  in `result.json`. Alliance totals are unchanged (517.1 vs 517.0, 765.0, 494.0, 859.0).
* **One example.** On qm41 blue, the rule's shares (0.432, 0.279, 0.289) become (0.324, 0.300, 0.376).
* **Setup.** Bumper OCR of crops missing from the scan cache was stubbed out, because CPU OCR takes
  tens of minutes per match. This changes identity equally for both runs.

### Caveats

* **One event, one scouting team.** Everything was fitted on 104 alliances at a single event, scouted by
  one team. The L2 search picked almost no shrinkage, so the constants moved far from the old rule
  (`prior` 0.12 → 0.002). The 54-fold held-out numbers are the honest estimate. Re-fit the model when more
  scouting arrives:

      build_ds.py SNAPSHOT ds.pkl
      fit_cv.py ds.pkl --boot
      export_model.py ds.pkl attr_model.json --lam 3e-5 --cv cv.json

* **Shots, not makes.** The scouting reference counts shots, not makes. A team that shoots more
  accurately than its partners is under-weighted by the reference, not by the model.
* **The OPR prior is a team-level average.** It pulls a team's per-match share toward that average.
  Early in an event, or for a team that broke down in one match, the OPR is a poor guide. The strength
  (power 0.37) does not depend on how many matches the OPR is built from. The prior multiplies the video
  share, so a robot the video gives almost nothing (dead or absent) stays near zero.
* **Not affected:** alliances whose counter comes from TBA period totals instead of the overlay
  (2 of 262 here) use the same weights, and `split_nnls` is untouched.

### Refit check on the finalize_v4 outputs (`v4_snapshot/`, `ds_v4.pkl`, `cv_v4.log`)

finalize_v4 changed the model's inputs: the parked-robot rule (`Options.park_ext = 1.0`), the
other-match / phantom counter-read filter (cc_qm14 red was 0) and more cached bumper OCR. The dataset
builder now reads the rule's split from `fuel_video` and rebuilds the parked flag (`prior_off`), and the fit
applies that flag to the pseudo-shot term, as the runtime does. That brings the rebuilt-rule parity back to
5 of 260 alliances off by more than 3 balls (each ≤ 4).

| nested CV, 105 scouted alliances | share error (TV) | share corr | top scorer | fuel MAE |
| --- | ---: | ---: | ---: | ---: |
| old rule (rebuilt), v4 | 0.152 | 0.779 | 60 % | 51.5 |
| 0.7 video + 0.3 OPR, v4 | 0.149 | 0.821 | 65 % | 48.6 |
| lin/all+dist+mlprior, v3 (deployed fit) | 0.134 | 0.824 | 65 % | 44.5 |
| lin/all+dist+mlprior, v4 | 0.135 | 0.826 | 62 % | 44.6 |
| lin/view+dist+mlprior, v4 | 0.134 | 0.826 | 65 % | 44.4 |

The refit's constants are close to the deployed ones and its held-out scores are the same, so the
deployed `models/attr_model.json` was kept. The per-view variant is again within noise of the
deployed spec and is not adopted.

### Refit on the DINO identities (`v5_snapshot/`, `ds_v5.pkl`, `cv_v5.log`; deployed in finalize_v5)

The DINO probe plus duplicate-capable identity labels became the default (`data/exp_dino`), so every match
was reanalysed and the model refit. The absent identity state takes PRIOR/HIDDEN credit away from a team in
stretches where no segment carries its label. The dataset now replays those stretches (features `pres`,
`hid_on`), as `_model_weights` does at runtime; parity is 7 of 260 alliances off by more than 3 balls (each
≤ 4).

| nested CV, 105 scouted alliances | share error (TV) | share corr | top scorer | fuel MAE |
| --- | ---: | ---: | ---: | ---: |
| old rule, v4 identities | 0.152 | 0.779 | 60 % | 51.5 |
| old rule, DINO identities | 0.142 | 0.806 | 61 % | 48.8 |
| 0.7 video + 0.3 OPR, DINO identities | 0.143 | 0.832 | 69 % | 46.7 |
| lin/all+dist+mlprior, v4 identities | 0.135 | 0.826 | 62 % | 44.6 |
| **lin/all+dist+mlprior, DINO identities** | **0.131** | **0.841** | **67 %** | **43.9** |

Exported at λ 3e-5 (`attr_model_v5.json` -> `models/attr_model.json`; the v4 model is kept as
`models/attr_model_v4.json`). With better identities the model leans less on its fallbacks: HIDDEN
6.2 -> 1.06, OPR prior power 0.37 -> 0.29, zone power 0.76 -> 0.43.
