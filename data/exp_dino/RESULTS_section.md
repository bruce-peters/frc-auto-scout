## Cross-match DINOv2 robot recognition for team identity (2026-10-18)

**Question.** The 2026-10-17 DINO attempt (PLAN.md section 11) trained a classifier inside one match and made
identity worse: it learned where a robot was, not which robot. Each team plays ~10-12 matches per event,
so a classifier trained on a team's crops from its *other* matches sees it against many backgrounds and
positions. Does that recognise robots, and does it improve identity?

**Answer.** It recognises robots very well. Plugged into today's identity model, it barely moves held-out
identity, because the remaining errors are mostly duplicate tracks. The 6-permutation Viterbi cannot label
those, however sure the evidence is. Combined with the duplicate-capable Viterbi from `data/exp_identity`
(`identity.assign_absent`), the probe takes held-out identity from **91.2 % to 97.4 %**, and every event
improves.

* **Probe accuracy.** Leave-one-match-out on Chezy, the probe names the right team of the alliance for
  **99.3 % of OCR-confident segments** and **97.3 % of single crops** (97.2 % in the high main camera).
* **Background.** Handling it barely matters once training spans matches. The whole-box CLS token scores
  96.5 % of crops, the box-interior patch mean 97.3 %, and the robot-pixel mask 97.0 %.
* **Probe alone.** As an extra evidence term in the current 6-permutation Viterbi, the probe moves held-out
  identity only from 91.15 to **91.71 %** (`w_dino=2`). ResNet galleries reach 91.40 %.
* **Why so little.** On Chezy, the probe already votes for the team named by the held-out bumper reads in
  11.6 of the 12.8 points of error.
  * In 11.4 of those points, a concurrent segment of the same alliance holds that team.
  * In 9.4 points, that concurrent segment *also* looks like that team. These are two tracks of one robot,
    and the permutation model has to give the second track another team.
* **With the duplicate-capable Viterbi.** Using `absent_cost=2` for every team, the probe resolves those
  duplicates: **97.42 %** (Chezy 89.5 -> **98.7 %**). The same model without the probe gives 94.00 %.
* **Against 5026's scouting** (315 team-matches):
  * The video-only split improves: fuel MAE 51.5 -> 48.8, correlation 0.808 -> 0.825.
  * We refit the learned credit model on the new identities with nested leave-one-match-out CV. All four
    scouting metrics improve slightly: MAE 44.6 -> 44.3, share error 0.135 -> 0.131, share correlation
    0.826 -> 0.841, top scorer 62 -> 68 %.
  * The shipped `models/attr_model.json` was fitted on the *current* identities of these same matches.
    With it unchanged, the new identities score worse in-sample (MAE 42.7 -> 46.8), so the credit model
    must be refit together with any identity change.

### Method (`scripts/build_dino_probe.py`, tools in `data/exp_dino/tools/`)

* **embed**
  * Re-reads the crop frames from `data/videos/<match>.mp4` and embeds every stored robot crop with frozen
    DINOv2 ViT-S/14 on the GPU: 671 k crops in 131 matches.
  * Each crop is embedded in 9 crop / pooling strategies at once, at 224 px (the 96 px thumbnails are too
    small for end-camera close-ups).
  * Takes 25 s per match (decode-bound), and writes 35 MB per match to `data/exp_dino/emb/` (4.4 GB).
* **labels**
  * Per crop: its segment, alliance, assigned team and the segment's bumper-OCR evidence.
  * Comes from the finalized `output/<match>/result.json` + `dets.json` (default Options).
* **probe**
  * Per event and leave-one-match-out: a softmax probe over the event's teams (L-BFGS, L2 1e-3,
    standardised features). It is trained on the crops of *other* matches' confident segments, which have
    OCR mass >= 2, >= 80 % on one team, and agree with the assigned team (the `build_galleries.py` rule).
  * Crops are weighted 1/sqrt(segment length), with classes balanced.
  * For the held-out match it stores per crop the log-probabilities over each alliance's 3 teams. Teams the
    probe never saw share the mass it gives to "none of the alliance's known teams", and probabilities are
    floored at 0.02.
  * Output goes to `data/exp_dino/probe/<name>/<match>.npz`. A match's own OCR never trains its probe.
* **pipeline**
  * `Options.w_dino` (default 0 = off) adds `w_dino x` the centred mean per-crop log q of each segment to
    the OCR evidence before the first Viterbi. It gets full weight from 5 crops on.
  * The probe files are read from `pipeline.DINO_PROBE_DIR` (default `data/exp_dino/probe/default`, a copy
    of `box_inner_ocr`).
* **evaluation**
  * `data/exp_dino/tools/eval_ident.py` uses the same metric as `eval_analysis.py --holdout`. It is faster:
    each scan is loaded once for all configs, held-out runs stop after identity, and bumper reads are cached
    in `data/exp_dino/ocr_cache/`.
  * Its base number equals `identity_holdout.py` from finalize_v3 exactly (91.15 %).
  * `scripts/eval_analysis.py` got an optional `--ocr_cache DIR`, which gives the same results about 3x
    faster.

### Crop strategy (Chezy, leave-one-match-out, every 3rd match held out, 1,017 confident segments)

| features | segments | crops | main-camera crops |
| --- | ---: | ---: | ---: |
| whole box, CLS | 99.0 % | 96.5 % | 96.2 % |
| whole box, mean patch | 99.2 % | 97.0 % | 96.8 % |
| **box interior (x 20-80 %, y 15-85 %) patch mean** | **99.3 %** | **97.3 %** | **97.2 %** |
| robot pixels only (not carpet, not FUEL yellow) | 99.1 % | 97.0 % | 96.9 % |
| lower 45 % patches (bumper + chassis) | 99.4 % | 96.8 % | 96.3 % |
| centre crop re-embedded (CLS / mean) | 99.0 / 98.7 % | 95.7 / 95.9 % | 95.6 / 96.0 % |
| lower part re-embedded (CLS / mean) | 99.0 / 98.8 % | 94.1 / 94.8 % | 92.9 / 93.8 % |
| whole-box CLS + robot pixels | 99.2 % | 97.4 % | 97.2 % |

* **Strategy.** Re-embedding a sub-crop loses context and is worse. Masking helps by tenths of a point.
* **Regularisation.** L2 1e-4 to 1e-2 is flat.
* **Label policy.** Training on all Viterbi-assigned segments instead of confident-OCR ones is worse at 3 of
  the 4 other events (better at cagle).
* **Training across all events** (`--pool`) is worse everywhere: on cancmp, segments drop from 76 to 67 %,
  and on joh from 71 to 61 %. Robots change between events.

### Generalisation (box interior, every match held out)

| event | matches | team-slots with training data elsewhere | segments | crops |
| --- | ---: | ---: | ---: | ---: |
| 2026cc | 69 | 100 % | 99.3 % | 97.3 % |
| 2026cagle | 18 | 85 % | 93.3 % | 89.9 % |
| 2026casnv | 16 | 90 % | 92.2 % | 87.3 % |
| 2026cancmp | 15 | 51 % | 76.0 % | 67.5 % |
| 2026joh | 13 | 49 % | 71.2 % | 69.6 % |

Outside Chezy we only have 5026's schedule. Most teams there appear in one of our matches, so the probe knows
only the teams that recur. Segment accuracy counts a tie between two unknown teams as wrong; in the Viterbi
such a tie is simply no evidence.

### Held-out identity (131 matches; % of held-out bumper-read mass that agrees)

| config | all | cagle | cancmp | casnv | cc | joh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current defaults | 91.15 | 94.0 | 88.6 | 92.7 | 89.5 | 94.1 |
| + ResNet galleries, leave-one-match-out (`w_gal` 6) | 91.40 | 94.4 | 88.5 | 92.7 | 89.9 | 94.1 |
| + DINO probe `w_dino=2` | 91.71 | 94.8 | 87.3 | 93.5 | 90.2 | 94.3 |
| + DINO probe `w_dino=4` (8) | 91.60 (91.25) | 94.6 | 87.1 | 93.1 | 90.2 | 94.2 |
| `w_dino=4`, whole-box CLS probe | 91.44 | 94.2 | 87.2 | 92.9 | 90.1 | 93.9 |
| `w_dino=4` instead of in-match ResNet prototypes (`w_app=0`) | 91.51 | 94.6 | 86.7 | 93.1 | 90.2 | 93.9 |
| duplicates for dead robots only (`absent_cost_dead=2`, near bonus 2) | 91.55 | 94.0 | 89.5 | 92.9 | 90.0 | 94.2 |
| ... + `w_dino=2` | 92.70 | 94.8 | 87.9 | 94.5 | 91.7 | 94.6 |
| duplicates for every team (`absent_cost=2`, `absent_cost_dead=2`, near bonus 2) | 94.00 | 95.8 | 92.4 | 94.7 | 93.1 | 95.7 |
| **... + `w_dino=2`** | **97.42** | **96.4** | **93.0** | **97.3** | **98.7** | **95.9** |
| ... + `w_dino=2`, `absent_cost=1` | 97.25 | 96.3 | 92.2 | 97.0 | 98.8 | 95.1 |
| ... + `w_dino=2`, `absent_cost=4` (`w_dino=4`) | 96.61 (96.82) | 96.4 | 91.7 | 97.0 | 97.4 | 95.6 |
| ... + `w_dino=2`, no near bonus | 96.25 | 96.0 | 91.4 | 96.3 | 97.2 | 95.2 |

On cancmp the probe alone hurts (88.6 -> 87.3), because it knows only half the teams there. With duplicate
labels it still adds 0.6 points.

We confirmed the headline numbers with the official tool,
`scripts/eval_analysis.py --holdout --snap output --workers 1 --ocr_cache data/exp_dino/ocr_cache`:

* current defaults: **91.2 %**, share correlation vs fuel OPR train 0.835 / test 0.791;
* `--opt w_dino=2 --opt absent_cost=2 --opt absent_cost_dead=2 --opt absent_near_bonus=2`: **97.4 %**,
  0.831 / 0.789.

Those correlations use the shipped credit model, which has an OPR prior and was fitted on the current
identities, so the flat result is expected.

**What the held-out metric cannot see.** A robot whose bumper is never read could be absorbed as a
teammate's duplicate without costing held-out mass. We checked this two ways:

* **Coverage.** The share of match time with at least one segment labelled for a team falls from 84.8 to
  80.8 % (Chezy). Team-matches with less than 50 % coverage rise from 4.1 to 8.7 %.
* **Who loses coverage.** The 16 scouted team-matches that drop below 50 % have an ordinary scouted shot share
  (0.31; all robots 0.33).
* **Spot check.** In 2026cc_qm25, team 9032 loses 54 s of segments. Those segments carry bumper reads and
  probe votes for its teammate 6238 (q 0.96); the current model had to label them 9032 because a concurrent
  track already held 6238. So the coverage loss there removes wrong labels.

### Per-team fuel (Chezy quals scouted by 5026, 315 team-matches, 54 matches)

The reference is the alliance's TBA total split by scouted shots. Each comparison below is paired, with a
95 % bootstrap CI over matches.

| | current identity | DINO + duplicates |
| --- | ---: | ---: |
| video-only split (`fuel_video`): fuel MAE | 51.5 | **48.8** (-2.7, CI -6.2 .. +0.9) |
| video-only split: correlation / top scorer | 0.808 / 61 % | **0.825** / 61 % |
| learned credit model refit on each identity, nested leave-one-match-out CV: fuel MAE | 44.6 | **44.3** |
| ... share error / share correlation / top scorer | 0.135 / 0.826 / 62 % | **0.131 / 0.841 / 68 %** |
| shipped `models/attr_model.json` (fitted on the current identities of these matches, so in-sample for the left column): fuel MAE | 42.7 | 46.8 |
| share correlation vs fuel OPR, all events, rule split (train / test events) | 0.688 / 0.627 | 0.709 / 0.638 |

* **DINO alone** (6-permutation): video-only MAE 51.5 -> 52.1, not significant.
* **Duplicates alone:** 51.5 -> 50.0.
* **OPR correlation** sits near its noise floor. The learned credit model has an OPR prior, so the
  correlation is only reported for the rule split.

### Recommendation and use

* **Adopt** the probe *together with* duplicate labels and a refit of the credit model:
  * Identity flags: `Options(w_dino=2, absent_cost=2, absent_cost_dead=2, absent_near_bonus=2)`.
  * Refit the credit model with `data/exp_attrmodel/tools/build_ds.py` + `fit_cv.py` / `export_model.py` on
    results analysed with those flags.
* **Do not adopt** `w_dino` alone: it gains +0.6 points of held-out identity and nothing in the fuel split.
* **Adopted 2026-10-18 (finalize_v5).** These four flags are the `Options` defaults, and the credit model was
  refit on the reanalysed results (`data/exp_attrmodel`, "Refit on the DINO identities"). A match without a
  probe file gets the duplicate-capable Viterbi without DINO evidence (94.0 % held out above).
* **New matches:**

  ```
  python scripts/build_dino_probe.py embed                 # GPU, 25 s/match
  python scripts/build_dino_probe.py labels                # after finalize
  python scripts/build_dino_probe.py probe --feat box_inner --name default   # leave-one-match-out, ~8 min
  ```

  Then analyse with the flags above.
* **Space:** `data/exp_dino/emb` keeps all 9 strategies. Only `box_inner` is needed (about 4 MB per match).
