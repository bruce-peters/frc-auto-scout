## Main-camera calibration at Chezy Champs (failure-finder cycle 8)

**Problem.** 30 % of the Chezy (2026cc) main-view robot boxes had no field position, so they got no track,
team or credit. Other events had 0.1-1.1 %. Within a match the off-field share swung from 0 to 0.89 per 4 s
pose window, while every window fitted its 4 tags to 0.5 px.

**Root cause: a Rodrigues-branch bug, not the tags.** The Chezy main camera looks back across the field from
high in the stands, so its rotation is close to 180 deg (|rvec| 2.96-3.34 at cc; 1.1-2.1 for every other main
view). Near 180 deg every rotation has two rotation vectors, r and r(1 - 2pi/|r|).

* The per-window fits returned either branch: consecutive windows switched branch 8-22 times per match in
  19 of 23 matches.
* `CameraTrack.at` blended neighbouring windows linearly, so between two windows on different branches the
  pose was garbage and the floor points landed tens of metres away.
* The four matches without a single switch (qm3, qm7, qm15, qm30) are exactly the four with ~0 off-field
  boxes.

**Two smaller, real problems from the tags.** The main view sees only tags 2, 11, 21 and 24. They lie in one
plane at one height, ~12 px wide, 20-30 m away.

* Each window's own fit puts the camera centre somewhere else: the window centres scatter 0.45-1.56 m around
  their median at cc, against at most 0.36 m at any other event.
* Focal length trades against distance: per-match fits give f 1800-3000 px and centre y 25-38 m.
* The camera is physically the same in every cc match: the tags sit at the same pixels to within 3 px and the
  same width.

### What changed

`autoscout/field.py`: one local edit, already live.

* `CameraTrack.at` now blends on the first pose's branch: before blending, r1 is moved to its other branch if
  that is closer to r0.
* Where consecutive rvecs are already on one branch the result is bitwise identical. Checked on a 0.1 s grid
  for all 48 non-cc cameras of the 16 non-cc matches below: 0 difference.
* In all of `output/` (92 scans) it changes only views that actually switch branch:
  * 2026cc main views;
  * 7 cc end views (e.g. qm28 blue_end, 10 switches);
  * one non-cc camera, cancmp_qm44 blue_end: one window has |rvec| 7.9, which is the 2pi-wrapped twin of
    1.6.
* Every saved scan gets this on a plain reanalyse. No rescan or recalibration is needed.

`autoscout/maincal.py` (new):

* `fit_static`: one focal length, k1 and camera centre for every window of a match, with one rotation per
  window (a pan/tilt head may turn but does not move). It is a robust least-squares fit on all windows' tag
  corners, started from the stored fit and from the planar-PnP centre (`ippe_start`, `cv2.solvePnPGeneric`
  `SOLVEPNP_IPPE`; physically impossible solutions are dropped). Output rvecs are kept on one branch
  (`unwrap_rvecs`).
* Optional terms, used in the experiments only:
  * a floor term (robot boxes must land on the field);
  * a far-wall term;
  * fixed landmark points.
* `static_from_prior`: the event K, distortion and centre held fixed, with a rotation per window about that
  centre.
* `load_main_prior` / `update_main_prior` store the event prior in `data/cache/calib_<event>_main.json`:
  * each match's static fit is recorded there;
  * the prior is their median (via `field.load_calib_prior`, >= 3 matches, the match itself excluded).
  * A hand-checked camera in `calib_<event>_main_landmarks.json`, if present, is "trusted" (see below).
* `guard`, the safety net:
  * a window pose that puts >= 20 % of its own boxes off the field is dropped, and its time is interpolated
    from the neighbouring poses;
  * a match is flagged when more than half of its poses are bad.
* `centre_scatter`: the trigger. Nothing new happens unless the stored window centres scatter by more than
  0.4 m. Over all 92 scans in `output/`, the largest non-cc value is 0.36 m (cagle_qm27) and the smallest cc
  value is 0.45 m (qm17).

`autoscout/pipeline.py`

* `calibrate()`: for a main view over the trigger, it builds the static fit and, if an event prior exists, the
  prior's rotation-only track. The static fit is recorded in the prior. The candidates go into
  `fallbacks["main"]`.
* `accept_main_cam(md, candidates, prefer, trusted)` is the post-scan gate, called by `process_match` before
  the end-view gate. Every candidate is first cleaned by `guard`. Pass rules:
  * **Normal candidate:** the share of main boxes off the field must drop by more than 0.002, and the end
    views' mean `cross_view_error` median may rise by at most 0.02 m.
  * **Preferred candidate** (the event-prior track) also passes if the share does not rise (beyond 0.002).
    This deviates from "must drop": the event prior is there to make the cameras consistent across matches,
    which also matters where nothing is off the field.
  * **Trusted landmark camera:** the cross-view check is only reported, not enforced.
  * The report is kept as `md.main_calib`; a flag is printed.
* The flag `MAIN_STATIC = True` turns all of this off when set to False.

`scripts/recalibrate_main.py <scan_dir> <match>... | --event EV`:

* **Pass 1:** the main tags are detected again from `data/videos/<match>.mp4`, at the same frames as
  `calibrate`. The static fit is made and recorded in the event prior.
* **Pass 2:** the same candidates and gate as fresh scans.
* Saving: `pipeline._save_scan` writes to a temporary folder, then `os.replace` puts it in place, with
  `maincal_recovery.json` beside it.
* A second run leaves fixed scans byte-identical (checked by md5).
* Options: `--dry-run` (it still records the static fits in the prior cache), `--landmarks FILE`,
  `--no-prior`, `--force`.

Tests (`tests/test_maincal.py`, 4 tests) cover:

* blending across branches;
* the static fit;
* the guard;
* the gate, including that a camera with nothing off the field is left alone.

`tests/`: 34 passed.

### Results: in-memory `pipeline.analyse` on copies of the 23 cc scans (`data/exp_maincal/before`)

OCR used cached reads only, so every variant sees the same reads (as in failure-finder c8).

| main camera | off-field boxes | held-out identity | cross-view m | share corr vs fuel OPR (top) | vs scouting: MAE / corr / top / burst |
| --- | ---: | ---: | ---: | ---: | --- |
| before (stored, old `at`) | 0.269 | 85.1 % | 0.748 | 0.573 (24/46) | 52.5 / 0.785 / 49 % / 60.6 % |
| `at` fix only (live now) | 0.026 | 90.3 % | 0.659 | 0.510 (21/46) | 54.0 / 0.779 / 46 % / 61.5 % |
| + guard only (earlier version: blended poses) | 0.011 | 90.8 % | 0.634 | 0.502 (23/46) | 54.9 / 0.771 / 46 % / 61.6 % |
| static fit for every match | 0.011 | 90.5 % | 0.634 | 0.520 (21/46) | 56.4 / 0.749 / 54 % / 60.6 % |
| **default path** (`recalibrate_main` / fresh scans) | **0.010** | **90.9 %** | **0.631** | **0.551 (23/46)** | **52.9 / 0.807 / 54 % / 61.2 %** |
| + trusted landmark camera (`--landmarks`) | 0.010 | 92.8 % | 0.679 | 0.533 (20/46) | 50.7 / 0.809 / 49 % / 62.6 % |
| (per-match landmark fit, reference) | 0.010 | 93.3 % | 0.717 | 0.543 (20/46) | 49.4 / 0.807 / 49 % / 63.7 % |

* The scouting columns use `score_scouting.score()`: 105 team-matches in 18 scouted quals. "burst" is the
  share of balls per scoreboard jump that went to the robots the scouts saw shooting.
* The default path uses the leave-one-out median prior. Of the 23 matches, it replaced main with the
  event-prior track in 17, with the static fit in 2, and kept the stored camera in 4. None was flagged.
  qm54 and sf9m1, where c8d's guard failed, were bad only because of the branch bug.
* Per match, off-field share goes before -> after the `at` fix -> default:
  * qm12 0.32 -> 0.13 -> 0.06;
  * sf2m1 0.37 -> 0.09 -> 0.09;
  * every other match is at 0.03 or below.
* Held-out identity per match moves by up to ±8 points either way (e.g. qm6 93 -> 86, sf12m1 88 -> 96).
  Read the pooled numbers.
* Share correlation vs OPR falls with every fix, from 0.573 to 0.50-0.55, on 46 alliances. The default path
  recovers most of the drop, and the scouting correlation and top-scorer rate rise.

**Other events are untouched.** On 16 matches (cagle, cancmp, casnv, joh: 4 each):

* the trigger never fires, and the `at` change is bitwise identical there;
* so the default path gives exactly the "before" numbers: off-field 0.016, held-out 89.9 %, OPR corr 0.613.

For comparison, forcing the static fit or the guard onto those events gave:

* off-field 0.016 -> 0.001-0.004;
* held-out identity 89.9 -> 89.5 / 89.8 %;
* per-match swings up to -12 points (cancmp_qm5).

That is why the trigger exists.

### What the tags cannot fix: the far side of the field is compressed by ~1.5 m

All tag-only cameras put the far wall (y = 0) about 45 px above the real carpet edge. They also put the hub
openings far above the hub tops: see `img/2026cc_qm3_60.jpg` and `img/manual_2026cc_qm3_60.jpg`.

**Reference camera.** I hand-picked two pixels on 2026cc_qm3: where the white centre line meets the far and
the near wall base. A fit to the tags plus these two points:

* draws the far wall, the hub bases and the hub openings where they are;
* gives f 1423 px, centre (7.8, 21.5, 6.5) m, tag rms 0.66 px.

Its file is `prior_lm.json`, built by `tools/m19_prior_lm.py`.

**Test against the robots.** Robots do drive along the far wall:

* Under the reference camera, the 0.5 % quantile of robot y is 0.52-0.58 m, about half a robot from the
  wall.
* Under every tag-only fit it is 1.56-1.75 m: no robot ever comes within 1.5 m of the far wall.
* The low cc end cameras (0.6-0.7 m high, at the near corners) do the same: 0.95-1.8 m (`m18_endq.log`).

So cross-view agreement with them cannot referee the far side. This is why the reference camera's
cross-view number is worse even though it fits the image.

**What did not help.** Neither the floor term, nor a far-wall coverage term, nor fixing k1 = 0 pinned focal
length and elevation from the tags (`m16_far.log`, `m17_k1.log`). Automatic field-line landmarks would, but
are not built.

**Downstream effect.** Using the reference camera as a trusted event camera (rotation-only per match):

* held-out identity rises 90.9 -> 92.8 %;
* scouting MAE falls 52.9 -> 50.7;
* OPR correlation is 0.551 -> 0.533 (within noise).

### Caveats

* **Hub-region features.** `hub_feats` (the vision-only counter's hub regions) were computed at scan time with
  the broken interpolation, in cc scans made before the fix. Only `pipeline.rescan_fuel` refreshes them.
  Robot positions, shots and credit are all recomputed by `analyse`.
* **The live prior.** Fresh cc scans already use the wiring: most scans from qm44 on carry `main_fix`, and
  `data/cache/calib_2026cc_main.json` holds 7 matches. The median prior keeps the tag bias. It buys
  consistency across matches, not accuracy.
* **Disk.** `data/exp_maincal/{before,script_test}` (~3.2 GB) are experiment copies and can be deleted.

### Apply to the installed scans

```
.venv/Scripts/python scripts/recalibrate_main.py output --event 2026cc
```

Then reanalyse the cc matches, e.g. with `finalize.py`. Each refitted match takes about 10-15 s on the CPU.
Scans that fresh processing already fixed are skipped.

**Optional: the landmark camera.**

```
cp data/exp_maincal/prior_lm.json data/cache/calib_2026cc_main_landmarks.json
.venv/Scripts/python scripts/recalibrate_main.py output --event 2026cc --force
```

With the file in `data/cache`, fresh cc scans prefer the landmark camera too.

Tools and logs are in `data/exp_maincal/tools/` (m1-m22) and `data/exp_maincal/*.log|json`.
