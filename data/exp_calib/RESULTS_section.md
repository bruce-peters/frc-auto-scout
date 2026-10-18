## End-camera calibration recovery: event prior + fixed-intrinsics fallback

**Problem.** In 7 of the 76 processed matches, the alliance's own end camera had no calibration
(failure-finder cycle 3h). Every box in that view then had no field position, so its launches could
not be credited. Three causes:

* the optimiser landing in a bad minimum with 4-7 tags in view (cancmp_qm114 f 3001 px / rms 7.9;
  cancmp_qm52 f 52 px / rms 43);
* only two coplanar tags in view (cagle_qm32, cagle_sf13m1: tags 5 and 8 on one face of the red hub);
* the camera aimed away from the tags (casnv_qm7: one tag; cc_qm15: 0-2 tags; joh_qm14: none).

**Result.** Six of the seven views are recovered. joh_qm14 never shows a tag and stays uncalibrated.
For the affected alliances:

* credited launches 423 -> 722;
* held-out identity 86.1 -> 88.7 %;
* share error vs fuel OPR 0.191 -> 0.174 on average, but only 3 of the 6 recovered alliances improve;
* on cc_qm15, agreement with 5026's scouting improves (team fuel MAE 72.4 -> 65.9, correlation
  0.77 -> 0.85).

Views that already calibrate are unchanged.

### What changed

`autoscout/field.py`

* **Event prior (a).** `update_calib_prior` / `load_calib_prior` keep
  `data/cache/calib_<event>_<view>.json`:
  * it stores each match's plain end-view fit (rms < 4, centre near the field) and the median K,
    distortion and camera centre over them, used once there are at least 3 matches;
  * recovered cameras are marked `cam.recovered` and never feed it.
  * The caches for all 5 events are seeded from the scans in `output/`. Per event and view this is
    12-18 matches; e.g. cancmp blue_end: f 687 px, k1 -0.307, centre (0.62, -0.13, 0.52) m.
* **Focal-length gate (b).** `calibrate_view_dynamic(..., prior=)` and `calibrate_view(..., prior=)`
  also start from the event K and reject any fit whose focal length is more than 30 % off the event
  median. With `prior=None` (the default) the code path is unchanged.
* **Fixed-intrinsics fallback (c).** `calibrate_view_fixed`. K and distortion are fixed to the
  event values, and one pose is solved per sampled frame (every 4 s, the same frames as
  `pipeline.calibrate`):
  * frames with 2 or more tags: `solvePnPRansac` drops outlier tags. Frames where at least 3 tags
    agree are "anchors".
  * every frame with at least 1 tag: a rotation about the camera centre (one tag is enough: 8
    equations, 3 unknowns), with a tag-level consensus. Each tag in turn proposes the rotation, and
    the proposal the most other tags agree with wins.
  * with 3 or more anchors, this match's camera centre and all rotations are refined jointly. The
    refine starts once from the event centre and once from the anchors' median centre, and keeps the
    start whose result has the lower truncated-quadratic tag cost.
  * isolated pose jumps are dropped, and poses are interpolated in time like any `CameraTrack`.
* `recover_view` returns the candidates: the event-K fit (if rms < 4) and the fixed-K track.

`autoscout/pipeline.py` (calibration section only)

* `calibrate(..., event=None, match=None, fallbacks=None)`. With `event` set:
  * every end-view fit is added to the prior;
  * a failed end view gets recovery candidates in `fallbacks`.
  * With the defaults it is unchanged.
* **Accept gate (d).** `cross_view_error` and `accept_recovered_cams`:
  * each robot box in the view is placed with the candidate camera exactly as `analyse()` places it;
  * the distance to the nearest same-alliance main-view robot at the same instant is measured;
  * a candidate passes on a median under 1 m over at least 30 boxes, and the best passing candidate
    replaces `md.cams[view]`.
* `process_match` is **not** wired yet: the main session asked for the default path to stay unchanged
  while its batch runs. To enable recovery in fresh scans, replace its `cams = calibrate(...)` line
  with:

  ```python
  recover = {}
  cams = calibrate(video, prod, tl.auto_start - 1, tele + config.TELEOP_LEN + 2, event=info.event_key,
                   match=key, fallbacks=recover)
  # ... after md = scan_video(...):
  if recover:
      accept_recovered_cams(md, recover)
  ```

Other files

* `scripts/recalibrate.py <scan_dir> <match>... | --missing`:
  * seeds the event priors from the scans of the same events, then re-runs only the missing end views
    from `data/videos/<match>.mp4`;
  * gates each candidate and saves with `pipeline._save_scan`, replacing `scan.pkl` atomically;
  * writes `calib_recovery.json` beside each recovered scan;
  * is idempotent: a second run leaves every scan byte-identical.
* `tests/test_core.py`: 2 synthetic tests, for pan poses with a misplaced tag and for the prior's
  median and exclusions. All 16 tests pass.

### The recovered views

All numbers are on the scans copied to `data/exp_calib/before`. The rescan_v2 copy is used where it
exists; the dry-run on `output/` gives the same decisions, with gate values of 0.34-0.70 m.

| match | view | tags per frame | method | poses | tag rms | gate median (null) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| cagle_qm32 | red_end | 2 (coplanar) | fixed K, event centre | 42 | 1.37 px | **0.47 m** (2.70) |
| cagle_sf13m1 | red_end | 1-2 (coplanar) | fixed K, event centre | 42 | 0.58 px | **0.70 m** (2.02) |
| cancmp_qm114 | blue_end | 4-7 (+2 outlier tags) | fixed K, refined centre | 42 | 0.84 px | **0.80 m** (4.54) |
| cancmp_qm52 | blue_end | 3-5 | fixed K, refined centre | 43 | 0.80 px | **0.54 m** (2.33) |
| casnv_qm7 | blue_end | 1 | fixed K, event centre | 41 | 0.66 px | **0.34 m** (4.53) |
| cc_qm15 | blue_end | 0-2 (15 of 42 frames) | fixed K, event centre | 14 | 0.49 px | **0.42 m** (5.83) |
| joh_qm14 | blue_end | 0 | none | - | - | stays uncalibrated |

"Null" is the same metric with the main-view robots taken 30 s later, i.e. chance level. In all 6
matches the recovered view agrees with the main view better than that match's own, normally calibrated
other end view does (e.g. qm114 red_end 2.41 m, qm52 red_end 2.73 m, casnv_qm7 red_end 0.92 m).

* **cancmp_qm114.**
  * Tags 3 and 4 (red hub, 11 m away) reproject 25 px off under any consistent pose. They are the
    outliers that pulled the plain fit to f 3001 px.
  * Even with them dropped, two camera heights fit the tags: 0.52 m (rms 0.66 px) and 0.87 m
    (1.46 px). Per-frame PnP with fixed K settles at 0.87 m, which the gate rejected (1.45 m).
  * Starting the joint refine from both centres and keeping the lower tag cost picks 0.52 m, which
    passes (0.80 m). This choice uses the tags only, not the gate.
* **cancmp_qm52.** The event-K start fixes the bad minimum: f 647 px, rms 0.55, gate 0.63 m. The
  fixed-K track agrees slightly better (0.54 m) and is the one kept.

`data/exp_calib/img/recovered_views.jpg` shows each recovered camera at three times in the match.
The projected tag outlines sit on the real tags and the hub openings sit above the hubs.

### Is the fallback right? Leave-one-out on views that calibrate on their own

12 end views of 6 normal matches, one per event plus joh_qm39 (`tools/validate_loo.py`). Each view is
rebuilt from the event prior with its own match excluded, using all tags, only the 2 largest, or only
the largest tag per frame. The robot positions are compared with those from the view's own
calibration (same boxes):

| tags used | position difference vs own calibration (median of per-view medians) | gate pass |
| --- | ---: | ---: |
| all | 0.10 m | 8 / 12 |
| 2 per frame | 0.33 m | 8 / 12 |
| 1 per frame (like casnv_qm7, cc_qm15) | 0.33 m (0.11-0.68 m by view) | 6 / 12 |
| own calibration | - | 8 / 12 |

The fallback reproduces a real calibration to about a third of a metre even from a single tag. The
1 m gate is conservative: its failures here are mostly at joh, where the views' own calibrations fail
it too.

**What the gate measures** (`tools/gate_baseline.py`, all 143 calibrated end views in `output/`):

* median of medians 0.72 m, and 78 % are under 1 m;
* chance level (null) has median 3.5 m.

The handful of normal views near chance level are mostly not end-camera errors. On 6 of 8 checked
(`tools/validate_loo.py --out=validate_suspect.json`) the fallback agrees with the view's own
calibration to 0.1-0.3 m and fails the gate by the same margin,
so the main view or its coverage is what disagrees (e.g. joh_qm14 red_end 5.3 m, cancmp_qm90 and
qm99 red_end around 4 m). One exception, under follow-ups below.

### Before / after: `pipeline.analyse(md, Options())` in memory on the 7 matches

| match | alliance | share err vs fuel OPR | credited launches (from own-end view) | held-out identity |
| --- | --- | ---: | ---: | ---: |
| cagle_qm32 | red | 0.087 -> 0.098 | 65 (0) -> 200 (134) | 97.1 -> 99.9 % |
| cagle_sf13m1 | red | 0.059 -> 0.057 | 37 (0) -> 62 (27) | 85.9 -> 91.0 % |
| cancmp_qm114 | blue | 0.155 -> 0.197 | 44 (0) -> 84 (43) | 81.1 -> 80.1 % |
| cancmp_qm52 | blue | 0.339 -> 0.152 | 56 (0) -> 97 (41) | 88.5 -> 85.7 % |
| casnv_qm7 | blue | 0.556 -> 0.545 | 27 (0) -> 37 (12) | 79.1 -> 89.0 % |
| cc_qm15 | blue | 0.014 -> 0.042 | 31 (0) -> 79 (48) | 78.2 -> 62.2 % |
| joh_qm14 | blue | 0.129 (unchanged) | 163 (0) (unchanged) | 91.5 % (unchanged) |

Summary:

* share error vs fuel OPR (total variation, as in failure-finder c3f), affected alliances:
  0.191 -> 0.174 on average;
  * the 6 recovered: 0.202 -> 0.182, better in 3 and worse in 3. Most of the gain is cancmp_qm52.
  * the other alliance of the same matches: 0.158 -> 0.158.
* credited launches, affected alliances: 423 -> 722 (median 44 -> 84).
* held-out identity (`eval_analysis.py --holdout` logic), affected alliances pooled: 86.1 -> 88.7 %;
  both alliances of the 7 matches: 87.4 -> 88.7 %.
* **OCR.** Uncached bumper reads were made with EasyOCR on the CPU (no GPU) in the four rescan_v2
  scans: 139-184 new reads each.
  * They added no team evidence, so the numbers are the same as with cached reads only.
  * The three v1 scans store no crop images, so no new reads are possible there (or in a real
    reanalyse of `output/`).

**cc_qm15 vs 5026's human scouting** (`score_scouting.py` logic, both alliances, 6 team-matches):

| | before | after |
| --- | ---: | ---: |
| team-match fuel error, mean | 72.4 | **65.9** |
| correlation with the scouting reference | 0.77 | **0.85** |
| balls per scoreboard jump given to the robots scouts saw shooting | 57.5 % | **59.6 %** |

Blue's split moved from 0.25 / 0.42 / 0.34 to 0.22 / 0.41 / 0.37 (4698 / 254 / 5940). The scouts
had 0.04 / 0.50 / 0.47.

### No regressions

* **Calibration path** (`tools/regress_calib.py`). For 13 matches (the 6 normal ones and the 7
  affected), `pipeline.calibrate` was re-run from the video twice: the old call, and the event path
  with priors and recovery.
  * Every view that calibrated before calibrates again. The parameters match the stored camera to 0
    or float noise (at most 5e-6).
  * The only exception is cc_qm26 main, which differs by 5.3. Main goes through identical code in
    both calls, so this is OpenCV run-to-run nondeterminism in the existing plain fit, not this
    change.
  * The event path adds only the recovery candidates for the 7 failed views.
* **Scans.** `recalibrate.py` on the 6 normal matches reports "every end view already calibrated"
  and does not touch their `scan.pkl` (byte-identical, md5).

### Caveats and follow-ups

* **cc_qm15 held-out identity fell 78 -> 62 %** (blue, similar read mass), even though fuel moved
  toward the scouts.
  * Only 15 of its 42 sampled frames show a tag, so poses are up to 36 s apart. The recovered
    positions degrade with distance from a solved pose: gate median 0.40 m within 2 s, 0.65 m beyond
    12 s.
  * Sampling end frames more densely, or trusting a pose only near a solved frame, would help.
* **joh_qm14 has no tags at all.** It could only be placed from robot correspondences with the main
  view, but that would use the same evidence as the gate.
* **Focal-length outliers among accepted plain fits.**
  * 5 of 145 accepted end-view fits have a focal length more than 30 % off their event median.
    cagle red_end alone ranges from 127 to 904 px, because two coplanar tags let focal length trade
    against distance.
  * cagle_qm63 red_end (f 1.8x the median) is wrong: gate 2.59 m. The fallback fixes it (gate
    0.27 m, positions 3 m apart). cagle_qm23 red_end (1.33x) likely too: 1.28 -> 0.81 m.
  * Applying gates (b) + (d) to plain fits would catch these. That is not done, to keep the normal
    path unchanged.
* **Order of operations.** If rescan_v2 scans are later installed over `output/`, re-run
  `recalibrate.py` afterwards. The rescans copy `md.cams` from the scan they start from.
* **Disk.** `data/exp_calib/{before,after}` hold the experiment copies (~1.6 GB); they can be
  deleted.

### Apply to the installed scans

```
.venv/Scripts/python scripts/recalibrate.py output 2026cagle_qm32 2026cagle_sf13m1 2026cancmp_qm114 2026cancmp_qm52 2026casnv_qm7 2026cc_qm15 2026joh_qm14
```

(or `scripts/recalibrate.py output --missing`), then reanalyse those matches, e.g. with
`pipeline.reanalyse(key)` or `scripts/finalize.py`. Runtime is about 10-30 s per view, on the CPU,
plus a quick prior seeding (the caches are already seeded from `output/`).

Tools and logs are in `data/exp_calib/tools/` and `data/exp_calib/*.log|json`.
