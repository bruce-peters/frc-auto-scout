## Two-view launch points: which robot launched a ball, from two cameras at once (2026-10-18, not adopted)

**Question.** A launch arc is tied to the robot box it rises from in one view. Teammates side by side get the
right credit only about 65 % of the time, and a robot hidden behind another one (254 behind 3847 in the Chezy blue
end camera) has no box at all. A ball seen in two views at the same instants is pinned down in 3-D, and every robot
has a fused field position. Does rebuilding the arc in 3-D, following it back to launch height and taking the
robot standing there beat today's single-view link?

**Answer.** No, not with today's calibrations. The 3-D rebuild is correct (exact on synthetic balls), but the three
Chezy views disagree about points in the air by roughly 0.1-0.5 m.

* **Why it fails.** True cross-view pairs fit at 7-15 px instead of the ~2-3 px the tracks allow, so they are
  hard to tell from chance pairs between unrelated tracks. For launches first seen in an end camera (most of
  them), launch points land a median 1.4-1.6 m from the scouted shooter, while the fit itself claims ±0.15 m.
* **Coverage.** 83 % of launch tracks have some cross-view candidate, but most of those are chance pairs. 3.0 % of
  launch tracks (5.4 % of credited launches) have a partner that fits both views to 6 px, and 13 % (22 %) to 10 px.
  Almost all pairs are an end camera with the main camera; the two end cameras share almost no balls (< 1 %).
* **Picks** (scout-judged truth sets of `data/exp_arcs`, today's rule 71.4 %, same-alliance 61.3 %):
  * Taking the two-view robot outright is worse wherever it covers much: −1.0 pp (10 px rule) to −3.3 pp (any fit).
  * Only the strictest rule is neutral: +0.2 pp (CI −0.3 .. +0.8), 4.5 % of cases.
  * As a soft term added to today's score (λ by leave-one-match-out), +0.7 pp (CI +0.1 .. +1.4) on c24 and +0.5 pp
    (−0.1 .. +1.1) on the fresh set. That is a quarter of what the single-view arc check gained (+3.0).
* **Hidden robots.** It does reach robots with no box: on the 1,028 judged launches whose shooter had no box in
  the view, it credits the right one in up to 17 % (today: 0 %). It loses as many elsewhere, so the launches with
  a teammate behind the 2-D winner stay at 59.9 → 58.4-60.6 % (CIs span 0).
* **Fuel vs 5026's scouting** (54 scouted Chezy quals, paired by match): neither arm moves per-team fuel.
  * Strict relink (6 px): video-only MAE +0.08 (CI −0.23 .. +0.40).
  * Soft term (λ = 0.01): −0.08 (−0.33 .. +0.17).
  * 254's burst check is unchanged in both.
* **Recommendation: don't adopt.** Nothing to port. `autoscout/triangulate.py` stays an experimental module.
  The precondition for another try is a consistent 3-D calibration of all three views (below).

**In plain words.** Two cameras looking at the same ball should say exactly where it is. Here they disagree by
about a ball-to-robot distance, because the cameras' positions are only known well along the floor, not in the
air. With that much disagreement the method can't reliably tell which ball in one view is which ball in the
other, and when it does pair them the launch point is about 1.5 m off. So it picks the shooter no better than
today's single-camera rules. A better camera fit makes it clearly better, but still not good enough.

### Coverage (`tools/coverage.py`, 65 Chezy matches)

Launch tracks are the `fuel.is_shot` tracks of every view (55,347). Credited launches are today's links in
`output/<match>/result.json` (15,266). "Partner" is the best candidate under the rule (lowest per-view rms).

| acceptance rule | launch tracks with a partner | credited launches with a partner | blue_end + main | red_end + main | main + an end view | end + end | scored balls with a two-view launch in their credit window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fits both views to 6 px | 3.0 % | 5.4 % | 6.1 % | 4.5 % | 4.1 % | 0.7 % | 22.5 % |
| ... to 10 px | 13.0 % | 22.1 % | 27.1 % | 22.2 % | 12.8 % | 1.3 % | 59.0 % |
| main 8 px, end 4 px (normalised ≤ 2) | 18.7 % | 34.2 % | 43.0 % | 39.2 % | 14.5 % | 0.9 % | 71.9 % |
| mutual best pair (each track the other's best) | 2.8 % | 6.7 % | 5.0 % | 4.9 % | 9.9 % | 0.4 % | 20.1 % |
| any candidate (mostly chance pairs) | 83.4 % | 97.9 % | 97.1 % | 96.9 % | 92.9 % | 2.9 % | 92.9 % |

* The view-pair columns are the share of that anchor view's credited launches (blue_end 4,962, red_end 6,019, main
  4,285).
* "Scored balls" counts a counter jump as covered if any credited launch of that alliance with a partner lies
  0.8-5.5 s before it. That is an upper bound on how much credit the method could touch.
* With the landmark main camera (diagnostic, below) the 6 px share of credited launches rises to 12.4 %, and the
  10 px share to 38.1 %.

### Are the pairs real? (cc_qm24 unless noted; `tools/selfcal_test.py`, `tools/long_pairs.py`, `tools/diag_offset.py`)

* **The code is right** (`tools/synth_check.py`). Synthetic balls through the real qm24 blue_end and main cameras
  are recovered to 0.005 px and launch points to < 1 mm. With 2 px of image noise the launch points are still
  within 0.4-2.5 cm.
* **Chance level.** Shifting every main track by 7.007 s (a whole number of frames) keeps the search identical but
  removes every true pair. qm24 blue_end launch tracks with a main partner (least-squares fit on the common
  instants, every candidate within 0.6 m):

  | best partner fits to | 4 px | 5 px | 6 px | 8 px |
  | --- | ---: | ---: | ---: | ---: |
  | real (of 298) | 17 | 36 | 51 | 97 |
  | main shifted 7 s | 4 | 6 | 17 | 39 |

  There is signal, but a third to a half of the accepted pairs at 6-8 px would be chance.
* **Not a time offset.**
  * Shifting the main view by −417 to +417 ms (whole frames and half frames) gives a flat count of good pairs,
    with no peak at any offset.
  * A rolling-shutter model (±16 ms across the image, per view) is flat too.
* **It is the calibration.** `data/exp_maincal` found that tag-only fits of the Chezy main camera put the hub
  openings far above the hub tops. The hand-checked landmark camera there (f 1423 px, centre at y 21.5 m and
  6.5 m up) differs from the tag-only fit (f 2141 px, y 28.7 m, 11.9 m up).
  * Posing that landmark camera per match (rotation only, so it reproduces the stored camera's image of the four
    tags; `tools/lmcam.py`) helps clearly but not enough.
  * Pairs over ≥ 6 common instants, qm24:

    | main camera | end tracks with a ≥ 6-instant candidate | best fit ≤ 6 px | ≤ 10 px | median best rms, end / main view |
    | --- | ---: | ---: | ---: | --- |
    | stored (tag-only) | 257 | 7 (chance 0) | 40 (chance 8) | 11.2 / 14.4 px |
    | landmark | 261 | 14 (chance 5) | 68 (chance 16) | 7.2 / 13.5 px |

  * A local search over a small correction of the stored main camera finds nothing: rotation, centre, focal
    length and time offset, 8 parameters, Nelder-Mead, end cameras fixed. It reaches 12 instead of 6 anchors
    at ≤ 4 px. The real error is a large focal-length / distance / tilt trade-off that four coplanar tags can't
    fix, and the low end cameras share part of it.
* **Launch point vs the scouted shooter** (hidden truth set, best partner with normalised cost ≤ 2):

  | launch seen first in / partner | n | median distance to the shooter's fused position | landmark main camera | to the nearest other candidate |
  | --- | ---: | ---: | ---: | ---: |
  | blue_end / main | 640 | 1.37 m | 1.08 m | 3.0 m |
  | red_end / main | 784 | 1.57 m | 0.90 m | 3.6 m |
  | main / blue_end | 116 | 0.59 m | 0.64 m | 3.1 m |
  | main / red_end | 97 | 0.83 m | 0.76 m | 3.8 m |

  * The fit's own launch-point uncertainty is ±0.15 m. The spread of the error is ~2 m (sd), with heavy tails
    from wrong pairs.
  * The robots' fused positions come from the same tag-only cameras, so part of the error is theirs (the far
    side of the field is compressed by ~1.5 m, `data/exp_maincal`).

### Pick accuracy on the scout-judged truth sets (`tools/eval_mv.py`, `tools/eval_soft.py`, `tools/eval_same.py`)

Same sets as `data/exp_arcs`:

* **c24** is `data/exp_audit/c24_truth_cases.json`: 1,084 contested launches in 59 matches. Today's rule gets
  71.4 % (same-alliance 61.3 %).
* **fresh** is the c24 construction on the arc-log labels (993; today 73.5 %).
* **hidden** is every box-linked launch with its boxes plus the box-less teammates that have a fused position
  (4,606; today 72.9 %).

Every case found its pass-1 launch track. **Hard pick:** where a two-view fit is accepted, the robot is the
candidate whose fused position is nearest the launch point. The gate is χ²(2) 99 % with σ_robot 0.45 m and a
margin of 2; otherwise today's pick stands. Stored (zero-context) cameras:

| acceptance rule | c24: covered | two-view vs today, on covered | all c24, diff [95 % CI] | fresh: diff [95 % CI] | hidden: diff [95 % CI] |
| --- | ---: | --- | --- | --- | --- |
| any fit | 57.1 % | 66.9 vs 72.7 % | −3.3 [−5.7, −1.1] | −3.9 [−6.5, −1.2] | −0.4 [−1.4, +0.4] |
| fits both views to 15 px | 31.3 % | 67.6 vs 72.0 % | −1.4 [−2.8, +0.1] | −1.7 [−3.4, +0.0] | −0.2 [−1.0, +0.5] |
| ... to 10 px | 15.9 % | 68.0 vs 74.4 % | −1.0 [−2.1, +0.1] | −0.8 [−2.0, +0.4] | −0.0 [−0.5, +0.4] |
| ... to 6 px | 4.5 % | 77.6 vs 73.5 % | +0.2 [−0.3, +0.8] | +0.4 [−0.1, +0.9] | +0.0 [−0.2, +0.2] |
| main 8 px, end 4 px | 24.4 % | 67.0 vs 74.2 % | −1.8 [−3.1, −0.4] | −2.1 [−3.8, −0.4] | −0.1 [−0.7, +0.4] |
| ... ≤ 1.5 and 1.5× better than the next candidate | 8.1 % | 69.3 vs 77.3 % | −0.6 [−1.4, +0.1] | −0.4 [−1.2, +0.4] | −0.0 [−0.3, +0.2] |
| mutual best pair | 4.2 % | 78.3 vs 80.4 % | −0.1 [−0.6, +0.3] | −0.1 [−0.7, +0.4] | +0.2 [+0.0, +0.3] |
| any fit, no gate (always the nearest) | 98.6 % | 64.3 vs 71.7 % | −7.4 [−11.2, −4.0] | −8.4 [−11.9, −4.9] | −1.2 [−2.8, +0.4] |

* **Gate settings don't change this.** On c24, σ_robot 0.3 / 0.45 / 0.7 m × margin 0 / 2 / 4 gives +0.2 .. +0.3 pp
  for the 6 px rule, −0.6 .. −1.2 for 10 px and −1.1 .. −3.0 for the normalised rule.
* **Same-alliance** (c24, 721 cases; today 61.3 %): 6 px rule 61.4 % (−0.5 .. +0.9), 10 px rule 60.2 %, normalised
  rule 59.9 %.
  * Cross-alliance picks drop with the looser rules (91.5 → 89.0 %), because the two-view pick ignores the zone
    rule.
  * The fresh set agrees: same-alliance 63.5 → 64.0 % at 6 px, 62.4 % at 10 px.
* **Soft term:** today's score + λ · Mahalanobis² of each box's robot from the launch point, λ by leave-one-match-out:

  | rule | c24 covered | c24 LOMO, diff [CI] | fresh LOMO, diff [CI] |
  | --- | ---: | --- | --- |
  | fits to 15 px | 54.6 % | 72.5 %, +1.1 [−0.1, +2.4] | 74.0 %, +0.5 [−0.3, +1.2] |
  | fits to 10 px | 24.8 % | 71.7 %, +0.3 [−0.2, +0.7] | 73.7 %, +0.2 [−0.3, +0.7] |
  | fits to 6 px | 6.2 % | 71.5 %, +0.1 [−0.4, +0.6] | 74.0 %, +0.5 [+0.0, +1.0] |
  | main 8 px, end 4 px (λ = 0.01 in every fold) | 38.1 % | 72.1 %, +0.7 [+0.1, +1.4] | 74.0 %, +0.5 [−0.1, +1.1] |

* **Landmark main camera** (diagnostic only: the landmarks were picked by hand once for the event, so it breaks
  the zero-context rule; robot positions still come from the stored cameras):
  * Hard pick, 10 px rule: c24 −0.4 (−1.9 .. +1.2), fresh +0.1, hidden +0.8 (+0.1 .. +1.4), at 30 % coverage.
  * Soft term, 10 px rule: c24 +0.7 (−0.7 .. +2.1), fresh +0.8 (−0.7 .. +2.3).
  * Better than the stored camera and no longer harmful, but no rule clears its CI on both contested sets.

**Hidden robots** (`tools/eval_hidden_mv.py`, hidden set; picks may go to box-less teammates):

| subset | n | today | two-view, 10 px, gated | two-view, any fit, nearest | box-less picks right (any fit, nearest) |
| --- | ---: | ---: | --- | --- | --- |
| all | 4,606 | 72.9 % | 72.9 % (−0.5 .. +0.4) | 71.7 % (−2.8 .. +0.4) | 176 of 452 |
| teammate behind the 2-D winner | 807 | 59.9 % | 60.6 % (−0.7 .. +2.2) | 58.4 % (−7.8 .. +4.4) | 114 of 280 |
| ... end cameras | 758 | 59.9 % | 60.3 % (−1.0 .. +1.8) | 59.1 % (−7.1 .. +5.1) | 108 of 259 |
| shooter has no box in the view | 1,028 | 0 % | 1.8 % | 17.1 % | 176 of 226 |

* The two-view point does find box-less shooters: 176 of its 226 box-less picks on the last row are right. On the
  full set, though, 276 of its 452 box-less picks are wrong, and the full set does not improve.
* **cc_qm24 70-74 s** (254 behind 3847, `tools/qm24_window.py`): today credits only 9 launches in the window, 5
  in blue_end (3847 4, 254 1).
  * At 10 px one of the five has a partner, and its pick fails the gate. Of all 10 blue_end launch tracks in the
    window, two pair at 10 px; both picks fail the gate (nearest robots 254 and 5199).
  * Any fit, nearest: blue_end 254 2, 3847 1, 5199 2. Too few to matter. The volley's launches are lost upstream
    in the tracker.

### Against 5026's scouting (`tools/ab_mv.py`, 54 scouted Chezy quals)

All arms ran in one process on the same scans: pipeline.py 21a4195ff904, credit model 553b145d9546, triangulate.py
63c5e4db4188. The baseline is the unhooked analysis and reproduces today's 47.99 / 40.89. Paired bootstrap over
matches, 95 % CI:

| arm | video-only fuel MAE | video-only corr | top scorer (video only) | best-estimate MAE | burst |
| --- | ---: | ---: | ---: | ---: | ---: |
| today | 47.99 | 0.830 | 63 % | 40.89 | 63.8 % |
| strict relink (fits both views to 6 px, gated) | 48.07 | 0.830 | 62 % | 41.01 | 63.9 % |
| paired diff | +0.08 [−0.23, +0.40] | −0.000 [−0.003, +0.002] | −1.0 pp [−1.7, 0] | +0.12 [−0.18, +0.45] | |
| soft term (main 8 / end 4 px, λ = 0.01 by LOMO) | 47.91 | 0.830 | 64 % | 40.87 | 63.8 % |
| paired diff | −0.08 [−0.33, +0.17] | +0.000 [−0.001, +0.002] | +1.0 pp [0, +1.7] | −0.03 [−0.22, +0.16] | |

* **Strict relink** (`Relink`, inserted just before the "no robot" exit of the launch loop).
  * Whether a launch counts is decided as today; only its robot can change, to any of the six robots, boxed or
    not.
  * Of 12,707 credited launches, 680 have an accepted fit and 498 pass the gate. 96 change robot.
  * 16 matches better, 18 worse.
* **Soft term** (`SoftTerm`, at the launch-to-box candidate step via `data/exp_arcs/tools/arc_hook.py`).
  * Gate decided on the 2-D winner, as the arc work recommends.
  * It changes 49 of the 1,025 contested launches that have a fit. 18 matches better, 10 worse.
* **254 burst check** (`team_volleys.py 254 --out ab/<arm>`), the same in all three arms:
  * 1,159 jumps, 3,846 balls on its alliance's counter; scouts give 254 1,606.
  * Credit gives it 1,772 (today and soft) or 1,768 (relink).
  * 63 jumps / 152 balls where scouts say ≥ 60 % 254 and credit is > 0.3 lower.
* **Arms not run.** A first run with six arms was stopped at the main session's request to free shared CPU. Its
  partial log is `ab_log_stopped_6arms.txt`. The other four arms were the looser relink, a second soft rule and
  two landmark-camera arms; none had a truth-set gain with a CI above zero on both contested sets.

### Recommendation

**Don't adopt. Port nothing into `pipeline.py` / `attribution.py`.**

* On today's calibrations no variant beats the 2-D link on both contested truth sets with a CI above zero, and
  per-team fuel doesn't move.
* `autoscout/triangulate.py` stays as an experimental module, not imported by the pipeline.
* **If a consistent 3-D calibration exists later**, the piece to port is the soft term, not the outright relink:
  * Code: `triangulate.MultiView.candidates` → `fit_pair` → `launch_point`, then λ · Mahalanobis² per close box,
    added to the score in `pipeline.analyse` step (1) after the zone rule.
  * The launch gate stays decided on the 2-D winner.
  * Reference implementation: `SoftTerm` in `data/exp_multiview/tools/ab_mv.py`, with rule "main 8 / end 4 px"
    and λ re-chosen by leave-one-match-out.
  * Re-test first with `mv_pairs.py` → `eval_mv.py` / `eval_soft.py`.

### Method (`autoscout/triangulate.py`, tools in `data/exp_multiview/tools/`)

* **Geometry.**
  * The three views share one frame grid: the fuel scan runs every 2nd frame for the whole broadcast frame, so
    all views have the same timestamps.
  * Each track point becomes a normalised image point (`cv2.undistortPoints`) and a world ray. The pose at its
    own time comes from `CameraTrack.at(t)`, and the intrinsics are fixed per view.
* **`MultiView.candidates(va, i, vb)`.**
  * Tracks of view vb seen at ≥ 3 of the anchor's instants.
  * Median closest distance between the two views' rays ≤ 1 m.
  * Both depths positive, and the midpoints inside the field box (z −0.3 .. 8 m). At most 12 per view.
* **`MultiView.fit_pair` → `fit_linear`.** A closed-form ballistic fit.
  * X(s) = P0 + V0 s − g s²/2. Each sighting gives two equations linear in (P0, V0): the projection constraint.
  * Each equation is weighted by f / depth, so residuals are image px, and re-weighted 4 times (soft-L1, 3 px).
  * It uses the anchor's points up to the end of the overlap + 0.1 s and the partner's inside the overlap ± 0.1 s.
  * About 0.3 ms per fit, ~80 times faster than `least_squares`. The answer is the same to a few px (`fit_path`
    is kept as the reference).
* **`launch_point`.**
  * The earlier time the path is at 0.55 m (rising, vz > 0.3 m/s, at most 1 s before or 0.3 s after the first
    sighting), and the field point there.
  * Covariance = the fit's parameter covariance (σ² = robust residual variance, floor 1.5 px) pushed through by
    finite differences, plus a launch-height spread of 0.2 m along the path.
* **`pick_robot`.**
  * Mahalanobis² of each robot's fused position under the launch covariance + (0.45 m)² I.
  * The pick must pass χ²(2) 99 % (9.21) and beat the next robot by 2.
  * Robots need no box in any view: the A/B uses the fused team tracks (`team_arr`), the truth sets the logged
    `xy_fused` of each candidate.
* **Zero-context.** Each fit uses only that match's own tracks, camera poses and robot tracks. The constants are
  global and were not tuned per event. The landmark camera is the only exception, and it is used as a
  diagnostic only.
* **Passes.**
  1. `mv_pairs.py`: every candidate pair of every launch track, one scan at a time, read and closed at once.
     About 15 s per match. Output `pairs/<match>.pkl`, or `pairs_lm/` with `--lm`.
  2. `eval_mv.py`, `eval_soft.py`, `eval_same.py`, `eval_hidden_mv.py`, `coverage.py`, `diag_*.py`: offline,
     from the pairs and the `data/exp_arcs` logs.
  3. `ab_mv.py`: in-memory A/B (above). About 13 s per arm per match, CPU only.
* **Constants not tuned to the truth sets.** The acceptance rules are a small fixed grid; σ_robot, the gate and
  the margin were set before scoring. λ of the soft term is chosen by leave-one-match-out.

### What would make it work: one consistent 3-D calibration of the three views

**How good it must be.** Side-by-side robots stand ~0.9-1.2 m apart (centre to centre). With a launch-point error
σ along the line between them, the pick is right Φ(0.5 m / σ) of the time:

| σ | 0.25 m | 0.35 m | 0.5 m | today's two-view (heavy-tailed, median ~1.5 m) |
| --- | ---: | ---: | ---: | ---: |
| pick right | 98 % | 92 % | 84 % | ~65 % on covered contested cases |

To clearly beat today's 61 % same-alliance, σ should be ≤ 0.35 m in total. The fused robot position already costs
~0.2-0.3 m of that, so the cameras must agree on a point in the air to ~0.1-0.15 m:

* the two views' rays meet within ~0.1 m at common instants;
* joint-fit residuals of ~3-4 px in the main view (70-85 px/m at 20-25 m) and ~3-5 px in an end view;
* long pairs (≥ 6 instants) fitting at ≤ 4 px for most end-view launch tracks that have a candidate, far above the
  7 s time-shift chance level.

Today the views agree to ~0.2-0.5 m: 11-14 px median on long pairs. The landmark main camera alone gets the end
views to 7 px and the median launch error from 1.4-1.6 to 0.9-1.1 m, so the end cameras need fixing too.

**What it would take** (automatic, per match, zero-context):

1. **Unknowns.**
   * Per end camera: focal length, k1 and a pose (8; these cameras barely move).
   * Main camera: one focal length, k1 and centre, plus a rotation per 4 s window, as `maincal.fit_static`.
   * About 25-30 numbers per match beyond the main camera's window rotations.
2. **Constraints.** Each kind fixes what the others can't:
   * **Tags**, as today. They fix each view's image of its tags but not the main camera's focal length,
     distance and tilt (four coplanar tags at one height).
   * **Floor.** Robot box bottoms from different views must land on the same carpet point at the same instant.
     That ties the views together on the ground (the `maincal` floor term, but between views, not just "on the
     field").
   * **Points above the floor.**
     * Balls dropping into a hub vanish at the opening (1.83 m). On qm24 and qm1 their track ends cluster
       0.2-0.35 m below the projected opening centre in every view (`tools/hub_ends.py`). That is a free, nearly
       known 3-D landmark in all three views.
     * The hub's top corners and the tower uprights would serve too, if detected automatically.
   * **Shared ball tracks.** These are the strongest constraint on the air geometry. Each long cross-view pair adds
     2 × (instants) − 6 equations, and gravity fixes scale and the vertical. A match has ~250 end-view launch
     tracks with a ≥ 6-instant main candidate.
3. **Solve.** Robust bundle adjustment:
   * `scipy.optimize.least_squares`, soft-L1, sparse Jacobian. Unknowns are the camera parameters plus 6 path
     numbers per ball.
   * Pairing is re-done as the cameras improve (EM / RANSAC): start from long, mutual-best pairs only; refit;
     re-pair with a tighter threshold; repeat. Chance pairs get down-weighted by the robust loss and dropped at
     the next re-pairing.
4. **Checks.**
   * The time-shift null (`selfcal_test.py --shift`) must stay flat while real ≤ 4 px pairs multiply.
   * Robots re-fused with the new cameras must keep their cross-view agreement and reach the far wall (the
     `data/exp_maincal` test).
   * Then rerun `mv_pairs.py` → `eval_mv.py` / `eval_soft.py` unchanged.
5. **Cost.** A new module of a few hundred lines, and ~1-2 min of CPU per match for the adjustment. The
   per-match local search tried here (8 parameters on the main camera only, end cameras fixed) shows that a
   small correction is not enough: it must move focal length and distance together and include the end
   cameras.

* **The robots' fused positions must come from the same corrected cameras**, or the launch point and the robot are
  in different frames.
* **Recall still caps it.** In the qm24 window the volley's launches are missing before any linking (tracker work).
