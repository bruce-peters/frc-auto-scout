## 3-D arc check for which robot launched a ball (2026-10-18, not adopted)

**Question.** A launch arc is tied to the robot box it rises from. When two boxes overlap in one view, the 2-D
box rule (`dist + TOP_TIEBREAK * top + LINK_X_TIE * xc`) is close to a coin flip for teammates. In the end
cameras a near robot's box also hides a far shooter. Robots have fused field positions and the hub is a known
3-D target. Does a single-view ballistic fit pick the shooter better?

**Answer.** It picks the right robot more often, but per-team fuel does not move.

* **Links.** On the failure finder's 1,084 scout-judged launches, adding the arc term to today's score takes
  pick accuracy from **71.4 % to 74.4 %** (+3.0 pp, 95 % CI +1.0 .. +4.9). Same-alliance picks go from
  61.3 % to 65.3 %.
  * A truth set rebuilt the same way from today's labels gives the same answer: 73.5 → 76.5 % (+3.0,
    +1.1 .. +5.2).
  * Scored as what the pipeline would credit, the gain holds only if the launch gate is decided on today's 2-D
    winner and the arc term just picks the robot: +3.0 and +2.8 pp. As a plain score term it is −1.0 / −1.8 pp.
    The section on the launch gate explains why.
  * The 62.5 % quoted for the current rule predates LINK_X_TIE and the zone rule. Today's rule gets 71.4 % on
    the same cases, so that is the bar.
* **Fuel vs 5026's scouting** (54 Chezy quals, paired by match): video-only fuel MAE −0.04 (CI −0.90 .. +0.80).
  25 matches are better and 26 worse. The links change, but the per-team totals don't.
* **The one-parameter fit as specified does not work.** With the launch fixed 0.55 m above the robot and the
  landing fixed at the hub centre, the true shooter's arc fits no better than a teammate's (+0.2 pp, CI
  −2.6 .. +2.7). The gain needs the launch and landing points free inside priors (below).
* **Recommendation: don't adopt into the fuel pipeline.** It adds about 2.5 s of CPU per match and a 7-parameter
  fitter with tuned constants, for no fuel gain. Port it only if per-launch links matter on their own (the
  inspector, audits), as the zone rule was. The port spec is at the end.

* **With split tracks** (`SPLIT_TRACKS` + `GATE_TRACE`, follow-up below), the arc term is just as flat:
  video-only MAE +0.08 (CI −0.49 .. +0.65) against split + traced gate. Letting it credit robots that have no
  box in the view (254 behind 3847) doesn't reach significance either.

**In plain words.** When two robots overlap on screen, the 3-D check asks which one the ball could have
flown from to reach its hub. It picks the right robot about 3 more times in 100. Per-team fuel ends up the
same, probably because a swap one way between teammates is often matched by a swap the other way (cycle 24
saw the same with every overlap rule).

### Pick accuracy on the scout-judged truth sets (`tools/eval_truth.py`)

Pick among the case's candidate boxes; zone rule first for cross-alliance launches in teleop.

| rule | c24 set: all | same-alliance | cross | fresh set: all | same-alliance |
| --- | ---: | ---: | ---: | ---: | ---: |
| old: dist + 0.5 top (the c24 pick) | 62.5 % | 57.8 % | 71.6 % | 67.0 % | 61.2 % |
| old + zone rule | 69.0 % | 57.8 % | 91.2 % | 71.9 % | 61.2 % |
| **today: + LINK_X_TIE 1, LINK_X_CROSS, zone rule** | **71.4 %** | **61.3 %** | 91.5 % | **73.5 %** | **63.5 %** |
| arc fit alone, one-parameter (hard) | 71.8 % | 62.1 % | 90.9 % | 74.0 % | 64.5 % |
| arc fit alone, soft priors | 72.7 % | 63.0 % | 92.0 % | 74.6 % | 65.5 % |
| today + 0.01 × hard cost (LOMO λ) | 71.6 % | 61.9 % | 90.9 % | 74.9 % | 65.8 % |
| **today + 0.3 × soft cost (LOMO λ)** | **74.4 %** | **65.3 %** | 92.3 % | **76.5 %** | **68.3 %** |

* **Sets.** c24 = `data/exp_audit/c24_truth_cases.json`: 1,084 cases in 59 matches (721 same-alliance, 363
  cross).
  * Every case matched its logged launch by time, box area and old score, and the old rule reproduces the c24
    pick in 100 %.
  * fresh = the c24 construction on today's labels: 993 cases in 58 matches.
  * The two sets share most launches, so they agree on labels, not on independent arcs.
* **λ.** Leave-one-match-out picked λ = 0.3 in every fold of both sets. Accuracy is flat from 0.2 to 0.5.
* **Robustness.** Every soft-prior setting tried gives +1.8 to +3.0 pp (LOMO λ), each with a CI above 0:
  σ_px 4 or 8, σ_xy 0.3, 0.5 or 0.8 m, σ_hub 0.4 or 0.8 m, σ_z 0.25 or 0.4 m, the first 8 points only, the
  box's own position instead of the fused one (+1.8, the lowest). The default is not the best of these.
* **Where it helps (same-alliance, c24).**
  * End cameras, arcs 0.2-0.5 s: 59.3 → 65.1 % (n = 415).
  * Main camera: 55.6 → 63.0 % (0.2-0.5 s) and 61.8 → 64.7 % (≥ 0.5 s).
  * End-camera arcs shorter than 0.2 s: 72.5 → 71.4 %, no gain, as the single-view argument predicts.

#### The launch gate decides how the term can be used

The table above ignores `pipeline._box_shot_ok`. In the pipeline the gate checks only the winning box, and a
winner that fails it drops the launch. The next table scores what the pipeline would credit (dropped = wrong):

| arc term wired in as | c24: credited right | dropped | fresh: credited right | dropped | fresh diff vs today |
| --- | ---: | ---: | ---: | ---: | --- |
| today (no term) | 64.4 % | 7.9 % | 73.3 % | 0.0 % | – |
| plain score term, gate checks the new winner | 63.4 % | 13.3 % | 71.5 % | 7.2 % | −1.8 [−3.1, −0.5] |
| term may only move the pick to a box that passes the gate | 64.9 % | 8.1 % | 73.7 % | 0.1 % | +0.4 [−0.4, +1.3] |
| **gate decided on the 2-D winner, term only picks the robot** | **67.4 %** | 7.9 % | **76.1 %** | 0.0 % | **+2.8 [+0.8, +5.0]** |

* Of the picks the term moves, 73 % (c24) and 78 % (fresh) land on a box that fails its own gate. Those moves
  are right 65-69 % of the time; the 2-D pick they replace was right 28-32 %.
  * So the gate often rejects the true shooter's box. Why is not established: the box moved to is the smaller
    one in only 54-60 % of these cases, so it is not simply far robots.
  * The only version that keeps the gain is "shot or not on the 2-D winner, robot from the arc".
* Side finding: today's rule drops 7.9 % of the c24 launches, which the old rule had credited. LINK_X or the
  zone rule moved the pick to a box that fails the gate. Not checked further.

### Against 5026's scouting (`tools/ab_arcs.py`, 54 scouted Chezy quals)

Both arms ran in the same process, on the same scans and current defaults (LINK_X_TIE 1, LINK_X_CROSS True,
`models/attr_model.json` sha a484e13bc897, pipeline.py 6223a90edba4). The baseline arm is the unhooked
analysis. Paired bootstrap over matches, 95 % CI:

| arm | video-only fuel MAE | video-only corr | top scorer (video only) | best-estimate MAE | burst |
| --- | ---: | ---: | ---: | ---: | ---: |
| today | 47.99 | 0.830 | 63 % | 43.12 | 63.5 % |
| **+ arc, gate on the 2-D winner** | 47.95 | 0.831 | 64 % | 43.20 | 63.6 % |
| paired diff | −0.04 [−0.90, +0.80] | +0.001 [−0.006, +0.009] | +1.0 pp [0, +1.7] | +0.08 [−0.57, +0.71] | |
| + arc, plain term (gate on the new winner) | 47.59 | 0.832 | 64 % | 42.77 | 63.6 % |
| paired diff | −0.40 [−1.10, +0.27] | +0.003 [−0.003, +0.010] | +1.0 pp [0, +1.7] | −0.35 [−0.90, +0.16] | |

* **Gate on the 2-D winner** (the version that links better): 25 matches better, 26 worse. It re-links 452 of
  the 3,164 launches it is applied to (14 %). Box-linked and gated launch counts are unchanged (within 8 of
  12.5 k and 20.6 k).
* **The plain term** leans better on fuel (30 better, 21 worse) but links worse. It drops 212 box-linked launches
  (12,533 → 12,321, −1.7 %; 234 more gated), and on the fresh truth set it is −1.8 pp. Its small fuel gain is from dropping contested launches, not
  from better links. c24 saw the same: no overlap rule moved shares.
* **An earlier run is void.** Its baseline came from pass 1, which loaded the credit model before the main
  session replaced `models/attr_model.json` (11:09), while the arms used the new model. Its files are in
  `ab_INVALID_confounded/`. Its video-only column happens to equal the valid run (the pure-video split doesn't
  use the model).

### Method (`autoscout/arcs.py`, tools in `data/exp_arcs/tools/`)

* **`arcs.arc_fit(track_pts, cam, cand_xy, hub_xyz, launch_z=0.55, max_pts=None, prior=None)`**
  * One view, one camera pose at the launch (`CameraTrack.at(t0)`), view-local px.
  * Drag-free ball from L = (x, y, launch_z) to H = (hub x, hub y, 1.829 m) in flight time T. The velocity is
    (H − L) / T + (0, 0, g T / 2).
  * Free parameters are T in [0.55, 2.6] s and the lead, the launch-to-first-sighting time, in [0, 0.5] s.
  * Fit: a 28 × 16 grid over (T, lead), then `least_squares` with soft-L1 (6 px). The result is the mean
    point cost.
* **`prior=arcs.SOFT_PRIOR`** frees the launch point (dx, dy, dz) and the landing point in the opening (hx, hy).
  * Priors: σ_xy 0.5 m and σ_z 0.25 m around the robot, σ_hub 0.4 m around the hub centre. Image residuals
    are in units of σ_px 4 px.
  * The MAP cost is soft-L1 summed over points plus the prior terms. The one-parameter fit is the start.
  * Why it is needed: with the launch fixed, the true shooter's arc misses by a median 25 px. The first point
    alone is off by 33 px, and the lead sits at its 0 bound in 39 % of fits. The tracks themselves are smooth
    (2.4 px about their own parabola). Robot position error, shooter offset and height swamp the depth signal
    until they are allowed to move.
  * Synthetic check (`tools/synth_check.py`, real Chezy blue_end camera, exact positions): the true launcher
    fits at 2-3 px RMS and a decoy on the same line of sight at 6-55 px, even for 0.13 s arcs.
* **Geometry.** Chezy's end cameras sit at the field corners only about 0.65 m up. A near robot's box top and a
  far one's line up at the same image height, which defeats the 2-D top rule. Gravity curves an arc by
  f · g / depth px/s² (about 2,500 px/s² at 3 m in the end views), and that carries depth.
* **Candidates.**
  * The boxes within the link distance (dist ≤ 0.6) after the zone rule.
  * Position: the box's segment's fused (Kalman) position at the sighting nearest t0, else the box's own
    back-projection.
  * Target: the box's alliance hub.
* **Likelihood ratio.** score += λ · (cost_c − min cost), λ = 0.3. Every candidate is fitted; nothing is
  reconstructed from a single view.
* **Zero-context rule.** Each fit uses only that launch's arc, camera and robot positions. The constants (the
  priors, λ = 0.3) are global, chosen by leave-one-match-out on Chezy, like LINK_X_TIE.
* **Passes.**
  1. `arc_log.py`: `arc_hook.build()` compiles `pipeline.analyse` with one hook call inserted before the zone
     rule, as `pipeline.analyse_arc` in memory. The Logger copy is identical to `pipeline.analyse` (checked on
     cc_qm24). It logs every launch with two or more close candidates (7,770 in 65 matches, 16,690 boxes) to
     `log/<match>.pkl`.
  2. `eval_truth.py`: truth-set scoring from the logs, with fit caches `fits_c24.pkl` / `fits_fresh.pkl`.
  3. `ab_arcs.py`: in-memory A/B, one scan at a time, read with `fh.read()` and closed at once.
     `arc_hook.install_gate_wrapper()` provides the gate on the 2-D winner.
* **Cost.** About 23 ms per candidate fit, 50 ms per contested launch: 2.5 s per match for the gate on the 2-D
  winner (it only fits launches that pass the gate, about 49 per match), 4.6 s for the plain term.

### If it is ported anyway (links only)

In `pipeline.analyse` step (1), after the `cands` loop and the zone rule:

1. Find the winner `best` as today.
2. Only if `best` passes `_box_shot_ok` and two or more pool candidates have `dist <= 0.6`:
   1. For each such candidate, compute `xy` = its segment's fused point nearest t0 (else `d.xy`) and
      `hub = (*hub_center(d.alliance), HUB_OPENING_HEIGHT)`.
   2. Fit: `fits = arcs.arc_fit(tr.pts, cam, xys, hubs, prior=arcs.SOFT_PRIOR)`.
   3. Add `0.3 * (fit["cost"] - min cost)` to each score. A missing fit gets the worst cost.
   4. Link to the new argmin without re-running the gate on it.
3. The reference implementation is `data/exp_arcs/tools/arc_hook.py` `ArcTerm(W=0.3, SIGMA2=1, prior="soft",
   gate_base=True)`. `arcs.py` is sha a8deafa1a224.

### Leads this turned up

* **The launch gate rejects true shooters' boxes.** 73-78 % of the arc term's re-links land on a box that
  fails `_box_shot_ok`, and they are right about twice as often as the pick they replace.
  * The gate tests only the first sighting and the rise against one box, with thresholds in box heights
    (LAUNCH_ABOVE_MAX 0.45, SIDE_MARGIN 0.2, RISE_ABOVE_MIN 0.6, INBOX_MAX 0.5). Which test fails here has not
    been checked.
  * Worth a look on launch recall: the gate decides whether a contested launch counts at all, which may matter
    more than which teammate gets it.
* **LINK_X and the gate.** Today's rule drops 7.9 % of the c24 launches the old rule credited. The cause is
  the same interaction.

### Follow-up: split tracks and hidden robots (`SPLIT_TRACKS`, `GATE_TRACE`; tools `ab_arcs.py`, `split_diag.py`, `hidden_log.py`, `eval_hidden.py`)

The main session's split tracks (`data/exp_split`) recover 254's dense-volley launches in cc_qm24. 2-D linking
then drops them at the gate or gives them to 3847 (in front) and 5507 (a non-shooter). Can the arc check decide
those? The pipeline constants were set in memory per arm; pipeline.py was not edited.

**Scouting A/B** (54 quals, all four arms in one process: pipeline.py 21a4195ff904, credit model 553b145d9546):

| arm | video-only fuel MAE | vs today [95 % CI] | vs split + traced gate | best-estimate MAE | burst |
| --- | ---: | --- | --- | ---: | ---: |
| today | 47.99 | – | | 40.89 | 63.8 % |
| split + traced gate | 51.10 | +3.11 [−1.35, +7.71] | – | 42.95 | 66.3 % |
| ... + arc term, gate on the 2-D winner | 51.18 | +3.19 [−1.41, +7.85] | +0.08 [−0.49, +0.65] | 42.98 | 66.3 % |
| ... + arc term, plain | 50.99 | +3.00 [−1.57, +7.68] | −0.11 [−0.71, +0.43] | 42.85 | 66.4 % |

* The split arm reproduces the main session's 51.1. The arc term doesn't change it in either form (24 / 29 and
  27 / 27 matches better / worse).
* **Truth set from the split-track logs** (`log_split/`, the fresh construction, 7,715 contested launches): the
  arc term adds +0.1 pp (CI −1.1 .. +1.3).
  * Split pieces are short: 1,341 same-alliance cases are main-camera arcs under 0.2 s, where the arc fit
    alone gets 57.4 % against the 2-D rule's 62.1 %.
  * A piece first seen mid-air can be fitted from either robot once the launch time and point are free.

**cc_qm24, 70-74 s** (scouts: 254 20 shots, 3847 5, 694 0.5; `split_diag.py ... --gate-trace`):

| view | 2-D, split + traced gate | arc fit, boxes within link distance | notes |
| --- | --- | --- | --- |
| main | 254 34, 3847 10 | 254 32, 3847 15 | the traced gate already favours 254; with the arc term 254 33, 3847 11 |
| blue_end | 3847 17, 254 1, 5507 1 | 3847 24 | in 14 of 3847's 17, 254 has no box near the arc; with 254's fused track position as a candidate, the arc fit picks 254 in all 14 |

**Robots without a box near the arc** (`hidden_log.py` logs every box-linked launch with the 2-D winner's
teammates that have no box within dist 2.0 in the view but a fused position; `eval_hidden.py` scores them on
scout truth). "Behind" = the teammate's position projects inside the winner's box. Rule: keep the 2-D winner
unless a teammate behind it fits better by more than m (m chosen leave-one-match-out):

| set | judged launches | with a teammate behind | 2-D right | rule right | diff [95 % CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| today's defaults, all | 4,606 | | 72.9 % | 73.8 % | +0.9 [−0.2, +2.0] |
| ... teammate behind the winner (94 % end cameras) | | 807 | 59.9 % | 64.8 % | +5.0 [−1.3, +11.0] |
| split + traced gate, all | 32,358 | | 73.5 % | 73.7 % | +0.2 [−0.1, +0.5] |
| ... teammate behind, end cameras | | 2,890 | 62.4 % | 64.1 % | +1.7 [−1.8, +5.3] |
| ... teammate behind, main camera | | 822 | 56.6 % | 56.6 % | 0 (m picks "never") |

* A teammate behind the winner is the scouted shooter in 28-31 % of these cases. The qm24 window is the good
  case and doesn't generalise at significance.
* In the main camera, letting the arc pick the hidden teammate outright is harmful: 45.0 % vs 56.6 %. With the
  launch point free, a robot behind fits short arcs too easily.
* No A/B was run for this variant. Per-team fuel didn't respond to a significant +3 pp link gain, and this
  one isn't significant.

**Verdict unchanged: don't adopt.** With split tracks, the gate (traced or not) and recall decide 254's credit,
not the link. In the main view the traced gate already sends 254's split arcs to 254. In the end view 254 has
no box, and only a hidden-robot candidate (above) could take those, which the truth sets don't support yet.
