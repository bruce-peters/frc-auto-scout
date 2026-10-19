# Visual QA of the worst scout-vs-credit bursts (2026-10-18)

*Written by the visual QA agent, saved by the main session. Images are in `img/`, tools in `tools/`, burst
lists in `bursts_cache.json` / `sel.json`, scout clock offsets in `scout_lag.json`. The in-memory analysis
cache, `cache/` (526 MB), can be deleted; `tools/prep_match.py` rebuilds it in about 18 min.*

## Summary for Bruce

I watched the 20 scoring bursts where the credit split and 5026's scouts disagree most, plus 10 random bursts
as a control. For each one I rendered broadcast frames with the robot boxes, the launch arcs and every flight
track drawn on, and looked at them next to the raw video.

**The worst disagreements are mostly not the system's fault.** In 11 of the 20 bursts the video shows the
credited robot doing the shooting, and the scouting is off at the burst level:
* "+20" taps are entered after a dump finishes;
* a scout's clock runs 5-20 s off;
* the only time-valid scout row of the robot that shot is empty.

Match totals in those rows can still be fine. So the "top 20 disagreements" mostly measure scout timing, and
burst-level audits such as `team_volleys.py` should not be trusted until the scout taps are cleaned. The credit
model fit uses match shares, so it is not affected.

When the system really is wrong, the video shows three things:

1. **The shooter has no usable box** (4 of 20):
   * 4499 against the glass in the corner;
   * 6800 too close to the blue end camera;
   * 3045 behind the hub;
   * 2910 hidden behind the hub and briefly labelled 4414.

   The shooter's arcs are tracked (grey) but tied to nobody, or tied by projection to the wrong robot, sometimes
   on the other alliance. The hidden-robot pseudo-shots then hand the balls to whichever teammate is also unseen.
2. **Two robots overlap on screen** (2 of 20, plus 1 control): one robot is in front of another in an end camera,
   or beside it at the hub. The arcs go to the front robot. The box "fill" column in dets.json shows which
   hopper is actually emptying; it picked the scouted shooter in all 3 cases checked.
3. **Smaller one-offs** (1 burst each):
   * a dense 20-ball spray that breaks into short tracks and never becomes launches;
   * a box with no field track (604 in 5026's qm54), whose launches fall back to projection at about half
     weight;
   * a long dump still queuing in the hub after the 5.5 s credit window, so stray launches inside the window
     take the credit.

**The control bursts look good.** In 4 of 10 the split is right to within 2 balls, and in 3 more it is mostly
right, with 5-20 balls leaking to unseen robots through the hidden-robot prior. The remaining 3 are unclear from
the video (a stacked pair, both robots shooting, robot not found).

**Suggested fix order:**
1. Clean the scout taps (clock offset per row, spread batch taps, drop duplicate rows) so the burst metrics mean
   something.
2. Hold a robot's box for a second or two after the detector loses it, and give launches from box-less spots to
   the nearest hidden teammate.
3. Break ties between overlapping boxes by which hopper is draining.

## How it was done

* **Data.** Every scouted Chezy qual with a scan (65 matches) was re-analysed in memory with today's defaults:
  `pipeline.analyse(md, Options())` with `load_dino`, the credit model `models/attr_model.json` (lin/all+dist,
  g_opr 0), split tracks and the traced gate. output/ was not used because finalize_v10 was rewriting it.
* **Bursts** (`tools/find_bursts.py`, which generalises team_volleys.py to all teams):
  * counter increments no more than 1.2 s apart, split at 6 s;
  * the scouted share is the score_scouting `tri` kernel over time-valid rows; the credited share comes from
    fuel_log;
  * "moved" = K × the total-variation distance between the two shares;
  * over 1,618 scouted bursts (42,805 balls), 12,616 balls (29.5 %) are moved.
* **Selection** (`tools/select_bursts.py` → `sel.json`):
  * 20 disagreement bursts with K ≥ 8, the scouted top robot ≥ 0.6 and its credit at least 0.4 lower; at most
    3 per match, over 10 matches, with 5026's qm54 forced in;
  * 10 random controls with K ≥ 5.
* **Rendering** (`tools/render_burst.py`): 4 frames from 3 s before the jump to the jump, annotated crop next to
  the raw crop.
  * Boxes are in team colours (cyan, magenta, green in station order); dashed white means unlabelled; the other
    alliance is grey.
  * Launch arcs are in the credited team's colour (orange for the other alliance); every flight track alive in
    the last 0.6 s is grey.
  * The header gives the shares, launches per robot / view / link kind, visibility and the scout taps.
  * `tools/zoom.py` makes tight crops; `tools/timeline.py` shows per-robot taps against launches.
  * Image names: `2026cc_<match>_<alliance><t>_<class>_{ann,full}.jpg` for burst sheets and
    `2026cc_<match>_t<t>_<class>_zoom.jpg` for zooms.

## Failure classes (20 disagreement bursts)

"Moved" is what the burst metric counts; "really wrong" is an estimate from the video of balls the credit put on
the wrong robot.

| # | class | bursts | moved (metric) | really wrong (est.) | where | example images |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | **Scouting reference wrong at burst level** (batch "+N" taps after the dump, scout clock offset, the only time-valid row empty, duplicate rows); the video shows the credited robot shooting | 11 | 490 | ~40 | all views | `img/2026cc_qm55_b143_scoutref_ann.jpg`, `img/2026cc_qm55_t140_scoutref_zoom.jpg`, `img/2026cc_qm17_r147_scoutref_ann.jpg`, `img/2026cc_qm23_t32_scoutref_zoom.jpg`, `img/2026cc_qm32_t66_scoutref_endcam_zoom.jpg` |
| 2 | **Shooter has no usable box** (missed detection against the glass or in a corner, too close to an end camera, behind the hub, wrong segment); arcs grey or projected to the wrong robot (once the other alliance); the hidden prior then picks among unseen robots | 4 | 161 | ~100 | main corner / behind hub, blue_end near camera | `img/2026cc_qm42_t15_nobox_corner_zoom.jpg`, `img/2026cc_qm6_t19_nobox_endcam_zoom.jpg`, `img/2026cc_qm55_t127_nobox_zoom.jpg`, `img/2026cc_qm23_t82_nobox_zoom.jpg` |
| 3 | **Two robots overlap on screen** (stacked in depth, or side by side at the hub): arcs go to the front robot | 2 | 80 | ~25 | main, near the alliance wall / hub | `img/2026cc_qm58_t102_stacked_zoom.jpg`, `img/2026cc_qm58_t101_stacked_hopper_raw.jpg`, `img/2026cc_qm33_t39_sidebyside_zoom.jpg` |
| 4 | **Dense spray never becomes launches** (a 20-ball cloud breaks into 4-7 point tracks; the shooter is behind an opponent's box) | 1 | 48 | ~35 | main | `img/2026cc_qm33_t140_spray_zoom.jpg`, `img/2026cc_qm33_t140_spray_wide_zoom.jpg` |
| 5 | **Box without a field track** (no segment, no team): launches fall back to projection at about 0.44 confidence | 1 | 40 | ~15 | main, left side | `img/2026cc_qm54_t23_unlabelled_zoom.jpg` |
| 6 | **Hub queue lag past the credit window**: a dump at 94-98 s is still being counted at 100-104 s | 1 | 34 | ~25 | timing | `img/2026cc_qm6_r102_hubqueue_ann.jpg` |
| | total | 20 | 852 | ~240 | | |

* **Not a main cause anywhere:** human player throws, bounces or rattle, launches from the neutral zone,
  outpost / corner balls.
* **Secondary causes:**
  * **Hidden-prior leak.** 3847 gets 12 balls with zero launches and zero visibility (qm17 r22); 9496 gets 34
    (qm55 r128). The qm13, qm41 and qm45 controls show the same.
  * **Wrong identity:**
    * qm23: segment 87 is labelled 4414 while the OCR-confirmed 4414 is on screen; it is probably 2910.
    * qm55: red 9496, lit purple, is classed blue and labelled 687 in the red end camera.
    * qm32: a one-frame 973/9408 swap.
  * **A shooter that starts unlabelled.** In qm17 r153, 24 of 4414's launches go to nobody while its box is still
    unlabelled.

### Per burst

| burst | K | scouts say | credit says | what the video shows | class |
| --- | ---: | --- | --- | --- | --- |
| qm55 blue 143 | 66 | 687 1.00 | 9032 0.99 | 9032 streams into the blue hub (bumper readable). 687's row 0 has a −20 s clock offset; its row 1 taps in +20 batches. | 1 |
| qm17 red 153 | 74 | 3847 0.79 | 4414 0.96 | 4414 streams. 4414's valid scout taps in late batches; 3847's row 0 is +12 s off. 24 early launches are unlabelled. | 1 |
| qm23 red 36 | 57 | 2910 0.89 | 4414 0.91 | 4414 streams. 2910 (bumper readable) is gathering in the neutral zone. | 1 |
| qm23 red 153 | 61 | 2910 0.95 | 4414 0.77 | 4414 streams from the red wall; 2910 is behind the hub. 4414's valid row logged 2 balls. | 1 |
| qm33 red 141 | 61 | 2910 1.00 | 4698 0.76 | A 20-ball spray from 2910, behind blue 6647's box: 164 tracks in 2.7 s, only 6 launches. | 4 |
| qm32 red 116 | 60 | 9408 1.00 | 973 0.78 | 973 (bumper readable) streams from the corner. 973's valid scout row logged nothing. | 1 |
| qm58 red 105 | 73 | 687 0.98 | 3847 0.47, 687 0.30 | 3847 is in front of 687. 687's box fill drains 0.57 → 0.06 while 3847's stays about 0.3. | 3 |
| qm17 red 147 | 57 | 4414 0.79 | 3847 0.91 | 3847 (bumper readable) streams; 4414 is unseen. The scout order is shifted against the video. | 1 |
| qm55 blue 112 | 51 | 687 0.91 | 9032 0.93 | 9032 shoots. The "687" box in the red end camera is red 9496. | 1 |
| qm6 blue 20 | 57 | 6800 0.77 | 2073 0.60 | 6800 is in front of the blue end camera with no box; its arcs are projected to red 2813. | 2 |
| qm52 red 96 | 50 | 841 0.97 | 9128 0.80 | Stacked trio at the red wall. 9128's scouts enter +20 batches 4 s late. | 1 |
| qm23 red 83 | 68 | 2910 0.83 | 4414 0.58 | 2910 drives behind the hub under segment 87, labelled 4414. | 2 |
| qm6 red 102 | 39 | 2813 0.90 | 8229 0.62 | A dump at 94-98 s; the counter is still rising at 100-104 s. | 6 |
| qm42 blue 14 | 60 | 4499 0.73 | 5940 0.77 | 4499 (against the glass) launches 9 clean arcs with no box from 14.9 to 17.5 s. | 2 |
| qm54 red 26 (5026) | 46 | 604 1.00 | 254 0.85 | Both shoot. 604's box has no segment, so its 30 arcs are projection-linked at about 0.44. | 5 |
| qm55 red 128 | 77 | 3045 1.00 | 3045 0.50, 9496 0.45 | 3045 shoots from behind the hub with no box; the hidden prior splits between 3045 and 9496. | 2 |
| qm17 red 22 (auto) | 53 | 6017 0.81 | 4414 0.67 | 4414 streams all of auto; its scout taps arrive late. 3847 (unseen) gets 22 % from the hidden prior. | 1 |
| qm32 red 68 | 47 | 973 1.00 | 9408 0.70 | Both shoot. 9408's scouts enter batches after the window. | 1 |
| qm33 red 40 | 47 | 4698 1.00 | 2910 0.62 | Side by side; 2910's fill drains, 4698's doesn't. The 4698 scout rows are duplicates with an offset, so the system is probably right. | 3 |
| qm58 red 69 | 35 | 9128 0.91 | 3847 0.86 | 3847's arcs reach the hub; its only scout row is +18 s off. 9128 lobs mostly untracked low arcs. | 1 |

## Control bursts (10 random, K ≥ 5)

| burst | K | scouts | credit | verdict |
| --- | ---: | --- | --- | --- |
| qm14 red 164 | 26 | 971 0.47 / 5507 0.53 | 0.51 / 0.49 | right |
| qm19 red 46 | 14 | 5199 0.38 / 9470 0.62 | 0.43 / 0.57 | right; 5199's box goes unlabelled mid-volley |
| qm38 red 153 | 35 | 9408 1.00 | 9408 0.95 | right |
| qm6 blue 145 | 12 | 2073 1.00 | 2073 0.84 | right; 15 % prior leak to 6800 |
| qm45 red 65 | 24 | 6238 1.00 | 6238 0.77 | mostly right; hub drops projection-linked to 8229 |
| qm41 blue 40 | 60 | 841 0.53 / 5199 0.47 | 841 0.69 / 2910 0.17 / 5199 0.15 | mostly right; 5199 (behind the hub) undercounted, 2910 gets ~10 from the prior |
| qm13 blue 95 | 20 | 6017 0.48 / 846 0.52 | 6017 0.72 / 846 0.25 | 846 visibly shoots; 6017 is never seen but gets 72 % from the prior |
| qm9 blue 174 | 15 | 604 0.73 | 5940 0.71 | unclear (both shoot) |
| qm15 blue 13 (auto) | 69 | 5940 0.58 / 254 0.32 | 254 0.54 / 4698 0.31 | unclear (stacked at the hub) |
| qm22 red 72 | 20 | 971 1.00 | 359 0.56 / 6800 0.43 | unclear; the "971" box sits on blue 3045 |

* **Right within 2 balls:** 4 of 10, and mostly right in 3 more.
* **Largest share to the visible shooter:** 7 of 10.
* **Moved:** the controls average 9.5 of 29.5 balls, the same ~30 % as all bursts. So roughly a third of
  "moved" balls in a typical burst is reference noise, not credit error.

## How big each class is across all 65 matches (from the cache, not visual)

* **Scout timing.**
  * 80 % of scouted balls are entered as batch taps (≥ 5 identical timestamps).
  * 84 of 520 time-valid rows (16 %, 10,414 balls) have a confident clock or batch offset beyond 3 s.
  * 11 of 740 rows are exact duplicates.
* **Hidden-prior credit.** 2,050 balls (4.8 %) go to robots with zero launches and under 50 % visibility; the
  scouts back about 57 % of that.
* **Unlabelled boxes.**
  * 7-12 % of main-view boxes have no segment (38 % in the leftmost 240 px, 100 % in the rightmost); 13-16 % in
    the end views.
  * 4.2 % of launches are box-linked to such a box and dropped; 9.4 % are projection links at 0.29-0.44
    confidence.
* **Duplicate labels.** 2.6 % of labelled boxes share their team with another segment in the same view and frame.
* **Hub queue lag.** 3.0 % of counted balls land on increments with less than 1 kernel-weighted launch but ≥ 10
  launches 5.5-10 s earlier.

## Fix ideas for the top 3 classes

### 1. Clean the scout taps (11 / 20)

* **Idea.** Per scout row:
  * estimate the clock offset by cross-correlating its taps with the alliance counter (`tools/scout_lag.py`);
  * shift rows with a confident offset;
  * spread "+N" batches backward at 6-8 balls/s;
  * drop duplicate rows.

  This touches only the reference, never the credit.
* **Test.** Rerun `find_bursts.py` and `scripts/score_scouting.py` (burst overlap) with the cleaned taps. Match-level
  MAE should not move.

### 2. Shooter without a usable box (4 / 20, ~100 balls)

* **Idea.**
  1. **Box hold.** Keep a segment's last box in a view as a link candidate for up to `BOX_HOLD_S` (1-2 s) after
     the detector loses it, with a small distance penalty.
  2. **Box-less arcs.** A clean rising arc heading for the hub, with no box and no projection within 1 m, goes to
     the alliance robot whose fused / interpolated position projects nearest (up to ~3 m), as `proj_hidden`.
* **Test.**
  * `link_x_ab.py --arms "BOX_HOLD_S=0;BOX_HOLD_S=1;BOX_HOLD_S=2"`, then `build_ds.py --season` and
    `fit_cv.py --specs "lin/all+dist" --lam 3e-5`.
  * The fitted `hidden` should drop and held-out MAE fall.

### 3. Overlapping robots (2 / 20 plus 1 control)

* **Idea.** Among boxes within link distance, favour the one whose same-view box fill (`d.fill`) fell over the
  preceding ~1 s: `score -= LINK_FILL_TIE * max(0, fill(t-1 s) - fill(t))`.
* **Evidence.** It points the right way in all three cases checked: qm58 687, qm32 973, qm33 2910.
* **Test.** Pick accuracy with `data/exp_arcs/tools/eval_truth.py` (today 71.4 %, same-alliance 61.3 %), then a
  link_x_ab A/B with a refit.

**Smaller classes:**
* **Spray:** count a cloud of short rising tracks as launches.
* **Unlabelled boxes:** inherit the overlapping segment's team.
* **Hub queue:** stretch `LAG_HI` with the queue length.

## Caveats

* One rater, at broadcast resolution.
* Burst-level scouted shares can be off even when a scout's match total is right.
* qm22 r72, qm9 and qm15 are judgment calls.
* "Really wrong" ball counts are rough estimates from watching.
