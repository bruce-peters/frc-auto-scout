## Hub entry times and the scoreboard lag (2026-10-18; re-timing built, credit change not adopted)

**Question.** Each scoreboard jump is shared among the alliance's robots by their launch arcs 0.8-5.5 s before it
(`attribution.kernel`). Bruce saw in the inspector that the delay between a ball going in and the counter moving
is wide and variable. If we knew when each counted ball dropped into the hub, could a narrow per-ball match to
the launch one flight time earlier replace the broad window and credit the right robot more often?

**Answer.** The delay is now understood and measurable per ball, but crediting from it does not pay off yet.

* **Why the lag varies.** The hub counts at most about 15 balls/s. A ball that drops into an idle hub is
  counted about 1.3 s later. In a hopper dump the balls queue, and the last ones are counted 2-4 s after they
  went in. Between matches and broadcasts the idle delay is stable (spread 0.2-0.4 s).
* **Entry times.** Balls dropping into the opening (flight-track ends plus blobs over the opening) re-time every
  counted ball, first in first out. This halves the timing spread when checked against a camera view the
  re-timing did not use (IQR 1.53 s to 0.72 s).
* **Arcs.** Only 1.6 % of counted balls can be followed along one tracked arc from a linked launch into the
  hub (6 % with loose thresholds). The launch-to-entry link therefore has to be statistical, by timing.
* **Credit.**
  * The best variant (half narrow entry kernel, half today's kernel) lowers video-only team-match error by
    1.0 ball (95 % CI −1.8 .. −0.1).
  * With the credit model refitted fairly for every arm, the best-estimate error drops 0.5 balls
    (−1.3 .. +0.2), which is within noise.
  * Per-jump agreement with the scouts drops 0.8 points (−1.0 .. −0.5).
  * 254 loses more volleys to teammates: 177 balls against 152.
  * Narrower variants trade more per-jump agreement for slightly lower match error.
* **The limit is launch recall.** About 0.22 launches are detected per counted ball, so a narrow window often
  holds no launch from the real shooter. With half the launches dropped, the gain disappears.
* **Recommendation: don't adopt the credit change now.** Keep `autoscout/hubentry.py` for diagnostics, fix the
  lag description in the docs, and re-run the A/B once the tracker's launch recall improves (commands at the end).

**In plain words.** The scoreboard is not just late, it is a queue. The hub lets about 15 balls a second
past its sensor. When a robot dumps 60 balls in two seconds, the scoreboard keeps ticking for another four or
five. We can now tell from the video when each ball actually went in. But we see only about one launch in five,
so pinning each ball on the launch exactly one flight before it is often a guess. Today's wide window, which
pools several seconds of launches, is about as good. The sharper timing will start to pay once more launches
are detected.

![A hopper dump in 2026cc_qm24](img/queue_qm24_blue.png)

*2026cc_qm24, blue, 33-46 s. About 80 % of a 65-ball dump drops into the opening (blue, orange) within 2.5 s, but the
counter (black) runs for 5.7 s at a steady ~11 balls/s. Green dashes: the counted balls re-timed to entry times.
Grey ticks: the 25 launches the tracker detected for those 65 balls.*

### 1. Lag study (`tools/lag_study.py`, `lag_deconv.py`, `lag_bursts.py`, `first_ball.py`, `delay_profile.py`)

All 131 processed matches (138 Chezy and 121 FIRST-layout alliance-matches with ≥ 20 balls). Inputs: overlay
counter (3 reads/s; each jump placed mid-way into its read gap), hub-top blob activity (`hub_feats`, 30 Hz),
entry tracks (section 2) and detected launches. "Peak lag" = normalised cross-correlation peak per
alliance-match. "IR" = NNLS-deconvolved impulse response (delay distribution with the signal's own clustering
removed), pooled.

| delay to the counter | Chezy | FIRST layout |
| --- | ---: | ---: |
| blobs over the opening, main view: peak lag, median (IQR) | 1.75 s (1.63-1.86) | 1.66 s (1.50-1.89) |
| ... spread between alliance-matches (SD) | 0.19 s | 0.36 s |
| balls dropping into the opening (entry tracks): IR q10 / median / q90 | 0.58 / 1.54 / 3.66 s | 0.56 / 1.44 / 2.77 s |
| first ball of a burst after ≥ 3 s of silence (idle hub), median (IQR) | 1.35 s (1.04-1.85) | 1.70 s (1.23-2.50) |
| detected launches: peak lag, median (IQR) | 2.43 s (2.23-2.65) | 2.43 s (2.19-2.73) |
| detected launches: IR q10 / median / q90 | −0.1 / 2.42 / 4.79 s | 0.3 / 2.49 / 4.51 s |
| launch to blobs over the opening (flight), main view | 0.74 s | 0.80 s |
| blobs at the hub exits (balls leaving) | 0.5 s *after* the counter | 0.4 s *after* |

**The hub is a queue.** In Chezy bursts of ≥ 30 balls, the counter's busiest second holds a median of 14 balls
and a 90th percentile of 16. Counted balls re-timed first in, first out (section 2) wait longer the deeper they
are in a burst:

| ball's position in its counter burst | Chezy delay q10 / median / q90 | FIRST delay q10 / median / q90 |
| --- | ---: | ---: |
| balls 0-5 | 0.76 / 1.25 / 1.83 s | 0.67 / 1.37 / 2.16 s |
| balls 15-30 | 1.04 / 1.67 / 2.24 s | 0.60 / 1.38 / 2.06 s |
| balls 30-60 | 1.08 / 2.03 / 3.10 s | 0.63 / 1.57 / 2.43 s |
| balls 60+ | 0.74 / 2.13 / 4.32 s | 0.65 / 1.78 / 2.93 s |

* **Between matches the delay barely moves.** The median is 1.75 s Chezy and 1.66 s FIRST over the opening,
  with an SD of 0.2-0.4 s. A per-match lag estimate therefore buys almost nothing. The variation is within a
  match, and it is the queue.
* **Exits.** Balls appear at the exits about 0.5 s after the counter moves. So the display does not sit
  seconds behind the sensor: the wait is inside the hub, before counting.
* **Broadcasts.** The FIRST overlay counter updates in bigger, irregular chunks. Single jumps reach 15+ balls,
  and only 55 % of consecutive jumps are one read apart, against 84 % on Chezy. FIRST events also show less
  queueing (smaller dumps). The idle delay is the same within about 0.3 s.
* **Today's kernel.** The launch-to-counter peak is 2.4 s, against the kernel's 2.8 s. Moving the peak to
  2.45 s changes nothing measurable (table in section 4).

### 2. Entry detection (`autoscout/hubentry.py`, `tools/extract.py`, `entry_quality.py`, `retime_quality.py`)

**Entry tracks.** A flight track counts as an entry when its last point lies in the projected hub opening:
* |u − c_u| ≤ 1.0 r and −0.6 r ≤ v − c_v ≤ 1.2 r, where r is the opening's image radius at the current camera pose;
* and it is moving down at the end.

The track ends of all 131 matches cluster tightly there (main and own-end views). The far end view is useless.

**Entry evidence.** `entry_series` sums four sources, each scaled to mean 1 over the match: entry tracks and
blobs over the opening (shifted +0.2 s), in the main view and the alliance's own end view.

**Re-timing.** `retime_queue` works per counter burst; a gap > 1 s means the queue emptied.
* The burst's first ball entered 0.8-2.6 s before its count.
* No ball entered before the previous burst's last ball.
* Cumulative counted balls are matched to cumulative evidence above background. Each read's balls are cut into
  4 pieces.
* Delays are clipped to [0.6, 7] s.
* A burst with too little evidence falls back to count time − 1.35 s.

The ball count is exactly the counter's.

| | Chezy | FIRST |
| --- | ---: | ---: |
| entry tracks per counted ball: main / own end / far end view | 0.47 / 0.52 / 0.34 | 0.81 / 0.68 / 0.23 |
| corr. of an alliance-match's entry-track count with its counted balls (main / own end / far) | 0.81 / 0.68 / 0.06 | 0.81 / 0.58 / −0.05 |
| entry tracks that are themselves linked launch arcs | 4 % | 4-5 % |
| counter reads re-timed from evidence (not the fallback) | 99.6 % | 99.1 % |
| **held-out check**: own-end entry tracks vs balls re-timed from the main view only: IQR of the IR (raw counter → re-timed) | **1.53 → 0.72 s** | **1.27 → 0.81 s** |
| ... IR mass within ±0.5 s of its peak (raw counter → constant shift 1.6 s → re-timed) | 0.39 → 0.39 → 0.65 | 0.45 → 0.45 → 0.59 |
| ... peak of the IR (raw counter → re-timed) | +1.0 → 0.0 s | +1.4 → 0.0 s |
| launches → balls: IR IQR (raw counter → re-timed, all views) | 2.11 → 1.42 s | 1.82 → 1.54 s |
| launches → re-timed balls: peak / median (flight time) | 1.2 / 0.9 s | 1.3 / 1.1 s |

Counting balls from entries alone would be poor (0.5-0.8 tracks per ball, varying by match). This is why the
vision-only counter is ~30 % off. Timing is what entries are good for. Against a view the re-timing never saw,
the re-timed balls line up at 0 s with half the spread of the raw counter. A constant shift does not help at
all; only the per-ball queue model does.

### 3. Entries to arcs to robots (`tools/arc_link.py`)

An entry track is linked to a robot in one of two ways:
* **direct:** it is a launch arc that `analyse` already linked;
* **chained:** a linked launch arc in the same view ended ≤ 1.2 s before it, and one projectile (u linear, v
  quadratic) fits both fragments within 4 px RMS, with a consistent speed.

Links are deduplicated across views.

| | Chezy | FIRST |
| --- | ---: | ---: |
| counted balls with a confident arc (direct + chained) | **1.6 %** (660 + 426 of 68,563) | **1.9 %** (392 + 328 of 38,305) |
| ... with loose thresholds (10 px, 2 s gap) | 5.8 % | 10.3 % |
| linked arcs whose robot is of the hub's alliance | 97 % (loose 95 %) | 95 % (93 %) |
| flight time of fully linked arcs, q10 / median / q90 | 0.87 / 1.10 / 1.33 s | 0.80 / 1.07 / 1.30 s |

The links that exist are right: 97 % come from the hub's own alliance. They are just rare. A launch fragment
is usually 4-6 points (0.1-0.2 s) and the ball is lost near the apex, so almost no ball is followed from robot
to hub. Linking has to go through timing (launch + ~1.0 s flight ≈ entry), which is what the credit variants do.

### 4. A/B of the credit split (`tools/capture.py`, `ab.py`, `burst_variants.py`, `refit_cv.py`)

**Method.**
* **Capture.** Each of the 65 scouted Chezy quals with a scan is analysed in memory once with today's defaults
  (`pipeline.analyse(md, Options())` after `load_dino`, as in `link_x_ab.py`). Every
  `attribution.split_increments` call's inputs are recorded: increments, TeamSignals, OPR, model on/off.
* **Re-run.** Each arm re-runs the split on exactly those inputs, both the learned-model call and the
  video-only call. It writes `result.json` with `teams` and `fuel_log` replaced. Log stamps stay at counter
  time − 2 s, so `score_scouting` and `team_volleys` read them unchanged.
* **Parity.** The base arm reproduces the in-memory analyse exactly (max difference 0.0, identical logs). It
  also matches the current RESULTS numbers: 40.9 / 48.0 / 75 % / 64 %.
* **Scoring.** 54 matches, 315 team-matches and 105 alliances, via `score_scouting.score`. Paired bootstrap
  over matches, 1,000 resamples.
* **Narrow kernels.** These are scaled to the same area as today's kernel (2.35 s), so launch evidence keeps
  the scale the credit model was fitted on.

| arm | best-estimate MAE | video-only MAE | top scorer (best / video) | balls per jump given to robots scouts saw shooting | matches better / worse (best) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **today** (kernel 0.8 / 2.8 / 5.5 s from the counter jump) | 40.9 | 48.0 | 75.2 % / 62.9 % | 63.8 % | – |
| peak moved to 2.45 s | +0.04 (−0.04 .. +0.11) | +0.07 (−0.03 .. +0.18) | ±0 / ±0 | +0.02 (−0.02 .. +0.06) | 25 / 29 |
| narrower from the jump, 1.2 / 2.45 / 4.2 s | +0.00 (−0.73 .. +0.72) | −0.27 (−1.18 .. +0.57) | ±0 / −1.0 | −0.96 (−1.26 .. −0.68) | 23 / 31 |
| narrow, constant 1.6 s entry delay (no entry evidence) | −0.28 (−1.73 .. +1.16) | −0.41 (−2.06 .. +1.20) | +1.0 / −2.9 | −2.68 (−3.13 .. −2.21) | 29 / 25 |
| **entry re-timed**, narrow kernel 0.2 / 0.8 / 1.9 s after the entry | −0.81 (−2.37 .. +0.84) | −1.82 (−3.63 .. +0.08) | −1.0 / ±0 | −2.65 (−3.26 .. −2.03) | 29 / 25 |
| entry re-timed, wider 0.0 / 0.9 / 2.5 s | −0.57 (−1.99 .. +0.94) | −1.59 (−3.32 .. +0.14) | −1.0 / +1.9 | −1.58 (−2.16 .. −1.04) | 27 / 26 |
| entry re-timed, today's broad kernel measured from the entry (+2.0 s) | −0.31 (−1.25 .. +0.71) | −0.56 (−1.67 .. +0.64) | −1.0 / +1.0 | −0.02 (−0.36 .. +0.28) | 25 / 28 |
| **entry re-timed, 0.3 narrow + 0.7 today's kernel (mix 0.7)** | −0.38 (−0.76 .. +0.04) | −0.43 (−0.86 .. +0.01) | ±0 / +1.0 | −0.14 (−0.29 .. −0.01) | 30 / 24 |
| **entry re-timed, half and half (mix 0.5)** | −0.50 (−1.00 .. +0.09) | −0.69 (−1.31 .. −0.05) | +1.0 / +1.9 | −0.40 (−0.60 .. −0.21) | 31 / 22 |
| **entry re-timed, 0.7 narrow + 0.3 today's (mix 0.3)** | −0.62 (−1.34 .. +0.19) | **−0.96 (−1.79 .. −0.09)** | +1.9 / +1.9 | −0.77 (−1.05 .. −0.52) | 31 / 23 |
| entry evidence from tracks only (narrow) | −0.80 (−2.21 .. +0.62) | −1.64 (−3.33 .. −0.01) | −1.0 / −2.9 | −2.62 (−3.21 .. −2.05) | 28 / 26 |
| entry evidence from blobs only (narrow) | −0.18 (−2.14 .. +1.84) | −1.45 (−3.68 .. +0.69) | −2.9 / ±0 | −2.89 (−3.56 .. −2.21) | 27 / 27 |
| re-timed without the queue structure (narrow) | −0.41 (−2.57 .. +1.71) | −0.69 (−2.95 .. +1.60) | −1.0 / −3.8 | −3.64 (−4.40 .. −2.87) | 24 / 30 |

All rows are differences from today, with 95 % CIs. Top scorer: one alliance is 0.95 points. Mix = the share of
today's counter-time kernel in the launch evidence. Every arm and its CIs are in `ab.json`, `ab_b.json` and
`ab_c.json`, including flight peaks 0.8 vs 1.05 s, tight/wide kernels and the unscaled kernel. None of those
beat the rows shown.

**The best estimate is in-sample for today's arm.** The learned credit model was fitted on today's evidence for
these same matches. Each arm's six model numbers were therefore refitted leave-one-match-out (`refit_cv.py`:
same objective for every arm, cross-entropy of match shares vs the scouted shares):

| arm | refit, held out: fuel MAE | top scorer | MAE vs today (95 % CI) |
| --- | ---: | ---: | ---: |
| today | 41.73 | 75.2 % | – |
| entry re-timed, narrow | 41.07 | 71.4 % | −0.66 (−2.20 .. +0.95) |
| entry re-timed, mix 0.3 | 41.19 | 75.2 % | −0.54 (−1.25 .. +0.22) |
| entry re-timed, mix 0.5 | 41.28 | 74.3 % | −0.44 (−1.01 .. +0.12) |
| constant 1.6 s delay, narrow | 41.40 | 75.2 % | −0.33 (−1.80 .. +1.06) |

**Is the per-jump drop an artifact of the burst metric?** For a jump at T, the metric's truth is the scouts' taps
in a ±2.5 s triangle around T − 2 s. That is the same smear as today's kernel, so it could favour today. It
mostly does not (`burst_variants.py`, change in points vs today, 95 % CI):

| truth window | narrow | mix 0.3 | mix 0.5 | constant 1.6 s |
| --- | ---: | ---: | ---: | ---: |
| official (T − 2.0 s, ±2.5) | −2.65 | −0.77 | −0.40 | −2.68 |
| narrow (T − 2.0 s, ±1.5) | −1.91 (−2.51 .. −1.19) | −0.58 (−0.84 .. −0.29) | −0.28 (−0.47 .. −0.06) | −1.81 |
| narrow, later (T − 2.4 s, ±1.5) | −1.90 | −0.59 | −0.29 | −1.79 |
| whole counter bursts (gaps > 2 s; timing inside a burst ignored) | −0.49 (−0.93 .. −0.09) | −0.02 (−0.27 .. +0.24) | −0.01 (−0.20 .. +0.18) | −0.18 |

A sharper truth window does not rescue the narrow arms: per jump they really agree less with the scouts. At the
level of whole bursts the mixes are neutral.

**254 check** (`data/exp_audit/tools/team_volleys.py 254 --out data/exp_hubentry/ab/<arm>`):

| arm | jumps where scouts say ≥ 60 % 254 but credit is > 0.3 lower | balls lost | balls credited to 254 (scouts: 1,606) |
| --- | ---: | ---: | ---: |
| today | 63 | 152 | 1,772 |
| entry re-timed, narrow | 74 | 198 | 1,717 |
| entry re-timed, mix 0.3 | 72 | 177 | 1,784 |
| entry re-timed, mix 0.5 | 68 | 166 | 1,787 |
| entry re-timed, broad kernel from the entry | 71 | 166 | 1,824 |
| constant 1.6 s delay, narrow | 79 | 209 | 1,653 |

The jumps 254 loses are mostly the same ones as today (qm12, qm33, qm6, qm24). Most have no detected 254 launch
near them, so no kernel can find it. The narrow arms lose some extra jumps where a teammate's one or two
launches fall closer to the re-timed entries.

**Recall probe** (`ab.py --drop 0.5`): half of every robot's detected launches are removed, with the same draw
in every arm. Today's arm falls to 46.2 best / 51.9 video-only, and the entry arms' gains vanish:
* narrow: best +0.55 (−1.48 .. +2.84), video −0.19 (−2.75 .. +2.62);
* mix 0.3: best +0.69 (+0.01 .. +1.44), video +0.35.

The per-jump drop stays (−2.0 and −0.6 points). The match-level gain of entry timing therefore grows with launch
recall. The per-jump cost is not explained by recall alone: a narrow window just pools fewer launches per
decision.

### 5. Recommendation

**Don't adopt the entry-based credit split now.**
* **Gain.** The best variant (mix 0.3) improves team-match error by about 1 ball video-only and 0.5 balls in
  the refitted best estimate, out of ~42. Only the video-only CI excludes zero.
* **Cost.** It costs 0.8 points of per-jump agreement, and 254 loses 25 more balls.
* **Rerun trigger.** This should be re-run when launch recall improves (the tracker work on dense volleys).
  The recall probe says the gain scales with it.

**Port now (small, no behaviour change):**
1. **Docs.** `attribution.py`'s module docstring says "hub sensor lag ~1.3 s, up to 3 s". Replace it with the
   measured picture:
   * ~1.3 s from the opening to the counter when the hub is idle (1.75 s from blobs over it);
   * the hub counts ≤ ~15 balls/s, so balls deep in a dump wait 2-4 s more;
   * launch to counter peaks at 2.4 s.
   `docs/ARCHITECTURE.md`'s knob table can say the same. Keep LAG_* as they are (moving the peak to 2.45 s
   changed nothing).
2. **Optional inspector aid.** Bruce noticed the lag in the inspector. `hubentry.track_ends` (≈ 0.5 s per match,
   from `md.tracks_all` and `md.cams`) and `entry_series` + `retime_queue` give the re-timed entry time of every
   counted ball. Drawing those under the counter shows the queue directly. Not done here, because the inspector
   was out of scope.

**Port spec, if a later re-run clears the bar** (mix 0.3; the main session ports, `pipeline.py` untouched here):
* **Hook.** In `pipeline.analyse`, in the "alliance scoring timeline" loop, only when `sources[al] == "overlay"`:
  ```python
  tables = {v: hubentry.track_ends(trs, md.cams.get(v)) for v, trs in md.tracks_all.items()}  # once per match
  rt, rate = hubentry.entry_series(tables, md.hub_feats, al, md_t0(md), md_t1(md))
  ents = hubentry.retime_queue(inc, rt, rate)
  split = hubentry.split_entries(inc, teams, tl.match_time, ents, log=fuel_log, opr=_attr_opr(info), mix=0.3)
  split_video = hubentry.split_entries(inc, teams, tl.match_time, ents, model=None, mix=0.3)  # hand rule
  ```
  Keep `attribution.split_increments` for the TBA-period fallback. For the version-2 credit model (DINO /
  drain), `_weights` would need the `g_dinoc` / `g_drain` terms of `_model_weights`; the default model is
  version 1.
* **Constants** (`hubentry.py`):
  * opening OPEN_DU 1.0, OPEN_DV (−0.6, 1.2);
  * evidence: EV_SMOOTH 3 bins of 0.1 s, BLOB_SHIFT 0.2 s;
  * `retime_queue`: gap 1.0 s, d_first (0.8, 2.6) s, d_min 0.6 s, D_MAX 7 s, d_typ 1.35 s, split 4, min_ev 0.3;
  * entry kernel 0.2 / 0.8 / 1.9 s, scaled to area 2.35 s.
* **Refit `models/attr_model.json`** on the new evidence with `data/exp_attrmodel/tools`. The simplified LOMO
  refit gives θ = (log prior −3.224, log hidden 0.091, log b −0.481, gz 0.416, g_dist −0.327, g_opr 0.550) for
  mix 0.3 (`refit_cv.json`).
* **Live.** Everything is per match: each evidence source is scaled by its own match mean, and `retime_queue`
  only needs the evidence's shape within each burst (a running mean works too). No event or later-match data
  is used.
* **Cost.** About 1 s CPU per match. No GPU, no rescan.
* **Tests.** `data/exp_hubentry/tools/test_hubentry.py` (move to `tests/`):
  * entries with conf 0 make `split_entries` reproduce `attribution.split_increments` exactly (logs included;
    also checked on 4 real matches, max difference 4e-15);
  * `retime_queue` keeps the ball count, first-in-first-out order and the delay bounds, and shows the queue in a
    synthetic dump;
  * the no-evidence fallback;
  * the kernel area.

**To re-run after the tracker changes** (one heavy process at a time; ~10 min total; the 556 MB `cache/` can be
deleted and rebuilt):
```
.venv/Scripts/python data/exp_hubentry/tools/extract.py --force    # track tables from output/*/scan.pkl, ~80 s
.venv/Scripts/python data/exp_hubentry/tools/capture.py --force    # in-memory analyse of the scouted quals, ~8 min
.venv/Scripts/python data/exp_hubentry/tools/ab.py --arms base,entry_q,entry_q_mix0.3,entry_q_mix0.5 --tag _rerun
.venv/Scripts/python data/exp_hubentry/tools/refit_cv.py --arms base,entry_q,entry_q_mix0.3
.venv/Scripts/python data/exp_audit/tools/team_volleys.py 254 --out data/exp_hubentry/ab/entry_q_mix0.3
```

### Files (`data/exp_hubentry/`)

* **Lag study.** `lag_study.json`, `lag_deconv.json`, `lag_bursts.json`, `first_ball.json`, `delay_profile.json`.
* **Entry detection.** `entry_quality.json`, `retime_quality.json`.
* **Arc links.** `arc_link.json`, `arc_link_relaxed.json`.
* **A/B.** `ab.json`, `ab_b.json`, `ab_c.json`, `ab_drop50.json`, `burst_variants_1.json`, `burst_variants_2.json`,
  `refit_cv.json`.
* **Arm results.** `ab/<arm>/<match>/result.json` for the main arms; `capture/` holds the recorded split inputs.
* **Tools.** All in `tools/`. No GPU was used and nothing in `output/`, `pipeline.py`, `attribution.py` or the
  inspector was changed.
