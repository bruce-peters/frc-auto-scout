## Tracking every ball: flight-tracker recall and 60 Hz (2026-10-18/19, 60 Hz adopted for the re-scan)

**Question.** Bruce, watching the inspector: the video "should be able to track ALL of the balls". How many of
the balls a robot scores become their own launch arc, tied to that robot? Can a better flight tracker, or
analysing every frame (60 Hz) instead of every other one (30 Hz), get close to all of them? And does that make
per-team fuel more accurate?

### How recall is measured (`data/exp_track/tools/track_lab.py`)

* **Truth.** Only moments where exactly one robot of an alliance is scoring count.
  * The alliance's scoreboard rises in an episode, with gaps of at most 2 s between increases.
  * Exactly one of its three robots, all scouted with usable timestamps, has scouted shots then.
  * The other two have none within ±7.5 s.
  * The episode's scoreboard increase is then that robot's balls.
  * Kept only when the scoreboard and the scouts roughly agree (ratio 0.5-2), so the "idle" teammates really
    were idle: 95 episodes, 2,304 balls, 37 teams in 64 scouted Chezy quals.
* **A ball counts for the shooter** if a launch arc meets all three tests:
  * it passes the pipeline's shot test;
  * it heads for the alliance's hub;
  * traced back along its fitted parabola, it rises out of the top of the shooter's box. The arc is tied to
    one robot only: the one whose box top it crossed nearest the centre.
* **Each physical ball counts once.** Arcs with the same traced launch merge: fragments of one ball, and the
  main and end camera seeing the same ball.
* **Recall** is unique balls per scored ball. It can exceed 1 because misses are launched but not scored.
* **False** is the same count credited to the two idle teammates, per scored ball. This is the precision side:
  more arcs only help if they land on the right robot.

### Results

**At 30 Hz** (all 95 episodes). Cells are recall / false per scored ball.

| flight tracker | own end camera | main camera | any view | 20+ ball volleys |
| --- | --- | --- | --- | --- |
| greedy (the tracker until today) | 0.19 / 0.08 | 0.08 / 0.02 | 0.19 / 0.08 | 0.17 / 0.07 |
| greedy + split at jumps (default since 19:15) | 0.78 / 0.22 | 0.64 / 0.15 | 0.84 / 0.37 | 0.77 / 0.31 |
| ballistic (`fuel.BallisticTracker`) + split | 0.90 / 0.25 | 0.66 / 0.16 | 0.92 / 0.41 | 0.84 / 0.35 |
| ballistic + fragment merge + split | 0.84 / 0.22 | 0.64 / 0.14 | 0.88 / 0.36 | 0.81 / 0.30 |

* Fragment merge = blobs within 10 px merged before tracking.
* Stitching tracklets along one parabola added nothing and was dropped.

**30 vs 60 Hz** (the same 95 episodes; 60 Hz = `Options.fuel_step = 1`, a fuel-only re-scan of all 65 scouted
quals). Cells are recall / false at 30 Hz -> 60 Hz.

| flight tracker | any view | own end camera | main camera | shooter seen only by the main camera | 20+ ball volleys |
| --- | --- | --- | --- | --- | --- |
| greedy | 0.19 / 0.08 -> 0.58 / 0.26 | 0.19 -> 0.59 | 0.08 -> 0.28 | 0.14 -> 0.48 | 0.17 -> 0.52 |
| greedy + split | 0.84 / 0.37 -> 1.15 / 0.51 | 0.78 -> 1.15 | 0.64 -> 0.80 | 0.72 -> 0.96 | 0.77 -> 1.04 |
| ballistic + split | 0.92 / 0.41 -> 1.22 / 0.58 | 0.90 -> 1.22 | 0.66 -> 0.84 | 0.75 -> 1.04 | 0.84 -> 1.10 |

* At 60 Hz nearly every scored ball gets its own arc, in volleys too. Recall above 1 is expected: misses are
  launched but never scored.
* Launches credited to idle teammates grow in proportion: about 0.44 per recalled ball at both rates.
* Why 60 Hz helps: in a 20 balls/s stream sampled at 30 Hz, each ball moves about as far between frames as the
  spacing to the next ball. No tracker can then tell which ball is which (prior-art note,
  `data/exp_research/prior_art.md`). At 60 Hz the step is half the spacing.

**Ruled out** on the stored blob logs:

* Balls erased by three-frame differencing in streams: only 4-16 % of the detector's balls have no motion blob.
* Merged blobs: median about 1x a single ball's area in the main view.
* Area-based blob splitting: the expected ball size from camera range is unreliable in the end views.
* The flying-fuel detector as the blob source: too few balls found in the main view.

**Does it make per-team fuel better?** Each arm is checked two ways, on 54 scored scouted Chezy quals (105
alliances):

* A scouting A/B against the defaults of the time (`data/exp_track/tools/track_ab.py`, arms analysed in memory,
  output/ untouched).
* The credit model refitted on each arm's own launch evidence: `build_ds --season`, `fit_cv --specs
  "lin/all+dist" --lam 3e-5` (evidence only, no OPR or season prior), nested leave-one-match-out CV, paired by
  match with `paired_cv.py`.

*Deciding run, with today's defaults* (process-of-elimination linking, BOX_HOLD_S, LINK_FILL_TIE, IDENT_FAR_COST,
`attr_model_elim`; 2026-10-19 01:36):

| arm | A/B best-estimate fuel MAE | A/B video-only fuel MAE | video-only top scorer | refitted fuel MAE | refitted TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 Hz (stored tracks) | 40.49 | 47.39 | 63 % | 41.2 | 0.117 |
| 60 Hz, greedy + split | **39.13** | **46.35** | **68 %** | **39.8** | **0.113** |
| paired Δ (95 % CI) | −1.36 (−2.55 .. −0.19) | −1.04 (−2.07 .. −0.02) | +4.8 pp (+1.5 .. +7.6) | −1.44 (−2.67 .. −0.22) | −0.0040 (−0.0071 .. −0.0007) |

* 34 matches get better and 20 worse.
* By the main session's rule (about 1 ball better on the refit, MAE and TV in the same direction) 60 Hz is adopted,
  and the remaining 66 matches were re-scanned at 60 Hz overnight.

*Earlier runs* (before elimination linking; credit model with priors of the time, or evidence-only where marked):

| arm vs the 30 Hz base of that run | A/B best-estimate MAE Δ (95 % CI) | refitted MAE Δ, evidence-only (95 % CI) |
| --- | --- | --- |
| ballistic + split, 30 Hz | +0.32 (−0.64 .. +1.30) | +0.35 (−0.84 .. +1.62) |
| fragment merge + split, 30 Hz | −0.26 (−1.15 .. +0.57) | −0.07 (−1.09 .. +0.91) |
| greedy + split, 60 Hz | −0.71 (−2.13 .. +0.71) | −0.98 (−2.26 .. +0.33) |
| ballistic + split, 60 Hz | −0.40 (−1.92 .. +1.14) | −0.75 (−2.12 .. +0.66) |

* At 30 Hz the choice of tracker makes no difference.
* 60 Hz was already the best arm before elimination linking, but inside the noise. With elimination linking,
  which ties more arcs to the right robot, its extra arcs pay off.
* The 3-D arc check (`data/exp_arcs/tools/arc_hook.py`) could not be run with today's defaults: the linker's
  candidate tuples changed shape (`ValueError: too many values to unpack`).

**Cost of 60 Hz.**

* A fuel-only re-scan at 60 Hz takes about 340 s wall / 820 s CPU per match with one process. The 30 Hz motion
  pass takes 160-240 s.
* Running 60 Hz only in the seconds before scoreboard jumps saves almost nothing: 6 s windows before each
  increase cover 94 % of a Chezy match's frames, because balls score nearly continuously.

**Seeing it.** `data/exp_track/render/qm24_254_zoom.jpg` shows 254's volley in 2026cc_qm24 at 70.5-73.5 s, today's
tracking next to ballistic at 60 Hz, with every credited arc drawn in its robot's colour.

* At 60 Hz there is about one arc per ball: 27 vs 16 credited in the last 0.8 s at 72.5 s.
* At 70.5 s both arms give the stream to 3847, whose box sits just below 254's. The traced-back arcs carry past
  254's box top down to 3847's. That is a linking error, not a tracking one: the trace should stop at the first
  box top it crosses.

### In plain words

The video analysis used to look at every other frame. When a robot fires 20 balls a second, each ball moves about
as far between those frames as the gap to the next ball, so the tracker could not tell the balls apart. It
followed only about one ball in five, and in volleys even fewer.

Two changes fix that:

* Cutting tracks where they jump from one ball to another, the default since 2026-10-18 evening.
* Looking at every frame (60 Hz). On the moments where only one robot was scoring, this finds an arc for about
  every ball the scoreboard counted, in the end cameras, in the main camera and in big volleys.

Finding the balls is only half of the job: each arc also has to be tied to the robot that shot it. About a third
of the extra arcs went to a teammate standing nearby. Once the process-of-elimination linking was in, the extra
arcs paid off: per-team fuel moved about 1.4 balls closer to the scouts' counts (95 % CI 0.2-2.7), and the top
scorer was right more often.

60 Hz costs about twice the CPU of the fuel scan. Limiting it to the seconds before scoreboard jumps does not help,
because at Chezy balls are scored almost all the time.

* **Wiring:** `pipeline.FUEL_60HZ` (off until the full re-scan is in). With it on, `pipeline.load_fuel60` swaps in
  the 60 Hz tracks wherever a scan is loaded for analysis (`reanalyse`, so finalize, and `link_x_ab.py`). The
  tracks are rebuilt by `autoscout/tracks60.py` from `config.BLOBS60` logs with the greedy tracker, exactly as
  the winning arm did. A match without a log keeps its 30 Hz tracks, and `result.json` records `fuel_hz`.
  * The credit model refit on that arm is `data/exp_attrmodel/attr_model_elim60.json` (evidence only, λ 3e-5).
  * `fuel.BallisticTracker` scales itself to 60 Hz too, but it did not beat greedy.
* **Logs:** 60 Hz blob logs (gzip, about 21 MB a match) are in `data/exp_track/blobs60/`.
* **Tools:** `data/exp_track/tools/` (`track_lab.py`, `track_ab.py`, `rescan60.py`, `render_arcs.py`).
