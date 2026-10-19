# auto-scout

Automated scouting of FRC 2026 **REBUILT** matches from the official match videos linked on
The Blue Alliance. For every team in a match it estimates FUEL scored (by period), tower
points, total points, and the robot's path (autonomous and full match).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for UML diagrams of the whole system (entry
point to exit point, with the commands to run each part), [PLAN.md](PLAN.md) for the design and
the algorithms, and [RESULTS.md](RESULTS.md) for measured accuracy.

## Match inspector (diagnose matches in a browser)

```bash
.venv/Scripts/python -m autoscout.inspector --open      # http://localhost:8765
```

Pick a match on the left (badges flag overlay mismatches and identity problems). The video plays
with what the system saw drawn on top: robot boxes labelled with the team it assigned, launch arcs
in the shooter's colour, and the scoreboard counters it reads. Beside it are a live top-down field
map and a selection panel: click a robot for a zoomed crop that follows it, its track segment,
its bumper-OCR evidence and reads. The timeline below shows counter jumps, each team's tracked
segments (red outline = labelled against its own bumper reads), launches and the fuel credited;
click anywhere to seek. Tabs list teams, identity segments (filter to the contradicted ones),
launches and flags. Links like `http://localhost:8765/#match=2026cagle_qm7&t=20` open a match at
a time. The "How it works" tab renders the architecture diagrams.

## Quick start

```bash
.venv/Scripts/python -m autoscout 2026casnv_qm2            # one match
.venv/Scripts/python -m autoscout --team 5026 --event 2026joh  # a team's matches at an event
.venv/Scripts/python scripts/evaluate.py                    # compare everything processed vs TBA
```

Keys live in `.env` (`TBA_API_KEY`, `ROBOFLOW_API_KEY`). Videos are downloaded with
yt-dlp into `data/videos/`. Output per match goes to `output/<match_key>/`:

| File | Contents |
| --- | --- |
| `report.html` | per-team table + auto-path and full-path plots |
| `result.json` | per-team fuel (total and by period), tower and total points, paths, shots, identity debug |
| `dets.json` | every robot detection with its assigned team (for debugging / rendering) |
| `hub_feats.json` | per-frame fuel activity around each hub (vision-only counter input) |

Watch it work: render an annotated video of a match (team-labelled robot boxes, detected
launch arcs, the scoreboard counters the OCR reads, a live top-down field map and a running
fuel tally per team). Ready-made examples are in `output/videos/`.

```bash
.venv/Scripts/python -m autoscout.render 2026casnv_qm2                 # whole match -> output/2026casnv_qm2/demo_2026casnv_qm2.mp4
.venv/Scripts/python -m autoscout.render 2026casnv_qm2 --t0 60 --t1 95 # a slice (video seconds)
```

## How it works (short)

1. **Overlay** — OCR of the broadcast score bar: match clock (sync), live alliance FUEL counter
   ("N / 100", "N / 360"), active-hub arrows. The counter is exact, so alliance-level totals and
   their timing come from it when present.
2. **Camera calibration** — each of the three webcast views (main wide shot, blue-end and
   red-end cameras) is calibrated automatically from the field AprilTags against the official
   tag layout (pinhole + radial distortion). Gives pixel -> field metres.
3. **Robots** — YOLO detector (trained on public 2023-2026 broadcast datasets), alliance from
   bumper colour. Every detection gets a field position *and* an uncertainty ellipse from its
   camera's geometry (a low end camera is very unsure along its line of sight; the high main
   camera is not). Detections of one robot from different cameras are matched by statistical
   distance and combined with inverse-covariance weights, then fed to a 3-slot-per-alliance
   Kalman tracker that trusts each camera accordingly. Identity by constrained bumper OCR
   (3 candidates per alliance from TBA) + appearance + a DINOv2 probe trained on each team's
   robot in its *other* matches, solved with a Viterbi over slot->team labels in which two
   tracks of one robot may share a team (held-out identity 97 %). While a robot is hidden from
   every camera its position is interpolated between sightings (gaps up to 15 s).
4. **Fuel** — moving-yellow blob detection at 30 Hz, ballistic tracking, shots = rising
   projectile tracks, traced back to the launching robot.
5. **Attribution** — each counter increment is shared among the alliance's robots by detected
   launches in the preceding 0.8-5.5 s (peak 2.8 s), gated by "robot is in its alliance zone" (rule G407).
   A robot no camera can see cannot show launch arcs, so a hidden in-zone robot is credited as if
   it had 2 launches per window (`attribution.HIDDEN`), while one in plain view launching nothing
   gets almost nothing; arcs rising from where a hidden robot is interpolated to be are traced to it.
   The weights come from a small model fitted once to FRC 5026's human scouting (`models/attr_model.json`).
   It uses only that match's video: no OPR or other team history moves balls between teams.
   `result.json` also keeps the older hand-tuned split as `fuel_video`.
6. **Points** — fuel points + tower points from TBA's per-robot breakdown.

## Tests

```bash
.venv/Scripts/python -m pytest tests -q
```

## Labelling who scored (per-team ground truth)

The alliance totals are exact (scoreboard counter), so per-team accuracy only depends on *who*
launched the balls behind each counter jump. The match inspector has a labelling mode for that:

```bash
.venv/Scripts/python -m autoscout.inspector          # open http://localhost:8765, pick a match
```

Click **Label shooters** under the video and type your name. Each item is one *scoring burst*
(the alliance counter went up by N); the clip of the seconds before it plays at half speed.
Press <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> for the robot(s) you saw launch the balls, <kbd>U</kbd> for
can't tell, <kbd>H</kbd> for human player/other, <kbd>Enter</kbd> to save and move on (<kbd>R</kbd> replay,
<kbd>P</kbd>/<kbd>N</kbd> previous/next). The system's own boxes, credits and field map are hidden while
labelling so they can't bias you. Labels are saved per person in `data/labels/<match>__<name>.json`;
two people labelling the same match gives an agreement number too. A match takes ~10-15 minutes.

Score the pipeline against the labels (also appended to RESULTS.md by `scripts/evaluate.py`):

```bash
.venv/Scripts/python scripts/score_labels.py
```

## Processing a new event

```bash
.venv/Scripts/python scripts/download_videos.py --team 5026 --events 2026xyz   # or --events only, for every match
.venv/Scripts/python scripts/run_batch.py --events 2026xyz --shard 0/3          # run 3 of these in parallel (0/3 1/3 2/3)
.venv/Scripts/python scripts/build_dino_probe.py embed                           # DINOv2 crop embeddings (GPU, ~25 s/match)
.venv/Scripts/python scripts/build_dino_probe.py labels                          # training labels from the results
.venv/Scripts/python scripts/build_dino_probe.py probe --causal --name causal          # per-match probes from earlier matches only
.venv/Scripts/python scripts/finalize.py                                         # re-analyse with the probes, reports, RESULTS.md
```

Per match, processing takes ~4-6 min on an RTX 4070 (three in parallel ≈ 2 min/match). Re-running
only the analysis (tracking, identity, attribution) from the cached scan takes seconds:
`python -c "from autoscout import pipeline; pipeline.reanalyse('2026casnv_qm2')"`.

## What the numbers mean

* **fuel** — FUEL that went into the team's *active* hub (the ones that score points), scaled so
  the alliance's three teams add up to TBA's official `hubScore.totalCount`. `fuel_unscaled` is
  the raw split of the live overlay counter (which misses the ~1-5 % counted after the buzzer).
* **fuel_by_period** — auto / transition / shift1-4 / endgame.
* **points** — fuel points + that robot's tower points from TBA's per-robot breakdown
  (fouls are alliance-level and not assigned).
* **shots_detected** — launch arcs traced back to the robot (evidence used for the split).

## Known limitations

* Per-team numbers are a *split* of an exact alliance total. They depend on correct robot
  identity (bumper OCR + appearance, ~89 % agreement on held-out bumper reads) and on launch arcs
  being visible. A hidden robot is credited from where it was interpolated to be and from being
  unseen while in its zone; this is a prior, not a measurement, so when two robots of an alliance
  are hidden at once their split is close to even. Gaps longer than 15 s are not bridged.
* Without the score overlay the alliance total comes from TBA (or, with neither, from the
  experimental vision counter, which is ~35 % off and should not be trusted).
* Robot paths are accurate to roughly ±0.5 m in the main camera; near-side robots hidden behind
  referees are bridged by the end cameras or interpolated.
