# auto-scout

Automated scouting of FRC 2026 **REBUILT** matches from the match videos linked on The Blue Alliance. For every
robot in a match it estimates the FUEL it scored (by period), its tower and total points, and its path around the
field, with no scouts in the stands.

How it works, in one breath:
1. It reads the alliance's live fuel counter off the broadcast.
2. It finds and identifies every robot in the three camera views (bumper numbers plus appearance).
3. It follows every ball in the air.
4. It splits each jump of the scoreboard among the robots that launched in the seconds before.

On 54 Chezy Champs quals that FRC 5026 scouted by hand:
* **Alliance totals** are within 3 % of the official count.
* **Robot identity** is right 96.7 % of the time.
* **Per-robot fuel** is about 40 balls off per robot-match. Two human scouts watching the same robot differ by
  about 46.

[RESULTS.md](RESULTS.md) has the full measurements, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) the system
diagrams, and [PLAN.md](PLAN.md) the design. The [replay page](https://claude.ai/artifact/57p6rqFgQRjhNeTSjAZ1TM)
shows ten annotated match clips. Working on the code with Claude Code? [CLAUDE.md](CLAUDE.md) has the project
rules and workflows.

## Get started

You need git, Python 3.10-3.12 and about 5 GB of disk (35 GB with `--full`). No GPU is needed to browse or
re-analyse matches. An NVIDIA GPU makes processing new videos much faster.

```bash
git clone https://github.com/bruce-peters/frc-auto-scout.git
cd frc-auto-scout
python scripts/setup.py
```

Video downloads may print a yt-dlp warning about a JavaScript runtime. It is harmless; if YouTube downloads start
failing, install [deno](https://deno.com) and run setup again.

On Windows, clone to a short path such as `C:\code\frc-auto-scout`. PyTorch installs files nearly 200 characters
deep, and unless [long paths are enabled](https://pip.pypa.io/warnings/enable-long-paths), a path past Windows'
260-character limit makes the install fail.

`setup.py` does the following:
* creates `.venv` and installs `requirements.txt`;
* downloads the **starter pack** (~0.35 GB): results for 131 matches, the trained detectors, cached TBA data and
  camera calibrations, and the robot-recognition probes;
* writes `.env`.

It is safe to re-run. Options:

| command | adds |
| --- | --- |
| `python scripts/setup.py --full` | the **full pack**: cached detection scans for all 131 matches (~10 GB) and the 60 fps ball logs (~1.7 GB), plus every match video from YouTube (~22 GB). Lets you re-run any part of the analysis. |
| `python scripts/setup.py --videos 5` | the first 5 match videos (or `--videos all`, or `--videos 2026cc_qm54,2026cc_qm17`) |
| `python scripts/setup.py --cuda` | the CUDA build of PyTorch, for an NVIDIA GPU |

Then use the environment it made: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS and
Linux. The commands below assume it is active.

### Things to try

**Browse every match** with everything the system saw drawn on the video. A match's video plays once you have
downloaded it with `--videos`.

```bash
python -m autoscout.inspector --open        # http://localhost:8765
```

**Re-run a match's analysis** (tracking, identity, fuel credit) from its cached scan. This needs the full pack,
takes seconds, and needs no GPU or video:

```bash
python -m autoscout 2026cc_qm54 --reanalyse
```

**Change something and see what moves.** Most behaviour hangs off module-level switches in
`autoscout/pipeline.py` and `autoscout/attribution.py`, each with a comment saying what it does and what it was
worth:

```python
from autoscout import pipeline
pipeline.ELIM_ARCS = False                      # turn off process-of-elimination for box-less launch arcs
res = pipeline.reanalyse("2026cc_qm54")         # rewrites output/2026cc_qm54/result.json
print({t: v["fuel"] for t, v in res["teams"].items() if str(t).isdigit()})
```

**Score everything** against the official totals and fuel OPR, which regenerates RESULTS.md:

```bash
python scripts/evaluate.py
```

**Render an annotated video** of a match slice. This needs the match video:

```bash
python -m autoscout.render 2026cc_qm54 --t0 110 --t1 140
```

**Process a match that is not in the pack.** Put a free TBA API key in `.env`
([thebluealliance.com/account](https://www.thebluealliance.com/account)). This downloads the video and runs the
full pipeline, at ~5 min a match on a GPU:

```bash
python -m autoscout 2026cc_qm1
```

**Run the tests:** `python -m pytest -q tests`

### What is not included

* **FRC 5026's scouting export** (`data/scouting/`) stays private. Tools that score against it
  (`scripts/score_scouting.py`, `data/exp_attrmodel/tools/build_ds.py` and the A/B harnesses in `data/exp_*/tools`)
  need a scouting CSV of your own in the same format (`autoscout/scouting.py`). The credit model those tools fit
  ships ready-made in `models/attr_model.json`.
* **Detector training sets** (`data/datasets/`, from Roboflow): rebuild them with a `ROBOFLOW_API_KEY` and
  `scripts/merge_datasets.py`.
* **DINOv2 crop embeddings** (`data/exp_dino/emb/`, 4.4 GB): only needed to retrain the robot-recognition probes,
  which ship in the starter pack. `scripts/build_dino_probe.py embed` rebuilds them.

### Repository map

| path | what |
| --- | --- |
| `autoscout/` | the pipeline: `pipeline.py` (scan + analysis), `attribution.py` (fuel credit), `identity.py`, `fuel.py`, `robots.py`, `overlay.py`, `tba.py`, `inspector/` (the browser viewer) |
| `scripts/` | batch runs, evaluation, setup and data packing |
| `data/exp_*/` | one folder per experiment: a `RESULTS_section.md` write-up and its `tools/` |
| `output/<match>/` | per-match results (from the data packs, or written when you process a match) |
| `models/` | the fitted credit model (`attr_model.json`) and detector weights (from the starter pack) |

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

## Command line

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
   robot in the event's *earlier* matches (so it works live), solved with a Viterbi over slot->team
   labels in which two co-located tracks of one robot may share a team but two tracks metres apart may
   not (held-out identity 96.7 %). While a robot is hidden from every camera its position is
   interpolated between sightings (gaps up to 15 s).
4. **Fuel** — moving-yellow blob detection, tracked at 60 Hz where the webcast has 60 real pictures a
   second (every Chezy match, some Championship ones) and at 30 Hz on webcasts that show each picture
   twice; shots = rising projectile tracks, traced back to the robot box they rise out of. A robot or
   arc nothing identifies goes, by elimination, to the one robot of its alliance not seen elsewhere.
5. **Attribution** — each counter increment is shared among the alliance's robots by detected
   launches in the preceding 0.8-5.5 s (peak 2.8 s), gated by "robot is in its alliance zone" (rule G407).
   A robot no camera can see cannot show launch arcs, so a hidden in-zone robot gets a few pseudo-launches
   per window, while one in plain view launching nothing gets almost nothing. The weights come from a small
   model fitted once to FRC 5026's human scouting (`models/attr_model.json`).
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
`python -m autoscout 2026casnv_qm2 --reanalyse`. A 60 fps fuel re-scan (`data/exp_track/tools/rescan60.py`) adds
~5 min a match and only helps webcasts with 60 real pictures a second (`autoscout/tracks60.py` checks).

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
  identity (bumper OCR + appearance, 96.7 % agreement on held-out bumper reads) and on picking the
  right robot for each launch arc: in one-shooter bursts about a third of the arcs still go to a
  teammate standing next to the shooter. A hidden robot is credited from where it was interpolated to be and from being
  unseen while in its zone; this is a prior, not a measurement, so when two robots of an alliance
  are hidden at once their split is close to even. Gaps longer than 15 s are not bridged.
* Without the score overlay the alliance total comes from TBA (or, with neither, from the
  experimental vision counter, which is ~35 % off and should not be trusted).
* Robot paths are accurate to roughly ±0.5 m in the main camera; near-side robots hidden behind
  referees are bridged by the end cameras or interpolated.
