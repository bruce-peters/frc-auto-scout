# CLAUDE.md: working on auto-scout

auto-scout turns FRC 2026 REBUILT match broadcasts into per-robot fuel counts. Built for FRC 5026; Bruce (5026's
programming manager) is the owner. Read this before changing anything. [README.md](README.md) covers setup and
use; this file covers how to work on it without breaking what's been learned.

## Hard rules (Bruce's, non-negotiable)

1. **Zero context.** The system must work live at a new event, match by match. Nothing may use an event's later
   matches or per-event fitting after the fact. Models trained once on past data are fine. Example: the DINO
   robot-recognition probe for a match is trained only on that event's *earlier* matches (`pipeline.NO_HINDSIGHT`,
   `data/exp_dino/probe/causal/`).
2. **Per-match evidence only.** Each robot's fuel must be an empirical measurement from that match's video. No
   OPR, season or team-history prior may move balls between teams. `models/attr_model.json` has `g_opr = 0`;
   `pipeline.SEASON_PRIOR` is dead code under that model and must stay unused.
3. **5026's scouting export is private.** `data/scouting/*.csv` is git-ignored and never goes into the repo, a
   release, a demo page or a published artifact. Aggregate metrics computed against it are fine to publish.
4. **Never print or echo the keys in `.env`** (`TBA_API_KEY`, `ROBOFLOW_API_KEY`).
5. **Commit and push only when Bruce asks.** The repo is public: github.com/bruce-peters/frc-auto-scout.
6. **No permanent deletes of Bruce's data.** Move things to the Recycle Bin; he empties it. Ask before deleting
   anything large.
7. **Ask before replacing `models/robots.pt`** (the robot detector).

## Where things stand (2026-10-19, finalize v12)

* **Alliance totals:** 3 % off TBA's official hub counts, from the overlay OCR. The goal was 15 %, so it is met.
* **Identity:** 96.7 % held-out bumper-read agreement on 131 matches.
* **Per-robot fuel vs 5026's scouting** (54 Chezy quals, 105 alliances, nested leave-one-match-out CV):
  * MAE ~39.8 balls per robot-match; the first hand-tuned rule was 50.9;
  * top scorer right 70 %;
  * share error (TV) 0.113;
  * median relative error ~19-20 %, against a target of ~15 %.
  * Noise floor: two scouts differ by ~46 per robot-match. Scout *tap timing* is unreliable at burst level (11 of
    the 20 worst disagreements in `data/exp_visual_qa/report.md` were scouting errors), so match totals are the
    reference, not per-burst taps.
* **Biggest open problem: which robot an arc came from.** In bursts with one shooter, about a third of the
  credited arcs land on a teammate next to the shooter. Recall is no longer the bottleneck. After that: shooters
  no camera sees (still covered by the credit model's `hidden` pseudo-shots).
* **Tried and not adopted (don't redo these without a new idea):**
  * launch-only credit (`attribution.CREDIT_RULE = "launches"`): +8.7 balls even with elimination;
  * the season / OPR prior (breaks rule 2);
  * 3-D arc fits and multi-view triangulation;
  * the ballistic tracker at 30 Hz;
  * the flying-fuel detector;
  * shooting signals (DINO "is shooting", hopper drain as credit features);
  * best-camera dedup;
  * first-cross linking;
  * hub-entry re-timing as the credit split.
  * The `data/exp_*/RESULTS_section.md` files say why.

## Machine and environment

* **Platform:** Windows 11, 24 cores, a 12 GB NVIDIA GPU, a ~70 GB memory commit limit. Disk is tight: check free
  space before writing GBs.
* **Python:** always `.venv/Scripts/python`. The venv sets `OPENBLAS_NUM_THREADS=4` in `sitecustomize.py` (each
  process otherwise commits ~1.6 GB at import). A recreated venv needs it back.
* **Memory:**
  * a rescan/inference process commits ~5 GB, and a full match ~10 GB;
  * check `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory` before starting another;
  * YOLO training must use `workers=0`.
* **Git Bash quirks:**
  * Inline `python - <<'EOF'` heredocs that contain backticks or `${...}` fail ("unexpected EOF"). Write the
    script to a file (scratchpad), then run it.
  * `grep -c $'\r'` miscounts; check line endings with Python bytes.
* **Editing files from scripts:** use `open(p, encoding="utf-8", newline="")` and keep LF. The cp1252 default
  corrupted files once.
* **ffmpeg** is on PATH (Gyan build). The match videos are in `data/videos/<match>.mp4` (1080p, 59.94 or 60 fps).
* **Background loops:** TaskStop kills the outer shell but not `bash script.sh` children. Find those with
  `ps -W`, or kill them by PID.

## Layout

* **`autoscout/pipeline.py`:** the heart of it. `process_match` scans a video into `output/<key>/scan.pkl`
  (detections, fuel flight tracks, bumper crops, OCR cache). `analyse(md, Options())` does everything after:
  * tracking and identity;
  * linking launch arcs to robots;
  * the zone filter;
  * calling the credit split.

  `reanalyse(key)` re-runs `analyse` from the cached scan in seconds.
* **The rest of `autoscout/`:**
  * `attribution.py`: splits each scoreboard jump among the alliance's robots with the learned model.
  * `identity.py`: the Viterbi with absent, duplicate and far-apart states.
  * `fuel.py`: blobs, `FlightTracker`, `split_at_jumps`, `is_shot`.
  * `tracks60.py`: 60 Hz tracks from blob logs, true-60p videos only.
  * `overlay.py`: scoreboard OCR.
  * `tba.py`: TBA client with a disk cache; it uses the cache when there is no key.
  * `inspector/`: the browser viewer.
* **`output/<key>/`:**
  * `result.json`: per team: `fuel`, `fuel_by_period`, `fuel_video`. Also `shots`, `inferred_shots`, `fuel_log`,
    `identity`, `timeline`, `paths`, `fuel_hz` and the counters `shots_elim_box` / `shots_elim_arc` /
    `shots_held_box`.
  * `dets.json`: rows `[t, view, x0, y0, x1, y1, alliance, team, segment, fill, elim_team]` in 1920x1080 pixels.
    Index-based readers depend on that order, so only ever append columns.
* **`data/exp_<name>/`:** one folder per experiment, holding `RESULTS_section.md` (picked up by
  `scripts/evaluate.py` into RESULTS.md) and `tools/*.py`. Only `.md` files and `tools/*.py` are versioned
  (`.gitignore`).
* **Other `data/` folders:**
  * `data/cache/`: TBA JSON, camera calibrations, overlay layouts.
  * `data/exp_track/blobs60/`: 60 Hz blob logs.
  * `data/exp_dino/`: embeddings and probes.
* **`models/`:**
  * `attr_model.json`: the credit model, fitted by `data/exp_attrmodel/tools/export_model.py`; older versions
    are kept as `attr_model_*.json`.
  * `robots.pt`: the robot detector.
  * `fuel.pt`: the ground-fuel map.

## The switches that define today's behaviour

Module-level constants are the experiment knobs. Each has a comment with what it does and what it was worth.

| switch | value | meaning |
| --- | --- | --- |
| `pipeline.NO_HINDSIGHT` | True | causal DINO probe (rule 1) |
| `pipeline.SPLIT_TRACKS`, `GATE_TRACE` | True | cut tracks at jumps; trace the launch gate back to the box top |
| `pipeline.LINK_X_TIE`, `LINK_X_CROSS` | 1.0, True | between overlapping boxes, prefer the box the arc rose out of |
| `pipeline.OVERLAP_ZONE_RULE` | True | between alliances, prefer the robot in its own zone (teleop) |
| `pipeline.LAUNCH_ZONE_WEIGHT` | 0.0 | teleop launches from >1 m outside the robot's own zone are not shots |
| `pipeline.BOX_HOLD_S`, `LINK_FILL_TIE` | 1.5, 3.0 | keep a lost box 1.5 s; prefer the robot whose hopper empties |
| `pipeline.IDENT_FAR_COST` | 20 | two tracks >2.5 m apart can't share a team |
| `pipeline.ELIM_BOXES`, `ELIM_ARCS` | True | process of elimination for "?" boxes and box-less arcs |
| `pipeline.FUEL_60HZ` | True | use 60 Hz tracks when a true-60p blob log exists |
| `attribution.CREDIT_RULE` | "model" | "launches" = launch-only credit (tested, worse) |

## How to test an idea (the A/B recipe)

Never adopt on one match or on in-sample numbers. The standard loop, on the 65 scouted Chezy quals:

1. Put the idea behind a switch, default off.
2. `python data/exp_audit/tools/link_x_ab.py --dst data/exp_<name> --arms ";SWITCH=1;OTHER=2,SWITCH=1"`.
   * Each arm is `NAME=value,...`; `attribution.NAME` works for the attribution module.
   * Arms run in memory from `scan.pkl`, nothing in `output/` changes, and it calls `load_fuel60`.
   * Split the matches over 2 processes with `--matches` (`logs/elim_chain.sh` is a worked example).
3. `python data/exp_audit/tools/arm_dets.py data/exp_<name>`. The harness writes only `result.json`, but
   `build_ds.py` also reads `dets.json`. Arms that change identity need their own (this script re-analyses
   them).
4. `python data/exp_audit/tools/score_arms.py data/exp_<name>`. This gives a quick read with the shipped model,
   paired against base.
5. **Refit the credit model on each arm's evidence.** Changing the evidence without refitting is not a fair
   comparison:
   * `data/exp_attrmodel/tools/build_ds.py <arm dir> ds.pkl --season`;
   * `fit_cv.py ds.pkl --specs "lin/all+dist" --lam 3e-5 --out cv.json`;
   * `paired_cv.py cv_base.json ds_base.pkl cv_arm.json ds_arm.pkl lin/all+dist`, a paired bootstrap over
     matches.
6. **Adopt** when MAE and TV move the same way and the CI mostly excludes zero. Small consistent gains that
   follow a game rule or remove guessing can be adopted inside noise; say so. Then:
   * set the default;
   * `export_model.py ds.pkl models/attr_model.json --spec "lin/all+dist" --lam 3e-5 --cv cv.json`, backing up
     the old model first;
   * `python scripts/finalize.py`, which re-analyses all 131 matches, runs the identity holdout and regenerates
     RESULTS.md, in ~70 min;
   * write `data/exp_<name>/RESULTS_section.md`.
7. **Check every event, not just Chezy.** Only Chezy has scouting, and the 60 Hz change looked great there but
   cut launches by 90 % on FIRST webcasts. Compare launch counts per match across events
   (`len(result["shots"])`) and the fuel-OPR check in RESULTS.md.

Also useful:
* `data/exp_arcs/tools/eval_truth.py` gives per-launch pick accuracy.
* `scripts/identity_holdout.py` measures identity.
* `scripts/score_scouting.py` scores `output/` against the scouting (in-sample: the model was fitted on these
  matches, so quote CV numbers for claims).

## Lessons that cost hours (don't relearn them)

* **Most FIRST webcasts are 30p pictures shown twice in a 60 fps file** (57 of 62; all Chezy and 5 Johnson
  matches are true 60p). Tracking every frame then makes balls go still, jump, still, jump, and `split_at_jumps`
  shreds the tracks. `tracks60.duplicated_pictures` detects it per video. New scans at `fuel_step=1` need the same
  check (`data/exp_track/tools/dup60.py duplicated_frames`).
* **The causal DINO probe can be confidently wrong about a team it has barely seen.** In qm17 it called 6800
  "6665", which gave two 6665 tracks at once with 6800 "absent". `IDENT_FAR_COST` fixed it. Watch for teams with
  few earlier matches.
* **Robots right under an end camera have no box at all**, so their arcs were dropped until `ELIM_ARCS`.
* **Robot photos for demos** must come from crops whose own bumper read is the team. A track label alone showed
  the wrong robot.
* **The scan time base is the video's.** Clips, overlays and crops must use the same `t`. Re-downloading a
  per-match YouTube video gives identical timestamps.
* **The inspector server** needs a restart after `server.py` changes; static files only need a page reload.
  Bruce runs his own viewer: don't navigate or restart his tab or server. Use a separate port
  (`.claude/launch.json` has `inspector-preview` on 8767).

## Working style

* **Bruce:** experienced (sim, vision, scouting), wants empirical results and honest numbers. Lead with what
  changed and the held-out number; say plainly when something didn't work or is within noise.
* **UI** (inspector, reports, demo pages):
  * near-black (`#0a0a0a`), one warm accent (fuel yellow `#ffcb1f`), surfaces told apart by tone steps rather
    than lines or shadows;
  * League Spartan for numerals and names, Figtree for text;
  * every team shown as a bumper plate in its tracking colour;
  * no pill badges, no blue accents in chrome, no glows or gradients, no emoji, no em dashes in UI copy;
  * inferred things are dashed with hollow plates.
* **Write-ups:** plain, specific, numbers with their CI and sample, and what was not adopted and why.
* **Multi-session work:** experiments can run in separate Claude sessions. One session at a time edits
  `pipeline.py`; others propose a switch, and the main session wires defaults and runs finalize. Ask peers for
  their results in a message.

## Sharing and releases

* `python scripts/setup.py [--full] [--videos N] [--cuda]` is the newcomer path. It needs only stdlib, installs
  `.venv`, downloads data packs from the GitHub release `data-v1`, and gets videos from YouTube.
* `python scripts/pack_data.py --tag data-vN` builds the packs into `dist/` (git-ignored):
  * `starter.zip`: results, detectors, caches, probes;
  * `scans-NN.zip`;
  * `fuel60-NN.zip`;
  * `manifest.json`.

  Upload them with `gh release create data-vN dist/data-vN/*`. Bump the tag and `setup.TAG` when `output/` changes
  in a way users should get.
* Never pack `data/scouting/`, `.env`, or match videos (those come from YouTube).
* The replay demo page (https://claude.ai/artifact/57p6rqFgQRjhNeTSjAZ1TM) is built by
  `docs/demo/export_clips.py` (with `--elim`) into `dist/demo/`, next to `docs/demo/auto-scout-replays.html`.
  It is shared with anyone who has the link, so keep scouting data off it.
