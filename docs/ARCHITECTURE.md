# How auto-scout works, from entry point to exit point

This page is the map of the system: which command starts what, which module does each step,
what data flows between them and which files come out at the end. The diagrams are UML
(activity, sequence, class and component diagrams) written in Mermaid. They render on GitHub, in
VS Code with a Mermaid extension, and in the match inspector (`python -m autoscout.inspector`, tab
"How it works"). Static SVG and PNG copies of every diagram are in
[`docs/diagrams/`](diagrams/) (regenerate with mermaid-cli after editing this file).

## 1. Running it yourself

All commands run from the repository root with the project virtualenv.

| You want to… | Command | Entry point |
| --- | --- | --- |
| process one match | `.venv/Scripts/python -m autoscout 2026casnv_qm2 --debug` | `autoscout/__main__.py` → `pipeline.process_match` |
| process a team's matches at events (parallel) | `.venv/Scripts/python scripts/run_batch.py --team 5026 --events 2026casnv 2026joh --shard 0/3` (run `1/3`, `2/3` alongside) | `scripts/run_batch.py` → `pipeline.process_match` |
| download videos only | `.venv/Scripts/python scripts/download_videos.py --team 5026 --events 2026casnv` | `video.download` (yt-dlp) |
| re-run only the analysis (seconds) | `.venv/Scripts/python -c "from autoscout import pipeline; pipeline.reanalyse('2026casnv_qm2')"` | `pipeline.reanalyse` (uses `output/<match>/scan.pkl`) |
| rebuild every report + RESULTS.md | `.venv/Scripts/python scripts/finalize.py` | `scripts/finalize.py` |
| browse / diagnose matches | `.venv/Scripts/python -m autoscout.inspector --open` → http://localhost:8765 | `autoscout/inspector/server.py` |
| annotated video of a match | `.venv/Scripts/python -m autoscout.render 2026casnv_qm2` | `autoscout/render.py` |
| label who shot each scoring burst (per-team ground truth) | inspector → open a match → **Label shooters**; then `.venv/Scripts/python scripts/score_labels.py` | `autoscout/labels.py`, `scripts/score_labels.py` |
| run the tests | `.venv/Scripts/python -m pytest tests -q` | `tests/` |

Prerequisites: `.env` with `TBA_API_KEY` (and `ROBOFLOW_API_KEY` for dataset downloads), the
models in `models/` (`robots.pt`, optional `fuel.pt`), an NVIDIA GPU for speed (CPU works, slowly).

## 2. The whole run, entry to exit (activity diagram)

```mermaid
flowchart TD
    start(["Entry: python -m autoscout MATCH<br/>or scripts/run_batch.py"]) --> tbaq["tba.match(key)<br/>teams by station, YouTube id, score breakdown"]
    tbaq --> dl{"video already in<br/>data/videos/?"}
    dl -- no --> ytdlp["video.download()<br/>yt-dlp, 1080p60 H.264"] --> open
    dl -- yes --> open["video.Video(path)"]
    open --> ovc{"overlay cache<br/>data/cache/overlay_KEY.json?"}
    ovc -- no --> ovs["overlay.scan()<br/>OCR clock + fuel counters + hub arrows, 3x/s"] --> ovt
    ovc -- yes --> ovt["overlay.build_timeline()<br/>match start, teleop start, clean counter series"]
    ovt --> prod["views.detect_production()<br/>FIRST template or Chezy: where the 3 camera views sit"]
    prod --> cal["pipeline.calibrate()<br/>field.calibrate_view_dynamic per view:<br/>AprilTags -> intrinsics + a pose every 4 s"]
    cal --> scan["pipeline.scan_video()<br/>one pass over the match frames (see 4)"]
    scan --> ana["pipeline.analyse()<br/>tracking, identity, shots, attribution (see 5)"]
    ana --> save["write output/KEY/<br/>result.json, dets.json, hub_feats.json,<br/>fuel_map.npy, scan.pkl"]
    save --> rep["report.write_html()<br/>report.html + plots"]
    rep --> stop(["Exit: per-team fuel, points, auto paths<br/>output/KEY/report.html"])
    save -. "later, any time" .-> fin["scripts/finalize.py<br/>reanalyse all + index + RESULTS.md"]
    save -. "browse" .-> insp["autoscout.inspector<br/>local web UI"]
```

## 3. Who calls whom for one match (sequence diagram)

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant CLI as __main__ / run_batch
    participant P as pipeline
    participant TBA as tba (The Blue Alliance API)
    participant V as video (yt-dlp + OpenCV)
    participant O as overlay (EasyOCR)
    participant F as field + views
    participant S as scan_video
    participant A as analyse
    participant R as report

    You->>CLI: python -m autoscout 2026casnv_qm2
    CLI->>P: process_match(key, Options)
    P->>TBA: match(key)
    TBA-->>P: MatchInfo (teams, youtube, hubScore, towers)
    P->>V: download(key, youtube) then Video(path)
    P->>O: scan(video) then build_timeline(rows)
    O-->>P: Timeline (auto_start, teleop_start, counts per alliance)
    P->>F: detect_production(frame) and calibrate(video)
    F-->>P: CameraTrack per view (main, blue_end, red_end)
    P->>S: scan_video(video, info, prod, cams, timeline, opts)
    loop every frame of the match window
        S->>S: robots every 6th frame, fuel blobs every 2nd frame
    end
    S-->>P: MatchData (dets, crops, flight tracks, hub features, fuel map)
    P->>A: analyse(md, opts)
    A-->>P: result dict (teams, paths, shots, identity, fuel_log)
    P->>P: save result.json, dets.json, hub_feats.json, scan.pkl
    CLI->>R: write_html(result)
    R-->>You: output/KEY/report.html
```

## 4. Inside `scan_video`: the per-frame loop (activity diagram)

```mermaid
flowchart TD
    f(["next video frame t"]) --> rb{"every 6th frame?<br/>(10 Hz)"}
    rb -- yes --> yolo["RobotDetector: YOLO on the full frame"]
    yolo --> perbox["for each box: which camera view?<br/>skip boxes in overlay graphics"]
    perbox --> col["robots.bumper_alliance(): red / blue from bumper colour"]
    col --> det["robots.Detection(t, view, box, alliance)<br/>+ hopper fill, crop id"]
    det --> crop["store bumper crop + appearance thumbnail<br/>(for OCR / identity later)"]
    rb -- no --> fz
    crop --> fz{"every 2nd frame?<br/>(30 Hz)"}
    fz -- yes --> blobs["FuelScan: three-frame differencing x FUEL-yellow mask<br/>(+ optional flying-fuel YOLO) -> moving blobs per view"]
    blobs --> ft["fuel.FlightTracker.update()<br/>link blobs into ballistic tracks"]
    ft --> hf["hub regions: count moving blobs over each hub<br/>opening / exits -> hub_feats"]
    fz -- no --> fm
    hf --> fm{"every 60th frame?<br/>(1 Hz)"}
    fm -- yes --> fuelmap["fuel YOLO on main view -> field heatmap"]
    fm -- no --> nxt(["next frame"])
    fuelmap --> nxt
```

## 5. Inside `analyse`: from detections to per-team fuel (activity diagram)

```mermaid
flowchart TD
    in(["MatchData"]) --> pos["field positions: back-project each box's bottom-centre<br/>through its view's camera at time t; covariance from the<br/>camera geometry"]
    pos --> fuse["robots.fuse(): merge same-alliance detections across views<br/>(inverse-covariance weighted)"]
    fuse --> trk["robots.FieldTracker: 3 Kalman slots per alliance;<br/>new segment whenever identity could swap"]
    trk --> ocr["OCR budget: biggest bumper crops -> ocr.read_numbers()<br/>-> ocr.team_evidence() against the 3 TBA teams"]
    ocr --> app["appearance: ResNet-18 embeddings of crops (identity.CNNEmbedder)<br/>+ DINOv2 probe trained on the team's other matches (w_dino)"]
    app --> vit["identity.assign_absent(): Viterbi over slot->team labels<br/>(OCR + appearance + DINO, continuity); two tracks of one robot<br/>may share a team, a team may be absent"]
    vit --> paths["team tracks -> auto / full-match paths"]
    paths --> shots["shots: fuel.is_shot() on flight tracks; launch point -><br/>robot box in same view (the box whose top it leaves), else robot tracked in another view"]
    shots --> gate["launch gate (SHOT_GATE): keep an arc only if it starts just above<br/>the robot, over it, and climbs clear of it - drops hopper rattle,<br/>bounces and human-player throws"]
    gate --> zone["alliance-zone occupancy per team (G407: may only<br/>shoot into the hub from your zone)"]
    zone --> inc{"overlay counter<br/>available?"}
    inc -- yes --> ovinc["attribution.increments_from_counts()"]
    inc -- no --> tbainc["attribution.increments_from_periods()<br/>TBA period totals, timed by the vision hub counter"]
    ovinc --> split["attribution.split_increments(): each counter jump shared by<br/>launches 0.8-5.5 s earlier x zone occupancy (learned credit model,<br/>models/attr_model.json), then a match-level fuel-OPR prior"]
    tbainc --> split
    split --> pts["scale to TBA total; + tower points (TBA per robot)"]
    pts --> out(["result: teams, fuel_by_period, points, paths,<br/>shots, identity, fuel_log"])
```

## 6. Data model (class diagram)

```mermaid
classDiagram
    class Options {
      robot_model, fuel_model
      det_hz, fuel_step, overlay_hz
      ocr_budget, w_ocr, w_app, switch_cost
    }
    class MatchInfo {
      key, event_key
      red: list~int~, blue: list~int~
      youtube: list~str~
      breakdown: dict
      hub(alliance) dict
      teams(alliance) list
    }
    class Timeline {
      auto_start, teleop_start
      counts: dict
      layout
      match_time(vt) float
      video_time(mt) float
    }
    class View {
      name, box, masks
      crop(frame), offset(w, h)
    }
    class Camera {
      K, dist, rvec, tvec
      project(pts3) uv
      backproject(uv, z) xy
    }
    class CameraTrack {
      K, dist, times, rvecs, tvecs
      at(t) Camera
    }
    class MatchData {
      key, prod
      cams: dict~str, CameraTrack~
      timeline: Timeline
      dets: list~Detection~
      crops: dict
      tracks_all: dict
      hub_feats: dict
      ocr_cache, cnn_cache
    }
    class Detection {
      t, view, box, conf, alliance
      xy, cov, crop_id, fill
    }
    class FieldTracker {
      slots: 3 per alliance
      step(t, observations)
      finish() list~Segment~
    }
    class Segment {
      id, alliance, slot, t0, t1
      points, dets
    }
    class FlightTracker {
      update(t, blobs)
      flush() list~FlightTrack~
    }
    class FlightTrack {
      pts: (t,u,v,area)
      velocity0(), fit()
    }
    class TeamSignals {
      team
      shots, shot_w
      zone_t, zone_v
      zone_frac(t0, t1)
    }
    CameraTrack ..> Camera : at(t)
    MatchData "1" o-- "*" Detection
    MatchData o-- Timeline
    MatchData o-- CameraTrack
    MatchData o-- MatchInfo
    FieldTracker "1" --> "*" Segment
    Segment o-- Detection
    FlightTracker "1" --> "*" FlightTrack
    TeamSignals ..> FlightTrack : shots from
    TeamSignals ..> Segment : zone from
```

## 7. Modules and what depends on what (component diagram)

```mermaid
flowchart LR
    subgraph entry["Entry points"]
      main["autoscout/__main__.py"]
      batch["scripts/run_batch.py"]
      fin["scripts/finalize.py"]
      insp["autoscout/inspector"]
      rend["autoscout/render.py"]
    end
    subgraph core["autoscout package"]
      pipe["pipeline.py"]
      tba["tba.py"]
      vid["video.py"]
      ovl["overlay.py"]
      vws["views.py"]
      fld["field.py"]
      rob["robots.py"]
      fu["fuel.py"]
      ocr["ocr.py"]
      idn["identity.py"]
      att["attribution.py"]
      vc["vision_count.py"]
      rep["report.py"]
      opr["opr.py"]
      cfg["config.py"]
    end
    subgraph ext["External"]
      tbaapi[("The Blue Alliance API")]
      yt[("YouTube via yt-dlp")]
      yolo[["ultralytics YOLO<br/>models/robots.pt, fuel.pt"]]
      easy[["EasyOCR"]]
      cv[["OpenCV: video, AprilTags,<br/>calibration"]]
    end
    main --> pipe
    batch --> pipe
    fin --> pipe
    fin --> rep
    insp --> rep
    rend --> ovl
    pipe --> tba --> tbaapi
    pipe --> vid --> yt
    pipe --> ovl --> easy
    pipe --> vws
    pipe --> fld --> cv
    pipe --> rob
    pipe --> fu
    pipe --> ocr --> easy
    pipe --> idn
    pipe --> att
    pipe --> vc
    pipe --> yolo
    rep --> opr --> tba
    tba --> cfg
```

## 8. Offline: training the models and measuring accuracy

```mermaid
flowchart TB
    subgraph train["Train detectors (one-off)"]
      direction LR
      fetch["scripts/fetch_datasets.py<br/>Roboflow broadcast sets"] --> merge["scripts/merge_datasets.py robots|fuel<br/>dedupe, polygons->boxes, mask unlabeled end panes"]
      pool["scripts/sample_frames.py<br/>frames from our videos"] --> pseudo["scripts/pseudo_label.py<br/>confident boxes from the current model"]
      pseudo --> merge
      merge --> tr["scripts/train_yolo.py<br/>YOLO26s, workers=0 on Windows"] --> models[("models/robots.pt<br/>models/fuel.pt")]
    end
    subgraph evals["Measure (after processing matches)"]
      direction LR
      ev["scripts/evaluate.py<br/>overlay vs TBA, team shares vs fuel OPR"] --> res[("RESULTS.md")]
      fvc["scripts/fit_vision_counter.py<br/>vision-only counter, leave-one-out"] --> ev
      idh["scripts/identity_holdout.py<br/>held-out bumper reads"] --> ev
      ea["scripts/eval_analysis.py<br/>analysis changes on frozen scans"]
      lab["inspector: Label shooters<br/>data/labels/match__name.json"] --> sl["scripts/score_labels.py<br/>per-team split vs hand labels"] --> ev
    end
    models --> run["process matches"] --> evals
```

**Per-team ground truth (shooter labels).** TBA and the scoreboard only give alliance totals, so
the per-team split is checked against hand labels. `labels.bursts` cuts each alliance's overlay
counter into scoring bursts (increments at most 1.5 s apart, at most 5 s long). In the inspector's
**Label shooters** mode a labeller watches each burst (from 4 s before the first ball, slowed to
0.5×) and marks which of the alliance's three robots launched those balls (keys 1/2/3, H = no robot,
U = unsure). Labels save straight away to `data/labels/<match>__<labeller>.json`, one file per person
so several people can label the same match. `scripts/score_labels.py` compares the balls
`fuel_log` credited inside each labelled burst with the label and with two baselines that need no
video (an even 1/3 split and a fuel-OPR split). It reports the share of balls credited to a robot
that really shot, the team-match error, the correlation and how often labellers agree, and it writes
`data/exp_labels/RESULTS_section.md`, which `scripts/evaluate.py` appends to `RESULTS.md`.

## 9. What comes out, and where to look

| File | Written by | What it is |
| --- | --- | --- |
| `output/<match>/result.json` | `pipeline.analyse` | per-team fuel (total, by period), tower and total points, paths, launches, identity segments, overlay timeline, credit log |
| `output/<match>/dets.json` | `pipeline.analyse` | every robot detection: time, view, box, alliance, assigned team, segment |
| `output/<match>/scan.pkl` | `pipeline._save_scan` | the expensive scan, so `reanalyse` takes seconds |
| `output/<match>/report.html` | `report.write_html` | per-team table and plots |
| `output/index.html` | `report.write_index` | list of all match reports |
| `output/team_5026.md`, `team_5026_autos.png` | `scripts/team_summary.py` | one team across all its matches |
| `RESULTS.md` | `scripts/evaluate.py` | accuracy numbers |
| `data/cache/overlay_<match>.json` | `overlay.scan` | raw scoreboard OCR reads (delete to force a re-read) |
| `data/labels/<match>__<labeller>.json` | inspector, Label shooters (`labels.save`) | hand labels: who launched each scoring burst (labeled / unsure / other / skip) |
| `data/exp_labels/RESULTS_section.md`, `score.json` | `scripts/score_labels.py` | per-team attribution vs the hand labels |
| `data/videos/<match>.mp4` | `video.download` | the downloaded match video |

## 10. Knobs that matter

| Where | Name | Effect |
| --- | --- | --- |
| `pipeline.Options` | `det_hz`, `fuel_step` | how often robots / fuel are analysed (speed vs detail) |
| `pipeline.Options` | `ocr_budget`, `w_ocr`, `w_app`, `switch_cost` | identity: how many bumper reads, evidence weights, how sticky labels are |
| `pipeline.Options`, `ocr.py` | `ocr_per_s`, `ocr_seg_cap`, `ocr_all_cached`, `ocr.PREP_H` | bumper reads per track (one per 2 s of a long track, up to 40), reuse of cached reads, upscaling before OCR (128 px) |
| `pipeline.py` | `SHOT_GATE`, `LAUNCH_ABOVE_MAX`, `RISE_ABOVE_MIN`, `INBOX_MAX`, `SIDE_MARGIN`, `PROJ_RISE_M` | launch-arc gate: an arc only counts if it starts just above a robot, over it, and climbs clear of it (drops hopper rattle, bounces, human-player throws) |
| `attribution.py` | `LAG_LO`, `LAG_PEAK`, `LAG_HI`, `PRIOR`, `HIDDEN` | launch -> counter delay window, the credit an in-zone robot gets without visible launches, and the pseudo-launches a hidden robot gets |
| `robots.py` | `FUSION`, `CHI2_GATE`, `TRACK_CHI2`, `SWAP_DIST` | multi-camera fusion and tracking |
| `field.py` | `in_alliance_zone(margin)` | how close to the zone line counts as "in zone" |
