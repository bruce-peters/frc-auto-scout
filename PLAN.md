# auto-scout — design plan

Automated scouting of FRC 2026 **REBUILT** matches from the official match videos linked on
The Blue Alliance (TBA). For every team in a match it reports:

* fuel launched and fuel scored in its HUB (count and points), auto vs teleop, per shift
* its autonomous path (field coordinates, 0–20 s) and full-match path
* tower points (from TBA's per-robot breakdown) and total points

The headline number — fuel scored per team — must be accurate. The acceptance bar is:
**predicted alliance fuel totals within ~15 % of TBA's official hub counts** on held-out
matches, with team 5026's matches always in the test set.

---

## 0. Ground truth and data we can get for free

| Source | What it gives | How it is used |
| --- | --- | --- |
| TBA API v3 `/match/{key}` | teams per alliance (station order), YouTube key, `score_breakdown.{red,blue}.hubScore` = `autoCount, transitionCount, shift1..4Count, endgameCount, teleopCount, totalCount, uncounted`, per-robot `autoTowerRobotN` / `endGameTowerRobotN` | ground truth for alliance fuel per period, hub-active schedule, per-team tower points |
| YouTube (yt-dlp) | match video, usually 1080p, with the FMS overlay | the input |
| Roboflow Universe | labeled FRC robot / fuel images | detector training (plus pseudo-labels, §3) |
| robot code in `FRC-2026` | hub centers (4.597, 4.035) and (11.938, 4.035) m, field 16.54 × 8.07 m, trench/bump boxes | field model for homography |

Per-team ground truth for fuel does not exist publicly. Team-level sanity is checked with a
least-squares **fuel OPR** over all matches at an event (alliance fuel = Σ team contributions):
our per-team average fuel/match should correlate with it.

## 1. Pipeline overview

```
TBA match ─► download video ─► match clock sync (overlay OCR / start detection)
                                │
          ┌─────────────────────┼──────────────────────────┐
          ▼                     ▼                          ▼
   field registration     robot detector (YOLO)      fuel detector (YOLO, tiled)
   (homography + camera   + ByteTrack tracklets      + ballistic multi-target tracker
    motion compensation)  + identity stitching       + hub-entry gate per alliance
          │               + bumper OCR (3-way vote)  + launch detection near robots
          │                     │                          │
          └──────────► per-robot field tracks ◄────────────┤
                                │                          │
                                ▼                          ▼
                      shot attribution: hub entries ◄─► robot launches (TOF matching)
                                │
                                ▼
            per-team fuel (by period, active-hub gated) + tower pts (TBA) ─► report
```

## 2. Match clock and hub schedule

* REBUILT match = AUTO 20 s, then TELEOP 140 s split into TRANSITION (10 s, both hubs
  active), SHIFT 1–4 (25 s each, hubs alternate), END GAME (30 s, both active). Exact
  boundaries confirmed from the game manual.
* Which alliance is inactive in shifts 1/3 vs 2/4 is read from TBA (`shiftNCount == 0`
  pattern / rule on auto result) and, if visible, the overlay's hub indicator.
* **Sync:** OCR the overlay match timer (digits, fixed ROI found once per event) on ~2 fps
  samples, fit `video_t = match_t + offset` with RANSAC (the timer pauses between auto and
  teleop, so fit the two periods separately). Fallback: detect the overlay appearing /
  the auto-start buzzer frame by large field motion onset.
* Fuel counted only while that alliance's hub is active (an inactive hub scores 0).

## 3. Training data

1. **Public data**: best Roboflow Universe robot datasets (broadcast/stands view, any year —
   bumpers look the same every season) and REBUILT fuel datasets.
2. **Domain pseudo-labels** from 2026 match videos (frames sampled every ~2 s across many
   events, plus dense bursts around shooting):
   * robots: run the public-data model, keep confident boxes, check bumper colour
     (red/blue HSV fraction in the lower third of the box), reject boxes off the field mask.
   * fuel: HSV yellow mask inside the field+airspace mask → connected components → split
     touching balls with a distance transform and peak finding, where the expected ball
     radius in pixels comes from the homography at that image row. Reject static
     non-fuel yellow by a per-video median background. Verify a sample by eye.
3. Train YOLO (ultralytics, latest small/medium model) with heavy mosaic/scale/blur
   augmentation, imgsz 1280 (robots) and tiled 960-px crops at full resolution (fuel,
   small objects). Iterate: model → re-label → retrain (self-training), checking precision on
   hand-reviewed frames.

## 4. Field registration

* Field plane model in metres, origin at the blue alliance wall corner (WPILib blue origin).
* Per-event reference homography from ≥6 floor-level landmarks (field corners, hub base
  corners, trench/bump corners, centre line). Solved with `cv2.findHomography` (RANSAC),
  checked by re-projecting the full field outline onto the frame.
* Per-video: register the first clean wide frame to the event reference with ORB/SIFT
  matches on the static structure (masked to field elements, not robots/fuel) → `H_video`.
* Within a video: every N frames, re-estimate frame→reference similarity; large residual or
  histogram jump = camera cut / replay / close-up → frames flagged *invalid* and skipped by
  the trackers (they coast).

## 5. Robots: detection, tracking, identity

* Detector classes: `robot` (+ bumper colour → alliance via HSV on the lower box third,
  robust to class confusion).
* **Tracklets:** ByteTrack per alliance on image boxes; the box bottom-centre (bumper
  contact point) is projected through the homography to field metres.
* **Kalman** (constant velocity, field frame) smoothing; gating by max robot speed 5.5 m/s.
* **Identity stitching** (exactly 3 robots per alliance):
  1. Cut tracklets wherever two same-alliance robots come within ~1 m (possible swap).
  2. Appearance embedding per tracklet = mean DINOv2 (fallback: ResNet) feature of robot
     crops (robots look very different from each other).
  3. Bumper OCR votes per tracklet: upscale the bumper strip 4×, CLAHE, EasyOCR digits-only;
     score each read against the alliance's 3 known team numbers (TBA) with a normalized
     edit-distance likelihood; accumulate log-likelihoods.
  4. Constrained assignment: tracklets → {team A, B, C}; temporally overlapping tracklets
     cannot share a team; cost = −(OCR log-lik) + λ·appearance distance to team prototype +
     μ·kinematic gap cost. Solved iteratively (Hungarian per time-slice + prototype update,
     i.e. constrained k-means with 3 labelled clusters).
* Output: per team, field track (t, x, y) at 10 Hz with confidence; auto path = t ∈ [0, 20].

## 6. Fuel: detection, tracking, counting, attribution

* Fuel detector on full-resolution tiles (SAHI-style overlap, merged with NMS).
* **Ballistic tracker** in image space: state (u, v, du, dv), constant-velocity + gravity-like
  term in v (projectile in a near-orthographic view ≈ parabola), Hungarian on
  Mahalanobis distance, tracks born from unmatched detections, deleted after 3 misses.
  Static ground fuel (speed ≈ 0) is separated from moving fuel.
* **Hub-entry gate** per hub: image polygon of the hub's top opening (projected from the 3-D
  hub model through a camera matrix fitted to the homography + known heights, or marked
  once per event). A *score event* = a moving track that enters the opening polygon from
  above with downward velocity and then disappears/stops being visible. Double counts
  suppressed by one-event-per-track.
* **Launch events:** a moving track whose first sighting is within/above a robot box and whose
  initial velocity points away from the robot (upward in image) → launch by that robot.
* **Attribution** of each score event at time t to a robot of that hub's alliance:
  1. if its track (or a ballistic stitch of track fragments) starts at a launch → that robot;
  2. otherwise match to unmatched launches with time-of-flight consistency
     (TOF ≈ distance(robot, hub) / v₀, window 0.3–2.5 s) via Hungarian on |Δt − TOF|;
  3. otherwise split fractionally by each robot's launch rate toward that hub in the
     preceding 2.5 s (robots far from / facing away from the hub get low weight).
* **Bias calibration:** the vision hub-entry count systematically under-counts (occlusion,
  merged balls). A per-period scale factor is fitted on *training* matches only
  (TBA count / vision count) and applied to held-out matches; the raw count is kept too.
* **Optional fusion (reported separately):** if the overlay shows a live score, OCR it to get
  the alliance's official scoring timeline and distribute each increment over robots with
  the attribution weights above. Alliance totals then match the scoreboard; team shares
  come from vision. Vision-only numbers are what the 15 % bar is measured on.

## 7. Points

* fuel points: 1 per fuel in an active hub (auto + transition + active shifts + endgame).
* tower points: TBA per-robot `autoTowerRobotN` / `endGameTowerRobotN` mapped through station
  order → team (official, per robot); vision is not needed for these.
* total per team = fuel points + tower points. Fouls are alliance-level and not assigned.

## 8. Evaluation

* Test set: all of 5026's 2026 matches with video (casnv, cagle, cancmp, joh, cc, catstd) plus
  other events; calibration fitted on a disjoint training set of matches.
* Metrics: per alliance-match absolute % error of total fuel, auto fuel, teleop fuel;
  MAPE and median; fraction within 15 %. Team level: correlation and MAE of per-team event
  averages vs fuel OPR. Identity: OCR assignment accuracy on hand-checked frames.
* Iterate on whichever stage dominates the error (detection recall at the hub, gate
  geometry, double counts, attribution).

## 9. Code layout

```
autoscout/
  config.py        paths, keys (.env), constants
  tba.py           TBA client with on-disk cache
  video.py         yt-dlp download, frame reader, scene-cut detection
  field.py         field model, homography, calibration I/O, drawing
  clock.py         overlay timer OCR, match-time mapping, hub schedule
  detect.py        YOLO wrappers (tiled inference), colour helpers
  robots.py        robot tracking, tracklets, identity stitching
  ocr.py           bumper OCR + team-number likelihoods
  fuel.py          fuel tracker, hub gates, launch / score events
  attribution.py   score-event → robot assignment, per-team tallies
  report.py        JSON / HTML / auto-path plots / annotated video
  pipeline.py      end-to-end match processing
scripts/           dataset fetching, pseudo-labeling, training, calibration, evaluation
tests/             unit tests (geometry, tracker, attribution, OCR scoring)
```

---

## 10. As built (2026-10-17) — what changed from the plan above and why

Findings from looking at the actual 2026 webcasts changed several parts of the design:

* **The broadcast overlay has a live per-alliance FUEL counter** ("92 / 100", "170 / 360"; a
  plain total in playoffs) plus the match clock, the shift ("3 / 6 :16") and active-hub arrows.
  It updates ~2 s after a ball enters the hub (hub sensor + display lag) and stops at 0:00, so it
  misses the ~1-5 % of fuel counted in the 3 s post-match window. The overlay is the
  alliance-level timeline whenever it exists (all FIRST events and Chezy Champs). Vision counting
  at the hub is used for timing when there is no overlay, and is evaluated on its own.
* **Prior art agrees**: nobody has public, working per-ball fuel counting from broadcast video
  (top robots shoot 20-30 balls/s); the teams that got furthest split counter increments among
  the robots seen shooting. That is the attribution design here.
* **Three camera views per frame** (FIRST template: main wide shot on top, blue-end and red-end
  cameras below a stone-arch graphic; Chezy: similar). Each view is processed and all
  observations are fused in field coordinates.
* **Calibration is fully automatic from AprilTags** (no hand-clicked points): the official tag
  layout gives 3-D corners, `cv2.calibrateCamera` fits focal length + radial distortion + pose
  per view. Cameras *drift* during matches (pans of ~40 px), so intrinsics are shared and a pose
  is fitted every 4 s (`CameraTrack`), with a fallback that takes intrinsics from a stable window
  and solves each frame with PnP. Reprojection error 0.5-1.8 px.
* **Robot detector**: YOLO26s, single class "robot", trained on 6,088 public broadcast images
  (5419's 2026 CA broadcast set, 2073's red/blue 2026 set, 2023 broadcast sets, a 2025 overhead
  set). The 5419 frames only label the main view, so their unlabeled bottom panes were greyed
  out before training (otherwise the model learns to ignore end-camera robots). Alliance comes
  from bumper colour (HSV).
* **Identity**: 3 Kalman slots per alliance in field coordinates; segments cut at every possible
  swap; bumper OCR (EasyOCR, digits only, scored against the 3 TBA team numbers of that
  alliance) + ResNet-18 appearance embeddings; Viterbi over the 6 slot->team permutations.
* **Multi-camera fusion**: every detection's floor position gets a covariance from its camera:
  back-projection Jacobian x box-edge pixel noise, plus 0.30 m along the line of sight (where the
  robot centre sits behind its visible edge) and a 0.08 m floor. Same-alliance detections from
  different views are matched by Mahalanobis distance (chi2 9.21, Hungarian) and combined with
  inverse-covariance weights; the tracker gates on Mahalanobis distance and uses the covariance
  as its measurement noise. Held-out identity agreement 83.8 % -> 88.7 % vs. the old 1.6 m
  greedy merge.
* **Occlusion**: a team's position between sightings is interpolated (gaps <= 15 s, held 4 s at
  the ends), so its zone occupancy stays known and launch arcs above a wall can be traced to it.
  In attribution a hidden in-zone robot counts as 2 launches per window (`attribution.HIDDEN`),
  a visible idle one keeps the small prior. Team-share correlation with fuel OPR on held-out
  events 0.630 -> 0.671.
* **Fuel**: moving-yellow three-frame differencing at 30 Hz in every view, greedy ballistic
  tracking, shots = fast rising projectile tracks, traced to the robot box in the same view or
  to a robot tracked from another view (projected into this view). Arcs that do not head for
  the shooter's hub count 35 %.
* **Attribution kernel**: counter increment at t is shared by shots in [t-5.5, t-0.8] s (peak
  2.8 s), gated by alliance-zone occupancy (G407: you may only shoot from your zone).
* **Fuel YOLO** (trained on densely labelled broadcast fuel sets) maps where fuel sits on the
  field (per-match heatmap). In-flight fuel is found by motion, which is more reliable for
  small blurred balls.
* **Flying-fuel YOLO — tried 2026-10-17, off by default** (`models/fuel_flying.pt`,
  `Options.flying_fuel_model`). Trained on the Roboflow airborne-fuel set plus pseudo-labels:
  boxes of the motion blobs on `is_shot` tracks in 49 matches, with all other moving yellow
  greyed out (`scripts/build_flying_fuel.py`; 960x540 native-resolution tiles, inference on full
  frames at imgsz 1920, ~20 ms/frame). Val P 0.81 / R 0.74 / mAP50 0.79. It cleanly separates
  airborne from floor fuel. A/B on the 27 held-out 2026joh + 2026cc matches
  (`scripts/fuel_det_experiment.py`; same analyse(), only the blob source differs; RESULTS.md has
  the table): merging detections into the motion blobs adds no launches and no share-correlation
  gain (top scorer -1 to -3 alliances), so it is not adopted. Tracking on detector blobs alone is
  the lead (+23 % launches, corr 0.574 -> 0.594, top 25 -> 27 of 54) but not significant and
  picked from 10 variants; confirm it on matches that were not used for training before using it.
  The motion blob set is cluttered (~115 blobs per frame); clutter, not missed balls, seems to
  limit the flight tracker.
* **Ballistic flight tracker — tried 2026-10-18, opt-in** (`Options.flight_tracker = "ballistic"`,
  `fuel.BallisticTracker`). In volleys the greedy tracker loses most arcs: a new one-point track
  predicts zero velocity and grabs the *following* ball, and the wide fixed gate lets tracks hop
  between balls (non-smooth fits). The new tracker uses speed-scaled gates, one optimal assignment
  per frame and a two-frame look-ahead to seed new tracks (12/12 smooth arcs on a synthetic
  20-30 balls/s stream vs 0/12). On the v3 scans (82 matches, 65 scouted cc quals,
  `scripts/stream_experiment.py`) launch recall per scouted ball rises 0.42 -> 0.63 (main view only
  0.28 -> 0.55) and credited arcs per ball in 20+ volleys 0.08 -> 0.35, but team shares do not
  improve (OPR corr 0.517 -> 0.524, scout corr 0.775 -> 0.765, both n.s.; 5026 gap unchanged at
  -0.08): the extra arcs land on every robot in proportion, and small bursts get two arcs per ball
  (main + end camera). Not the default; adopting it needs a fuel-only re-scan (`rescan_fuel`).
  The flying-fuel detector does not close the main-view gap (detector-only main-view recall 0.38:
  the balls are too small there), and under the learned credit model no tracker / blob-source
  variant beats greedy (scout TV 0.129-0.133 for all, worse 5026 gap; `scripts/stream_resplit.py`).

---

## 11. Ideas from the team's Slack thread (2026-10-17) and where auto-scout stands

| Idea (who) | auto-scout today | Next step |
| --- | --- | --- |
| Bumper-only boxes train better than whole-robot boxes; YOLO otherwise has to hallucinate the hidden part of a robot (Colin) | Whole-robot boxes. The box bottom-centre is the floor contact only when the bumper is visible; we push it 0.4 m along the camera ray to reach the robot centre | **Tried 2026-10-17** (`autoscout/bumpers.py`, `Options.bumper_model`, `models/bumpers_v1.pt`; RESULTS.md): on the 14 Chezy matches bumper boxes cut track fragmentation 29 % (fewer segments in 14/14 matches) and make cameras agree slightly better, but find 13-15 % fewer robots (coverage 69 -> 64 %), and identity / share vs OPR stay within noise. The 0.4 m push to the robot centre is still needed (the bumper edge is the near edge). frc-2023-hwue2 turned out to box whole robots. Opt-in, off by default; next: hybrid (robot box for coverage, bumper inside it for floor point and linking) |
| Mask the static background, focus on robot fragments (Colin) | Fuel uses three-frame differencing; robots do not use the background | A per-view median background (cameras drift slowly; we already fit a pose every 4 s) could gate detections and cheaply flag occluders (refs, hub) |
| OCR only works on occasional good frames, but any good read is valuable; constrain to the 3 known teams (Colin, jmarkn, ian1) | Exactly this: EasyOCR digits only, fuzzy-scored against the alliance's 3 TBA teams, accumulated over whole track segments | Keep. OCR reads are ~98 % self-consistent; the identity errors come from linking reads across time |
| Frozen DINO features + a small classifier to tell an alliance's 3 robots apart within a match (Colin) | ResNet-18 ImageNet features averaged per segment + nearest-prototype similarity | **Tried 2026-10-17** (`scripts/build_dino.py`, `identity.classifier_evidence_by_view`, `Options.w_cls`): DINOv2-S prototypes were neutral (92.8 % vs 92.7 % held-out identity on 5 matches) and a per-match softmax classifier trained on OCR-confident segments made it *worse* (90-92 %; 80 % on SV qm2), even per camera view and on large crops only. Whole-box broadcast crops are mostly background, so it learns where a robot was, not which robot. Off by default. The remaining identity errors are linking errors through occlusions, so better tracking/occlusion handling (bumper boxes, blind spots, bidirectional re-linking) is the more promising lever. **Re-tried 2026-10-18 across matches** (`scripts/build_dino_probe.py`, `Options.w_dino`, `data/exp_dino/RESULTS_section.md`): a linear probe trained on the event's *other* matches (labels: confident bumper reads) names the right team for 99 % of confident Chezy segments. Alone it adds only +0.6 points of held-out identity, because the remaining errors are duplicate tracks; with duplicate labels allowed (`identity.assign_absent`, `absent_cost=2`) held-out identity goes 91.2 -> 97.4 % (Chezy 89.5 -> 98.7 %). Off by default |
| Whole-match offline stitching instead of greedy front-to-back (Colin); ian1's pipeline: detect all, track with field positions, identity from all bumper reads + schedule, bidirectional gap filling with a visual tracker (ViTTrack), bridge hidden stretches, project a full robot box from the floor position | Detect all frames first (10 Hz), Kalman tracker in field coordinates (forward only), segments cut at possible swaps, **Viterbi over the whole match** for slot→team, bumper reads from the whole match + TBA schedule | Gap filling / occlusion bridging is being done in the camera-fusion session. A backward Kalman pass (RTS smoother) and re-linking segments with future evidence are the cheap versions of ian1's bidirectional idea; ViTTrack/CoTracker are the heavy versions |
| Mark blind spots and refs to reason about where an occluded robot can be (Colin) | Not modelled | Project the hub/tower/trench boxes and detected refs into each view; a robot that disappears next to one is "behind it", which constrains its position while hidden |
| Use Chezy Champs for best-case tests: high camera, whole field visible (Colin) | All 14 of 5026's Chezy matches are processed (main camera calibrated at ~26 m away, 10 m up) | Use Chezy as the clean benchmark when comparing detector/identity changes |
| A general multi-year robot detector is the most reusable piece (Larry) | Detector trained on 2023-2026 broadcast sets; robot-detector-v2 session is adding pseudo-labelled 2026 frames | Evaluate across seasons on held-out events, not just 2026 |
