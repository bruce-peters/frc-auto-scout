# Prior art for dense ball streams and "who shot" (2026-10-18)

Read-only research. Nothing in the pipeline was changed. Three light checks were run on existing files (video frame rate, blob sizes in `2026cc_qm24`, the Statbotics source).

## Summary for Bruce

* **Nobody public has solved per-robot fuel from broadcasts.**
  * 1148's Project Vector uses three iPhones in the stands. It matches TBA alliance totals to about 87 % and gives no per-robot number.
  * 334 fell back to humans tracing the robots.
  * 1678's scouts mark *when* each robot shoots and let the scoreboard count. Spot checks are within 20-30 balls (about 10 %).

  Your 3 % alliance totals beat all of these, and your design is 1678's idea, automated.
* **Dense streams: frame rate is the lever.** In particle tracking, errors are set by ξ = movement per frame ÷ spacing between particles. For a stream, ξ = shot rate ÷ frame rate.
  * At 20 balls/s and 30 Hz, ξ is 0.67, past where even the best trackers fail.
  * The broadcast is true 60 fps, so 60 Hz halves ξ.
  * Error grows about as ξ^2.7, so the gain should land mostly on 20+ ball volleys. Those are 5x under-credited today, which is part of why 5026's video share sits 7 pp under scouted.
* **Who shot: watch the robot, not just the ball.** Arc-linking gains (+3 pp of links) never moved fuel. The only video signal that did was robot-centric (DINO "is shooting"). Humans and sports AI both decide "is this robot shooting now", then count elsewhere.
* **Your yardstick is near its floor.** Two 5026 scouts differ by 45.9 balls per team-match. Gains of 1-3 balls are hard to prove on 54 matches; Label-shooters bursts are the sharper test.
* **New finding from your data.** In `2026cc_qm24`, launch-arc motion blobs are 7-9 px across no matter how close the robot is.
  * Robot boxes range from 130 to 380 px, and the correlation between blob size and box width is -0.12.
  * A real ball is about 15 % of robot width, so it should be 20-55 px. The difference mask returns fragments of near balls.

## Ranked top 10

### 1. 60 Hz with particle-tracking-style linking (feeds exp_track)
* **What.** The fluid-dynamics recipe for many identical particles: keep ξ low, start tracks from 3 frames, score candidates with a two-step look-ahead, assign globally.
* **Evidence.** Kearney & Bewley 2020 (Fig. 6, repeating Ouellette 2006), share of imperfect tracks:

  | ξ | nearest neighbour | 4-frame best estimate | hybrid (3-frame start + look-ahead) |
  | ---: | ---: | ---: | ---: |
  | 0.11 | ~30 % | ~1 % | 0.03 % |
  | 0.45 | – | 44 % | 2.5 % |

  Shake-the-Box predicts from track history and works at about 10x the classical particle density.
* **Mapping.**
  * The ballistic tracker already has assignment and look-ahead; what's left is ξ. 60 Hz takes it from 0.67 to 0.33 at 20 balls/s.
  * `rescan60.py` exists. Score recall by burst size: expect the gain at 20+ balls (0.08 launches/ball today) and a narrower 5x volley gap, which the 30 Hz ballistic tracker didn't narrow (0.21 → 0.22).
* **Effort.** Low. Fuel CPU doubles; cap it by running at 60 Hz only in the 6 s before each scoreboard jump.
* **Live, zero context.** Yes.

### 2. Robot-centric "is it shooting" spotting over time (credit model)
* **What.** An E2E-Spot-style model (RegNet-Y backbone + GRU, one GPU) on each tracked robot's crop plus the region above it, with a motion-mask channel.
* **Evidence.**
  * E2E-Spot: 96-97 mAP within ±1 frame on tennis.
  * SoccerNet 2025 team ball-action spotting: 60.0 team-mAP@1 s vs T-DEED's 51.7. *Which side* did it is the hard part; winners used a joint action-and-team head or a 4 s look-back.
  * 1678's human version (who's shooting × scoreboard) is within about 10 %.
  * Your single-frame DINO (AUC 0.63-0.68) already gave -2.75 fuel MAE in CV.
* **Mapping.**
  * Train on the 265k scout-tapped crops, then add weak labels from every webcast's scoreboard: fit per-robot rates whose lagged sum matches the counter. 3C-Net gained +4.6 mAP from count-only labels.
  * Many events instead of Chezy alone attacks DINO's "memorises robots" failure.
  * It plugs into the existing `fx:` term.
* **Effort.** High, and a new model each game.
* **Live, zero context.** Yes, once trained. Validate leave-teams-out.

### 3. Match tracklets across views over time (feeds exp_multiview and exp_hubentry)
* **What.** Multi-camera trackers of identical objects match over a time window rather than frame by frame, so false ("ghost") stereo matches die out.
* **Evidence.**
  * Shake-the-Box nearly eliminates ghosts. OpenLPT cut them from 110 % to 26 % at 0.125 particles/pixel.
  * GReTA tracks hundreds of overlapping birds or insects.
  * Flydra runs live (Kalman filter, 60 fps, under 40 ms).
* **Mapping.**
  * Pair main-view and end-view tracklets by epipolar residual summed over shared frames, plus launch-time agreement.
  * Fit one 3-D parabola and link it to **fused robot positions, not boxes**, so hidden robots can win.
  * Then match hub entries to launches one-to-one by flight time, as Project Vector does with Jonker-Volgenant assignment.
* **Effort.** Medium to high; already in progress.
* **Live, zero context.** Yes.

### 4. Ball size as a depth cue in the end cameras
* **What.** Known ball diameter ÷ pixel diameter gives distance.
* **Evidence.** Van Zandycke 2022: 1.6 px diameter error at 14-40 px, about 10 % depth error (Hough baseline 24-28 %).
* **Mapping.**
  * 94 % of "teammate behind the winner" cases are in end cameras. There, a ball at the far robot is about 1.5x smaller than one at the near robot.
  * Needs a real size measurement (see the summary): a circle fit on the undifferenced yellow mask at the arc's first points, or a small CNN self-trained on single-candidate launches, whose depth is known.
* **Effort.** Medium.
* **Live, zero context.** Yes. `FUEL_DIAMETER` is fixed.

### 5. Keep the shooter label sticky within a burst
* **What.** Project Vector scores each link partly on "burst continuity" and "hub-bearing consistency". SoccerNet's TBAS-2 aligned team labels over a 4 s look-back.
* **Evidence.** No per-robot numbers are published. Your per-launch teammate picks are 61-65 % right inside multi-second volleys, so idle neighbours leak credit.
* **Mapping.** Run Viterbi with a switch cost over a burst's contested launches, like `identity.assign_absent`. Fit the switch cost leave-one-match-out on the c24 and fresh truth sets.
* **Effort.** Very low.
* **Live, zero context.** Yes.

### 6. Statbotics per-shift fuel EPA as the prior
* **What.** Statbotics' 2026 EPA has per-team `auto_fuel`, `transition_fuel`, `first_shift_fuel`, `second_shift_fuel` and `endgame_fuel`, plus a derived `total_fuel`. I checked this in `backend/src/breakdown.py`. EPA updates after every match.
* **Evidence.** EPA predicts winners 72.0 % of the time vs OPR's 70.1 % (2016-2022).
* **Mapping.**
  * This confirms the schema `season._sb_fuel` marks "unverified".
  * A/B test the pre-match `total_fuel` against `prior_oprs`.
  * The per-shift values fit the alternating hubs.
* **Effort.** Low. Cache it; the API is often down.
* **Live, zero context.** Yes; it's causal.

### 7. Seed tracks from blur shape
* **What.** Fast balls smear into streaks, and a streak shows its direction in one frame.
* **Evidence.**
  * Rozumnyi 2017: standard trackers fail on such objects.
  * BlurBall 2025: blur-aware labels improved every detector it tested.
  * Asaf (Computer Vision Scouting thread) found shutter speed mattered more than frame rate.
* **Mapping.** Blobs already store w and h; elongation is 1.2 at the median and 1.6-2.0 at p90. Use the blob's orientation to seed one-point tracks, fixing "predicts zero velocity and grabs the following ball".
* **Effort.** Low.
* **Live, zero context.** Yes. Weaker while blobs are fragments.

### 8. Confidence-gated human review (live fallback)
* **What.** Project Vector sends only non-confident links to an operator.
* **Evidence.** 1678's human-marked shooters are within 20-30 balls.
* **Mapping.** Queue only ambiguous bursts in the inspector's Label shooters: the top two candidates within a margin, or a hidden teammate. One person clears them during the next match.
* **Effort.** Low.
* **Live, zero context.** Yes.

### 9. Learn each robot's launch signature during the event
* **What.** A shooter's exit height, side offset and speed are fixed. Learn them from that robot's unambiguous launches earlier in the event.
* **Evidence.**
  * Your synthetic check: with exact positions, the true launcher fits at 2-3 px and a decoy at 6-55 px. Real fits lose that to launch-point error: +0.2 pp with a fixed launch point, +3 pp with soft priors.
  * Chen 2009 gets basketball shooting locations from 3-D arcs in broadcast video.
* **Mapping.** Use these per-robot priors in place of `SOFT_PRIOR` in `arcs.arc_fit`.
* **Effort.** Medium.
* **Live, zero context.** Yes; it's causal within the event.

### 10. Fit the stream, not the balls
* **What.** Every ball in a volley follows nearly one path, and about 0.5 s of motion blobs traces it.
* **Mapping.** RANSAC one parabola per stream, take its base as the launch point, report "robot r streaming during [t0, t1]", and let the scoreboard count. This is a fallback for the main camera.
* **Evidence.** No direct benchmark. RANSAC trajectory fitting is standard in ball-tracking patents and multi-view tennis (Sohara 2022). 334 found that following the projected path beat SAM 3's per-ball tracking.
* **Effort.** Medium to low, offline on the blob logs.
* **Live, zero context.** Yes.

## Quick wins
* Run 60 Hz `track_lab` recall by burst size (#1).
* Run Viterbi burst continuity on the c24 and fresh truth sets (#5).
* A/B the pre-match Statbotics `total_fuel` as the prior (#6).
* Add the blob-orientation seed to `BallisticTracker` (#7).
* Live runs: subscribe to TBA webhooks, `upcoming_match` (about 7 min before) to pre-fetch and `match_score` to start processing.
* Filter Label shooters to ambiguous bursts (#8).

## Looked at, not recommended
* **ByteTrack / OC-SORT.** They associate by box overlap (IoU), which is zero between frames for small fast balls. The SMOT4SB 2025 winner had to switch to distance and motion direction.
* **Density or line-crossing counting.** The scoreboard already counts. The per-robot version is your exit-band flux, which passing balls swamped.
* **WASB / TrackNet.** +7.8-16.8 % AP across 5 sports, but they track a single ball, and detection isn't the loss.
* **Single-view 3-D arcs.** Already built (`arcs.py`).
* **Public FRC projects.** None reports per-robot accuracy.

## Links
* Kearney & Bewley 2020: https://arxiv.org/abs/2006.06593
* Ouellette 2006: https://doi.org/10.1007/s00348-005-0068-7
* Shake-the-Box: https://doi.org/10.1007/s00348-016-2157-1
  * DLR page: https://www.dlr.de/en/as/about-us/departments/experimental-methods/shake-the-box-3d-lagrangian-particle-tracking-at-high-particle-densities
* OpenLPT: https://me.jhu.edu/fluidtransportlab/wp-content/uploads/2020/02/Tan2020_Article_IntroducingOpenLPTNewMethodOfR.pdf
* u-track: https://doi.org/10.1038/nmeth.1237
* laptrack: https://laptrack.readthedocs.io
* Stone Soup: https://stonesoup.readthedocs.io
* GReTA: https://arxiv.org/abs/1305.1495
* Flydra: https://arxiv.org/abs/1001.4297
  * code: https://github.com/strawlab/flydra
* E2E-Spot: https://arxiv.org/abs/2207.10213
* SoccerNet 2025: https://arxiv.org/abs/2508.19182
* 3C-Net: https://arxiv.org/abs/1908.08216
* Learning from aggregate responses: https://arxiv.org/abs/2401.11081
* Ball 3-D from diameter: https://arxiv.org/abs/2204.00003
* Fast moving objects: https://arxiv.org/abs/1611.07889
* BlurBall: https://arxiv.org/abs/2509.18387
* WASB: https://arxiv.org/abs/2311.05237
  * code: https://github.com/nttcom/WASB-SBDT
* Chen 2009: https://doi.org/10.1016/j.jvcir.2008.11.008
* Sohara 2022: https://scitepress.org/PublishedPapers/2022/107687
* SMOT4SB: https://arxiv.org/abs/2507.12832
* Statbotics:
  * EPA: https://www.statbotics.io/blog/epa
  * code: https://github.com/avgupta456/statbotics
* TBA webhooks: https://www.thebluealliance.com/apidocs/webhooks
* Chief Delphi:
  * Project Vector: https://www.chiefdelphi.com/t/1148-project-vector-ai-scouting-for-rebuilt/519559
  * 1678: https://www.chiefdelphi.com/t/1678-citrus-circuits-2026-scouting-and-strategy-release/521537
  * 334 post-mortem: https://www.chiefdelphi.com/t/building-ai-assisted-frc-scouting-what-worked-what-failed-and-what-we-learned/522442
  * Computer Vision Scouting: https://www.chiefdelphi.com/t/computer-vision-scouting/511642
  * Rebuilt Scouting Systems: https://www.chiefdelphi.com/t/rebuilt-scouting-systems/510647
* FRC code:
  * https://github.com/derek-pohl/ai-scouting
  * https://github.com/NimbleValley/auto-scout
  * https://github.com/andrewda/frc-livescore
