## Bumper-only robot detector (Colin's suggestion): A/B on the 14 Chezy Champs matches

**Idea.** Box only the visible bumpers (red/blue) instead of whole robots: bumpers sit on the
carpet, so even a partly hidden bumper gives a floor point and its colour gives the alliance,
and the detector never has to guess a robot's hidden extent behind the hub, a referee or the
wall.

**Model.** YOLO26s, classes red_bumper / blue_bumper, fine-tuned from whole-robot v1
(`models/robots_v1.pt`) for 20 epochs at 1280 px → `models/bumpers_v1.pt` (val mAP50 0.936; the
val split includes pseudo-labels). Training data, 3,705 images:

* 4 public bumper-boxed Roboflow sets, 1,701 images: ventura-2025, frc-bumpers-h0p1g,
  seg-rob-2026 (bumper polygons → boxes) and bumper-based-detection-om4u6. Square-stretched
  exports were restored to 16:9.
* Checked and not used: frc-2023-hwue2 v4 and robot-seg-06wv0. Both box whole robots (box aspect
  ~1.2-1.5, vs ~2.3 for bumper boxes).
* 2,004 of our own 2026 webcast frames (no 2026cc frames). Their bumper boxes are cut out of the
  robot pseudo-labels by bumper colour (`scripts/pseudo_bumpers.py`). Robots without a clean
  bumper are greyed out. By eye ~15-20 % of these boxes are partial.

**Scan path.** `Options.bumper_model` (`autoscout/bumpers.py`):

* Grouping: a robot's bumper boxes in one view (one per visible face) are grouped into one
  detection before fusion. 4 % of main-view and 7 % of end-view detections were multi-side groups.
* Robot box: a robot-sized box is synthesised above the bumpers, so shot tracing, OCR and
  appearance run unchanged. OCR reads the bumper crop itself.

**Test.** All 14 2026cc matches, held out from bumper training; Chezy's high main camera is the
best case.

* Same inputs: the frozen scans' calibration, overlay and fuel tracks are reused, and only the
  robot detections are redone with each model (`scripts/rescan_bumpers.py`).
* Same code: every row was analysed with the same analysis code (as of 16:19) by
  `scripts/bumper_experiment.py` (runs `scripts/eval_analysis.py --holdout --snap ...`).
* Same settings: the defaults from before the main session's 16:55 change, i.e. no SHOT_GATE
  launch-arc gate and OCR from cached reads (ocr_budget 700, ocr_per_s 0, ocr_all_cached False).
  A separate bootstrap run, entirely on the 16:19 code, reproduced every identity, correlation
  and top-scorer number exactly.
* Baselines: whole-robot v1 is the frozen `data/eval_snapshot` (`models/robots_v1.pt`). v2 is the
  same scans re-detected with `models/robots_v2.pt`.

| detector | floor point | identity held-out | share corr vs OPR | top scorer | segments / match | mean segment (s) | coverage | 3 robots / alliance / step | end vs main camera (median m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whole robot v1 | box edge + 0.4 m | 86.2% | 0.589 | 39% | 285.1 | 2.9 | 69% | 26% | 0.53 |
| whole robot v1 | box edge | 84.3% | 0.693 | 57% | 303.6 | 2.7 | 69% | 26% | 0.71 |
| whole robot v2 | box edge + 0.4 m | 82.6% | 0.651 | 43% | 305.8 | 2.8 | 72% | 28% | 0.53 |
| bumpers | bumper edge + 0.4 m | 84.7% | 0.536 | 54% | 201.7 | 3.7 | 64% | 21% | 0.49 |
| bumpers | bumper edge | 86.0% | 0.562 | 46% | 232.1 | 3.2 | 65% | 21% | 0.66 |

Column definitions:

* **coverage:** share of 0.5 s match bins in which a team has a tracked position.
* **end vs main camera:** distance between an end-camera fix and the nearest same-alliance
  main-camera fix at the same instant.

Paired bootstrap over the 14 matches (`scripts/bumper_bootstrap.py`), bumpers (+0.4 m) minus
whole-robot v1, 90 % intervals:

| metric | bumpers − v1 | 90 % interval |
| --- | ---: | --- |
| identity held-out | −1.5 pts | −4.4 to +1.5 |
| share correlation | −0.05 | −0.13 to +0.02 |
| top scorer | +14 pts | −4 to +29 |

Whole-robot v2 is worse on identity here: −3.6 pts, interval −6.4 to −0.8.

**Findings**

* **Much less track fragmentation.**
  * Segments per match fall 29 % (285 → 202), and fewer in all 14 of 14 matches (ratio 0.52-0.90).
  * Segments are 28 % longer.
  * Steps with a phantom 4th robot per alliance halve (7.9 % → 3.7 %).
  * The cut is still ~24 % after allowing for the lower coverage.
  * Whole-robot v2 does not reduce fragmentation (306 segments).
* **Slightly more consistent positions.** End- and main-camera fixes of the same robot agree
  better: median 0.49 m vs 0.53 m.
* **Keep the 0.4 m push.** Colin's "bumper bottom-centre, no push" made cameras disagree more
  (0.66 m) and fragmentation worse (232 segments). A bumper box's bottom-centre is the robot's
  near edge, exactly like a whole-robot box's, so `ROBOT_HALF_DEPTH` applies to both.
* **Fewer robots found.** 13-15 % fewer detections, and coverage drops from 69 % to 64 %. In the
  far Chezy main view, small, FUEL-covered or dark bumpers are missed that the whole-robot model
  finds, and some bumpers are boxed one face only.
* **No measurable gain where it counts.** Identity, share correlation and top scorer vs fuel OPR
  all lie within noise of whole-robot v1 (28 alliances).
* **Verdict: not adopted as the default.**
  * It is kept as an opt-in option (`Options.bumper_model`, default None). The default pipeline
    is unchanged.
  * Reason to keep it: it measurably fixes the tracking problem it was meant to fix.
  * Likely next steps: a hybrid, with whole-robot boxes for coverage and the bumper box inside
    each for the floor point and segment linking; and more Chezy-style bumper labels for the far
    main view.
* **Side finding: Chezy main-camera calibration.** Only 70 % of main-view robot boxes back-project
  onto the field, vs 92-100 % in the end views, and the share swings 17-100 % between 10 s windows
  (4-tag pose fits). `analyse` drops those fixes, which costs more Chezy coverage than the choice
  of detector does.
