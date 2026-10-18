## Robot detector v2 (self-trained on our 2026 webcasts): A/B on 30 matches

**What changed.** YOLO26s fine-tuned from v1 (`runs/robots_v2`, imgsz 1280, 8 epochs) on the 6,088 public
images plus 2,540 frames of our own 2026 webcasts (`data/frame_pool`, all six events) pseudo-labelled by v1
(`scripts/pseudo_label.py --panes --hi_end 0.40`). The two low end-camera views were also detected on 2x
zoomed crops, where v1 finds many robots it misses at full-frame scale; end-view boxes need conf >= 0.40 and
a clear red/blue bumper (an 80-crop check of that band was ~95 % real robots), main-view boxes >= 0.55;
unsure boxes are greyed out. 12,691 boxes.

* Public val split (623 images, same split as v1): v1 mAP50 0.909 / mAP50-95 0.625, v2 0.908 / 0.626 (no regression).
* 20 fresh frames, 4 per event, not in the training pool (`data/debug/robots_v1_vs_v2_<event>.jpg`): robots
  found per frame, main view 4.20 -> 4.20, blue-end view 0.50 -> 0.70, red-end view 0.60 -> 0.75.

**Match-level A/B.** 30 matches (6 per event, `data/exp_robots_v2/subset.txt`) re-scanned robot-only with
each detector (`scripts/rescan_robots.py`; calibration, overlay and fuel reused from the cached scans), then
both analysed back to back with the same code (`scripts/compare_detectors.py`: autoscout sources a4b8fda64d44,
covariance fusion, `w_cls` 0). The v1 re-scan reproduces the production detections exactly.

| metric | v1 | v2 | v2 - v1 (95 % CI, paired bootstrap over matches) |
| --- | ---: | ---: | --- |
| held-out identity agreement (`identity_holdout.py`) | 85.6 % | 87.5 % | +2.0 pts [-0.8, +4.7] |
| within-alliance share corr vs fuel OPR (`evaluate.py`) | 0.751 | 0.754 | +0.003 [-0.036, +0.045] |
| top scorer = top fuel-OPR team | 51.7 % | 58.3 % | +6.7 pts [-5.0, +16.7] |
| mean abs(video share - OPR share), fuel per team-match | 37.3 | 38.2 | +0.9 [-2.8, +4.4] |
| robots observed, of 3 per alliance per 10 Hz step | 71.6 % | 74.1 % | +2.5 pts [+2.0, +3.0] |
| end-view detections per 10 Hz step | 1.32 | 1.68 | +27 % [+22, +35] |
| tracker segments per match | 184 | 190 | +6 [-4, +15] |

Held-out identity by event: Glendale 91.3 -> 92.9, NorCal 87.4 -> 84.7, Silicon Valley 80.7 -> 87.4,
Chezy 80.9 -> 82.3, Johnson 89.5 -> 90.6.

**Decision.** v2 meets the adoption rule (neither held-out identity agreement nor share correlation got
worse). The share-error and segment changes are within noise. Replacing `models/robots.pt` (still v1) with v2 (`models/robots_v2.pt`) is waiting on the user's OK. Since 2026-10-17 ~19:40 every `output/` scan is a robot-only re-scan with
v2 (`scripts/rescan_robots.py --model models/robots_v2.pt`), on the user's instruction, after a clean re-check:
38 matches re-scanned with v1 and v2 under identical code (launch gate, new bumper OCR) gave held-out identity
86.0 vs 87.3 % and share corr vs fuel OPR 0.807/0.687 vs 0.765/0.737 (train/test events); against 5026's human
scouting (51 team-matches) fuel MAE 48.9 vs 47.2 (`data/exp_detab/`).
