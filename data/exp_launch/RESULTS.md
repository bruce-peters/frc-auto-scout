# Per-robot launch signals vs 5026 scouting (data/exp_launch)

Question: which per-robot, per-second signal best predicts how many balls a robot launches?

## Setup

- **Truth:** 5026 scout taps (ScoutRow.shots), time-valid rows only, averaged over the scouts of a robot.
  - Coverage: 65 scouted Chezy quals; 105 fully scouted alliances (54 matches with an overlay counter) for match-level splits and the credit-model CV.
  - Taps map to video time as `auto_start + tap - 0.8 s` (continuous; what build_ds.py uses). The scoreboard check confirms it: the counter lags the taps by +2.75 s in auto and +2.50 s in teleop. With TL.video_time (which skips the auto->teleop pause) the teleop lag comes out at -0.75 s, so the scouting app's clock runs through the pause.
- **Inputs:** v5_snapshot (DINO identities), the v5 vol_signal cache, DINOv2 crop embeddings, and the broadcast video (flux only). There were no output/ reads, except scan.pkl for the crop -> detection map and the camera models (scans unchanged since 07:12).
- **Window metrics** (tools/leval.py, 4 s windows with >= 3 scouted balls for the alliance; each signal shifted by its best lag):
  - mis: share of balls misassigned within the alliance (even 1/3 where the signal is empty);
  - cover: share of scouted balls in windows where the signal has anything for the alliance;
  - corr: pooled within-alliance share correlation;
  - top: top-robot agreement;
  - rho: robot-window Spearman.
- **Match metrics:** the signal's per-match totals as the alliance split vs scouted shares.
- **Credit-model gain** (tools/fit_cv_launch.py, lcv_paired.py):
  - Each signal is added to the deployed spec lin/all+dist+mlprior as a factor (1 + s)^g on the robot's per-jump weight ("fx:"), or added to the launch evidence ("ax:").
  - s = kernel-weighted per jump, like s_beq.
  - Leave-one-match-out at the same fixed λ 3e-5 as models/attr_model.json, paired bootstrap over matches.
  - Baseline TV 0.1301 / fuel MAE 43.8 (= your deployed model's fit).

## Scout noise sets the floor at window level

Two scouts on the same robot (211 robot-matches; all-3-double-scouted alliances for the split):

| | 4 s windows | 8 s windows |
| --- | --- | --- |
| robot-window Spearman, scout A vs scout B | 0.39 | 0.51 |
| misassigned, scout set A vs set B | 43.8 % | 36.9 % |
| share corr | 0.37 | 0.43 |

Per-window scout timing is about as noisy as our signals. Window metrics can only screen; the match-level split and the credit-model CV decide.

## Ranking

- **Primary column: credit-model gain.** Each signal is added to the deployed model; nested leave-one-match-out on 105 scouted alliances; λ 3e-5; paired 95 % CI over 54 matches. Negative ΔTV / ΔMAE = better.
- **Window / match columns:** the signal alone (4 s windows, within alliance).
- **Other events:** the model fitted on Chezy, scored on cagle / cancmp / casnv / joh against split-half fuel OPR (the OPR prior is common to both arms): Δ share corr, Δ fuel MAE.

| signal | window corr | robot-window ρ | alone: match TV | + model ΔTV | + model Δ fuel MAE | Δ top | other events (corr / MAE) |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| credited launches (current evidence) | 0.377 | 0.369 | 0.224 | baseline: TV 0.1301, MAE 43.8 | | | |
| **DINO centred + hopper drain (raw)** | - | - | - | **-0.0081 [-0.0159, -0.0004]** | -2.41 [-5.59, +0.65] | +5.7 | -0.002 / +0.67 (n.s.): **passes** |
| DINO raw + hopper drain (raw) | - | - | - | -0.0095 [-0.0182, -0.0005] | -2.63 [-5.85, +0.70] | +6.7 | -0.003 / **+1.66 [+0.03, +3.31]: fails** |
| DINO "is shooting", raw | 0.259 | 0.293 | 0.220 | -0.0076 [-0.0161, +0.0010] (leak-free) | -2.30 [-5.35, +0.76] | +6.7 | -0.012 / **+1.96 [+0.15, +3.87]: fails** |
| DINO raw, leave-teams-out | (crop corr 0.181) | | | -0.0039 [-0.0109, +0.0028] | -1.05 [-3.54, +1.38] | +5.7 | |
| DINO raw, robot-demeaned prediction | 0.148 | 0.232 | 0.290 | -0.0050 [-0.0128, +0.0028] (leak-free) | -1.99 [-4.89, +0.94] | **+8.6 [+1.0, +15.9]** | -0.013 / +1.06 (n.s.) |
| DINO centred (identity removed) | (crop corr 0.169) | | | -0.0051 [-0.0124, +0.0022] | -1.93 [-4.65, +0.77] | +7.6 | -0.014 / +1.20 (n.s., leans worse) |
| DINO centred, leave-teams-out | (crop corr 0.149) | | | -0.0047 [-0.0121, +0.0029] | -1.78 [-4.56, +0.98] | +6.7 | |
| hopper drain, raw yellow fraction | 0.149 | 0.158 | 0.285 | -0.0049 [-0.0112, +0.0017] | -1.31 [-3.79, +1.32] | +2.9 | +0.010 / +0.07, top -3.8 pp (n.s.) |
| hopper drain + ball-equivalent volume | | | | -0.0063 [-0.0133, +0.0015] | -1.60 [-4.47, +1.47] | +1.9 | +0.005 / +0.18, **top -4.2 pp [-8.3, -0.4]** |
| ball-equivalent volume (cycle 28) | 0.273 | 0.290 | 0.271 | -0.0020 [-0.0069, +0.0033] | -0.68 [-2.55, +1.31] | -1.0 | no worse (cycle 28) |
| hopper drain, per-view normalised | 0.106 | 0.139 | 0.309 | -0.0015 | -0.61 | +1.0 | |
| robot stationary in its zone | 0.331 | 0.380 | 0.236 | +0.0005 | -0.04 | +1.0 | |
| rising box-linked tracks (count) | 0.193 | 0.215 | 0.299 | +0.0002 | +0.00 | -1.9 | |
| exit-band yellow flux (video) | 0.222 | 0.248 | 0.365 | +0.0003 | +0.00 | 0.0 | |
| hub arrivals x in-zone (position only) | 0.457 | 0.555 | 0.173 | n/a: the zone / PRIOR terms the model already has | | | |
| scout A vs scout B (ceiling) | 0.37 | 0.39 | | | | | |

### Signal definitions (tools/lsig.py, dino_*.py, flux.py)

- **DINO "is shooting":**
  - Ridge (λ 30) from the frozen DINOv2 crop embeddings (box_inner + box_cls, 768-d; data/exp_dino/emb) to the scouted launch rate in [t, t + 2 s].
  - Trained on 265k crops of robots with timed scouting, leave-one-match-out (leak-free double leave-out checked: identical).
  - Per robot: 0.5 s bins, mean prediction over the bin's crops.
  - Held-out per crop: corr 0.246, AUC 0.684 (rate >= 0.5 balls/s); within a robot-match over time, corr 0.21.
  - "centred": each robot's mean embedding per view in the match subtracted first (identity removed): per crop 0.169, over time 0.204.
- **Hopper drain:** dets.json fill (yellow fraction inside the box, per detection), per view median-of-3, on a 0.25 s grid; drain = positive decrease per step; views combined by max. "raw" = not normalised; the per-view p95 normalisation hurts.
- **Exit-band flux:** the scan's moving-yellow mask, every 30 Hz frame, in the analyse step (1) link band above each box (nearest box wins shared pixels), divided by one ball's area at that range. 38-60 s per match on CPU.
- **Ball-equivalents:** vol_signal (cycle 28).
- **Rising count:** the same tracks, counted.
- **Stationary:** in zone and moved < 0.25 m in 0.5 s.
- **Hub x zone:** counter increments 2.7 s back, split over in-zone robots.

## Findings

1. **Most of the error is in the attribution among robots, not in missing launches.**
   - Credited launches leave 15 % of scouted balls in 4 s windows where the alliance has no credited launch at all (5.5 % at 8 s).
   - Rising box-linked tracks are present in the windows that hold 99.5 % of scouted balls, but at window level every launch-type signal splits the alliance only a little better than even (misassigned 44-49 % vs 46 % even).
   - That is the same as two scouts agree with each other (43.8 %).
2. **DINO appearance is the strongest new information.** In the cc_qm24 120-130 s volley, 254 has 0 credited launches and 56 scouted balls, and DINO gives it 13.6 vs 1.8 / 2.9.
   - But the raw ridge partly memorises Chezy robots: leave-teams-out halves its gain, and it fails the other-event check.
   - Centring each robot's embeddings removes the memorisation (leave-one-match-out ≈ leave-teams-out), keeping about TV -0.005 / top +7 pp.
3. **Hopper drain (DRAIN_WEIGHT is 0 in production) does carry signal.**
   - Raw and smoothed, as a factor it is the best learning-free candidate and it transfers.
   - Combined with centred DINO it gives the only candidate that is significant on Chezy and within noise on the other events.
4. **Flux, rising-track counts and stationarity add nothing.** Untracked flux is swamped by passing and falling balls: in cc_qm24 120-130 s, 5507's band sees 5,105 vs 254's 1,161.
5. **The volume factor (cycle 28) is subsumed by DINO.** With both in the model, g_vol fits to -0.03.

## Caveats

- About 15 specs were tried on the same 54 matches. The winner's CI only just excludes 0, so treat it as promising, not proven.
- The other-event check is against team-level OPR. A per-match signal can be right and still disagree with OPR.
- A fresh scouted event would be the real test.

## (superseded by the production port below) If adopting DINO centred + hopper drain

- **No rescan.** It reads:
  - data/exp_dino/emb/<match>.npz (box_inner, box_cls, ids, t, view; already produced for DINO identity by scripts/build_dino_probe.py embed);
  - md.dets[i].crop_id (crop -> detection) with det_seg / team_of_seg (the team per detection);
  - md.dets[i].fill (already computed in the scan; dets.json column 9).
- **Weights:** data/exp_launch/dinoc_ridge.npz (mu, sd, w: 769 floats, trained on all scouted Chezy crops, centred).
- **Per match:**
  - centre each team's crops per view;
  - rate = ((x - mu) / sd) . w;
  - 0.5 s bins (mean over the bin's crops, clip >= 0, x 0.5) -> events at bin + 0.75 s;
  - drain events at grid t + 0.5 s;
  - s_x = sum(kernel(t_jump - t_ev) * w_ev);
  - robot weight *= (1 + s_dinoc)^g1 * (1 + s_drainraw)^g2.
- **Refit all parameters jointly:** spec lin/all+dist+mlprior+fx:dinoc+fx:drainraw at λ 3e-5. Fold means: lp -2.54, lh 3.93, lb -0.40, gz 0.42, g_dist -0.49, g_ml 0.29, g_dinoc 1.59, g_drainraw 4.69 (a full-data export is still to do).
- Fitter: tools/fit_cv_launch.py (generic fx:/ax: terms).

## Production port (DINO centred + raw hopper drain)

### Shipped ridge: models/shooting_ridge.npz (copy of data/exp_launch/shooting_ridge.npz; tools/shooting_ridge.py)

- **Trained on:** 264,971 labelled robot crops of the 65 scouted Chezy quals (2026cc) with 5026 scout taps.
  - Time-valid rows only, averaged over a robot's scouts.
  - launch = auto_start + tap - 0.8 s.
  - Label = that robot's scouted launches in [t, t + 2 s], per second.
  - Team per crop from the v5 (DINO-identity) results.
- **x:** box_inner + box_cls (768-d) of frozen DINOv2 ViT-S/14 (data/exp_dino/emb).
  - **Centred:** the robot's mean embedding over its crops in the same view in the same match is subtracted. Causal across matches; it uses only the current match.
  - Then standardised (mu, sd over the training crops).
- **Fit:** ridge λ 30, intercept unpenalised. Held-out per-crop corr 0.169, AUC 0.629; within a robot-match over time, corr 0.204.
- **File contents:** mu, sd, w (769), plus W_loo (65 x 769) leave-one-match-out weights with loo_keys.
  - The W_loo weights are used only when build_ds.py builds a training match, so nested CV on the scouted quals stays honest.
  - Production uses w.
- sha256 21935b0fdf022021ff759ba62f2317d740b58f9c7452d1fed51afd51347ea3fd.

### Functions: data/exp_launch/prod_signals.py

- `dinoc_series(md, emb_npz, ridge_npz, det_seg, team_of_seg, grid=None, exclude_key=None) -> {team: (grid, value)} | None`
  - Reads md.dets (crop_id) -> team_of_seg[det_seg[id(d)]], and emb npz ids, t, view, box_inner, box_cls.
  - Returns None when the match has no embeddings: skip the factor (S = 0 -> factor 1).
- `drain_series(md, det_seg, team_of_seg, grid=None) -> {team: (grid, value)}`
  - Reads md.dets t, view, fill (rounded like det_rows) and the team.
- `grid` defaults to `np.arange(md_t0(md), md_t1(md), 0.5)`, the TeamSignals grid. Values are already moved to launch time (+0.5 s), so store them like dist_t / dist_v.
- `jump_feature(grid, value, t, form)` gives the per-jump S.

### Canonical tools (data/exp_attrmodel/tools; backups of the 12:06 versions in data/exp_launch/backup_canonical/)

- **build_ds.py:** new features s_dinoc, s_drainraw (window form) and s_dinock, s_drainrawk (kernel form), through prod_signals.
  - Crop -> dets.json row: column 10 if the pipeline writes crop_id there; otherwise the cache data/exp_launch/cropmap/<key>.npy (validated by each crop's time and view); otherwise output/<key>/scan.pkl (read-only, then cached).
  - Matches without embeddings or a valid map get dinoc 0. The build log prints "DINO shooting signal in N/131 matches".
- **fit_cv.py:** spec extras `fx:<name>` (lw += g_<name> * log1p(s_<name>)) and `ax:<name>` (evidence += exp(lc_<name>) * s_<name>).
  - A dataset without s_<name> gives factor 1. Checked on ds_v5.pkl: the spec equals the baseline, g = 0.
- **export_model.py:** accepts +fx:dinoc+fx:drainraw (window) or +fx:dinock+fx:drainrawk (kernel).
  - Writes version 2, g_dinoc, g_drain and a "shooting" block: form, ridge path and sha256, embedding, centring, grid_step 0.5, dino_shift 0.5, drain_shift 0.5, drain_sub 0.25, drain_gap 0.6.
  - The doc string gives the formula.
  - Example: data/exp_launch/attr_model_shoot_v5.json (from ds_canon_v5.pkl, λ 3e-5): g_dinoc 5.30, g_drain 3.92, prior 0.0220, hidden 37.9, b 0.634, gz 0.459, g_dist -0.479, g_opr 0.266.

### Runtime rule (attribution._model_weights, model version >= 2)

```
lw[i] = b * log(e + 1e-9) + gz * log(z + zone_eps) + g_dist * dz          # unchanged
      + g_dinoc * log1p(S_dinoc_i) + g_drain * log1p(S_drain_i)            # new
S_x_i = mean(ts.x_v[(ts.x_t >= t - LAG_HI) & (ts.x_t <= t - LAG_LO)])  (0.0 if no grid point or no series)   # form "window"
      = sum(kernel(t - ts.x_t) * ts.x_v)                                                                     # form "kernel"
```

- Checked with tools/runtime_parity.py: this rule with the exported constants reproduces fit_cv's per-jump weights to 3e-16 (cc_qm15, cc_qm24, joh_qm39).
- Today's runtime has no version check. Don't deploy a version-2 JSON before the factors are in, because all its other constants were fitted jointly with them.

### Parity

| check | result |
| --- | --- |
| prod_signals wrappers on md (scan.pkl + det_seg/team_of_seg) vs build_ds series, cc_qm15 / cagle_qm23 / joh_qm39 | dinoc and drain series max diff 0 (tools/wrapper_parity.py) |
| canonical build_ds features vs the CV'd grid features | max diff 0 (all 4 features); all older features equal ds_v5.pkl |
| grid features vs the event-based features of the first CV | corr 0.992 (dinoc) / 0.975 (drain), sum ratio 0.999 / 0.997: grid phase only |

### CV on the canonical dataset

Your 12:06 build_ds, no-hindsight OPR prior; λ 3e-5; 105 alliances; paired CI over 54 matches:

| spec | TV | fuel MAE | ΔTV [95% CI] | Δcorr [95% CI] | ΔMAE [95% CI] | Δtop |
| --- | ---: | ---: | --- | --- | --- | ---: |
| lin/all+dist+mlprior | 0.1323 | 44.56 | | | | |
| + fx:dinoc + fx:drainraw (window) | 0.125 | 41.8 | -0.0076 [-0.0138, -0.0013] | +0.0178 [+0.0031, +0.0337] | -2.75 [-5.20, -0.36] | +5.7 |
| + fx:dinock + fx:drainrawk (kernel) | 0.125 | 42.3 | -0.0072 [-0.0148, +0.0007] | +0.0180 [-0.0011, +0.0377] | -2.23 [-5.23, +0.88] | +4.8 |

Other events (fitted on Chezy, vs split-half OPR, learned model; v5 prior):

| form | Δ share corr | Δ fuel MAE | Δ top |
| --- | --- | --- | --- |
| window | -0.010 [-0.023, +0.003] | +0.55 [-0.75, +1.85] | -2.1 pp |
| kernel | -0.003 [-0.019, +0.013] | +0.77 [-0.72, +2.30] | -2.9 pp |

- Both forms are within noise. The window form leans worse on share corr (P(worse) 0.94).
