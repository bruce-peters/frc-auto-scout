"""Per-robot shooting signals for the learned credit model (data/exp_launch/RESULTS.md), production form.

Both are series on the TeamSignals 0.5 s grid (pipeline.analyse: grid = np.arange(md_t0(md), md_t1(md), 0.5)),
already moved to launch time, so they can be stored like dist_t / dist_v and windowed per scoreboard jump:

  dinoc  identity-centred DINOv2 'is shooting': per robot crop, rate = ((x - mu) / sd) . w + b with x = the crop's
         box_inner + box_cls embedding (768-d) minus that robot's mean embedding over its crops IN THE SAME VIEW
         AND THE SAME MATCH ("centred": per robot, per view, within the current match only - nothing from other
         matches). value at grid point g = max(0, mean rate of the robot's crops whose time + 0.5 s is nearest
         to g) * 0.5 s  (expected balls in that 0.5 s cell); 0 where the robot has no crop.
  drain  raw hopper drain: per view, the robot's FUEL-yellow box fraction (Detection.fill) median-of-3, sampled
         on a 0.25 s sub-grid of the same origin (linear interpolation, only across gaps <= 0.6 s); a step's
         drain = max(0, f[k-1] - f[k]); views combined by the max per step; the steps (moved +0.5 s) summed into
         the nearest 0.5 s grid cell.

Per scoreboard jump at video time t, per robot (build_ds.py features; the runtime must use the same form):
  window form (s_dinoc, s_drainraw):   S = mean(value over the grid points in [t - LAG_HI, t - LAG_LO]), 0 if none
  kernel form (s_dinock, s_drainrawk): S = sum(attribution.kernel(t - grid) * value)
and the robot's log weight gets + g_dinoc * log1p(S_dinoc) + g_drain * log1p(S_drain) (model JSON version 2,
shooting.form says which form). Drain inputs are rounded like pipeline det_rows (t, fill to 3 decimals), so the
runtime series equal the dataset's exactly (tools/wrapper_parity.py).

If a match has no embeddings yet (before scripts/build_dino_probe.py embed), dinoc_series returns None: use
s_dinoc = 0 for every robot (factor 1).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

GRID_STEP = 0.5
DINO_SHIFT = 0.5   # s: crop time -> launch-aligned (its fitted lag minus the credited launches', 4 s windows)
DRAIN_SHIFT = 0.5  # s: the same for drain
DRAIN_SUB = 0.25   # s: drain sub-grid
DRAIN_GAP = 0.6    # s: largest gap the fill series is interpolated across
EMB_VIEWS = ["main", "red_end", "blue_end"]  # view codes in data/exp_dino/emb/<match>.npz
FEATS = ("box_inner", "box_cls")


def grid_of(md) -> np.ndarray:
    from autoscout import pipeline as P
    return np.arange(P.md_t0(md), P.md_t1(md), GRID_STEP)


def load_ridge(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- dinoc
def dinoc_core(crop_t: np.ndarray, crop_view: np.ndarray, crop_team: np.ndarray, X: np.ndarray, ridge: dict,
               grid: np.ndarray, exclude_key: str | None = None) -> dict[int, np.ndarray]:
    """crop arrays (team -1 = unknown) + embeddings X (n, 768) -> {team: values on grid}.
    exclude_key: use the ridge's leave-that-match-out weights if it has them (datasets for CV only)."""
    w = ridge["w"]
    if exclude_key is not None and "loo_keys" in ridge:
        hit = np.where(ridge["loo_keys"] == exclude_key)[0]
        if len(hit):
            w = ridge["W_loo"][hit[0]]
    X = X.astype(np.float32, copy=True)
    for tm in np.unique(crop_team[crop_team >= 0]):
        for v in np.unique(crop_view[crop_team == tm]):
            m = (crop_team == tm) & (crop_view == v)
            X[m] -= X[m].mean(0)
    rate = ((X - ridge["mu"]) / ridge["sd"]) @ w[:-1] + w[-1]
    idx = np.rint((crop_t + DINO_SHIFT - grid[0]) / GRID_STEP).astype(int)
    out = {}
    for tm in np.unique(crop_team[crop_team >= 0]):
        m = (crop_team == tm) & (idx >= 0) & (idx < len(grid))
        v = np.zeros(len(grid))
        if m.any():
            s = np.bincount(idx[m], weights=rate[m], minlength=len(grid))
            c = np.bincount(idx[m], minlength=len(grid))
            v = np.where(c > 0, np.maximum(s / np.maximum(c, 1), 0.0) * GRID_STEP, 0.0)
        out[int(tm)] = v
    return out


def dinoc_series(md, emb_npz, ridge_npz, det_seg: dict, team_of_seg: dict, grid: np.ndarray | None = None,
                 exclude_key: str | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """pipeline.analyse objects -> {team: (grid, value)}; None if the match has no embeddings."""
    if emb_npz is None or not Path(emb_npz).exists():
        return None
    E = np.load(emb_npz)
    by_crop = {d.crop_id: d for d in md.dets if getattr(d, "crop_id", None) is not None}
    team = np.array([-1 if (d := by_crop.get(int(c))) is None else (team_of_seg.get(det_seg.get(id(d))) or -1)
                     for c in E["ids"]], int)
    X = np.concatenate([E[f].astype(np.float32) for f in FEATS], 1)
    g = grid_of(md) if grid is None else grid
    vals = dinoc_core(E["t"].astype(float), E["view"].astype(int), team, X, load_ridge(ridge_npz), g, exclude_key)
    return {tm: (g, v) for tm, v in vals.items()}


# ----------------------------------------------------------------------------- drain
def drain_core(det_t, det_view, det_fill, det_team, grid: np.ndarray) -> dict[int, np.ndarray]:
    """per-detection arrays (fill None/nan = no mask that frame) -> {team: values on grid}."""
    sub0 = grid[0] - 4 * DRAIN_SUB
    per = defaultdict(lambda: defaultdict(list))
    for t, v, f, tm in zip(det_t, det_view, det_fill, det_team):
        if tm is not None and tm >= 0 and f is not None and np.isfinite(f):
            per[int(tm)][v].append((float(t), float(f)))
    out = {}
    for tm, views in per.items():
        best = defaultdict(float)  # sub-grid index -> max drain over views
        for arr in views.values():
            a = np.array(sorted(arr))
            if len(a) < 6:
                continue
            f = np.array([np.median(a[max(0, i - 1):i + 2, 1]) for i in range(len(a))])
            k0, k1 = int(np.ceil((a[0, 0] - sub0) / DRAIN_SUB)), int(np.floor((a[-1, 0] - sub0) / DRAIN_SUB))
            if k1 - k0 < 1:
                continue
            ks = np.arange(k0, k1 + 1)
            g = sub0 + ks * DRAIN_SUB
            fi = np.interp(g, a[:, 0], f)
            j = np.clip(np.searchsorted(a[:, 0], g), 1, len(a) - 1)
            ok = (a[j, 0] - a[j - 1, 0]) <= DRAIN_GAP
            d = np.maximum(0.0, fi[:-1] - fi[1:]) * (ok[:-1] & ok[1:])
            for k, x in zip(ks[1:], d):
                if x > best[k]:
                    best[k] = x
        v = np.zeros(len(grid))
        for k, x in best.items():
            gi = int(np.rint((sub0 + k * DRAIN_SUB + DRAIN_SHIFT - grid[0]) / GRID_STEP))
            if 0 <= gi < len(grid):
                v[gi] += x
        out[tm] = v
    return out


def drain_series(md, det_seg: dict, team_of_seg: dict, grid: np.ndarray | None = None
                 ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    g = grid_of(md) if grid is None else grid
    tm = [team_of_seg.get(det_seg.get(id(d))) for d in md.dets]
    # rounded like pipeline det_rows (dets.json), so runtime equals the dataset build_ds.py fitted on
    vals = drain_core([round(d.t, 3) for d in md.dets], [d.view for d in md.dets],
                      [np.nan if d.fill is None else round(d.fill, 3) for d in md.dets],
                      [-1 if x is None else x for x in tm], g)
    return {t: (g, v) for t, v in vals.items()}


# ----------------------------------------------------------------------------- datasets (no md in memory)
def crop_rows(key: str, emb: dict, rows: list, cache_dir: Path, scan_path: Path | None) -> np.ndarray | None:
    """For build_ds.py: embedding crop -> dets.json row index (-1 = none), validated by each crop's time and view.
    Sources in order: a crop_id column 10 in dets.json (if the pipeline writes one), the cache
    <cache_dir>/<key>.npy, the scan's md.dets[i].crop_id (then cached). None if nothing validates."""
    t_e, v_e = np.asarray(emb["t"], float), np.asarray(emb["view"], int)

    def ok(di):
        m = di >= 0
        if m.sum() < 0.5 * len(di) or not len(di):
            return False
        rt = np.array([rows[i][0] for i in di[m]], float)
        rv = np.array([rows[i][1] for i in di[m]])
        good = (np.abs(rt - t_e[m]) <= 2e-3) & (rv == np.array(EMB_VIEWS)[v_e[m]])
        return good.mean() >= 0.99

    ids = np.asarray(emb["ids"], int)
    if rows and len(rows[0]) > 10:
        pos = {r[10]: i for i, r in enumerate(rows) if r[10] is not None}
        di = np.array([pos.get(int(c), -1) for c in ids])
        if ok(di):
            return di
    p = Path(cache_dir) / f"{key}.npy"
    if p.exists():
        di = np.load(p)
        if len(di) == len(ids) and ok(di):
            return di
    if scan_path is not None and Path(scan_path).exists():
        import gc
        import pickle
        with open(scan_path, "rb") as fh:
            md = pickle.load(fh)
        if len(md.dets) == len(rows):
            pos = {d.crop_id: i for i, d in enumerate(md.dets) if getattr(d, "crop_id", None) is not None}
            di = np.array([pos.get(int(c), -1) for c in ids])
            del md
            gc.collect()
            if ok(di):
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                np.save(p, di.astype(np.int32))
                return di
    return None


# ----------------------------------------------------------------------------- per jump
def jump_feature(grid: np.ndarray, value: np.ndarray, t: float, form: str = "kernel") -> float:
    from autoscout import attribution as A
    if grid is None or not len(grid):
        return 0.0
    if form == "kernel":
        return float((A.kernel(t - grid) * value).sum())
    m = (grid >= t - A.LAG_HI) & (grid <= t - A.LAG_LO)
    return float(value[m].mean()) if m.any() else 0.0
