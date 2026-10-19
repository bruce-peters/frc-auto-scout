"""Hub entries: when scored balls dropped into a hub, and which launch arcs they came from.

The scoreboard counter is exact in count but late: a ball is counted ~1.3 s after it drops into the hub opening
when the hub is idle, and up to ~3 s later deep in a hopper dump, because the hub counts at most ~15 balls/s
(a queue: the k-th ball counted is the k-th ball that entered). This module

1. finds entry evidence per hub and camera view (`track_ends`, `is_entry`, `entry_series`): flight tracks whose
   last point drops into the projected hub opening, plus moving-FUEL blobs over the opening (`MatchData.hub_feats`);
2. re-times every counted ball to its most likely entry time (`retime_queue`; `retime` is the simpler variant
   without the queue structure), keeping the counter's ball count: per counter burst, cumulative counted balls
   are matched to cumulative entry evidence (first in, first out) inside delay bounds;
3. splits each re-timed ball among the alliance's robots by their launches one flight time earlier, with a narrow
   kernel (`entry_kernel`), optionally mixed with attribution's counter-time kernel (`mix`), as a drop-in for
   attribution.split_increments (`split_entries`; pieces with conf 0 reproduce split_increments exactly).

Everything is per match (no event-level fitting); the constants were set once on past matches. Status
(data/exp_hubentry/RESULTS_section.md): the re-timing halves the timing spread against an independent camera
view, but the credit split built on it is not adopted (gains within noise; per-jump agreement and the 254 check
get worse) while launch recall is ~0.2 detected launches per counted ball.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import field as fieldmod

# ---------------------------------------------------------------- geometry of an entry
# opening ellipse in image units of its horizontal radius r: an entering ball's last tracked point
OPEN_DU = 1.0           # |u - cu| / r
OPEN_DV = (-0.6, 1.2)   # (v - cv) / r: from just above the rim to just below the far lip (it drops out of view)
MIN_PTS = 3
TRACK_COLS = ("t0", "t1", "u0", "v0", "u1", "v1", "n", "vu1", "vv1", "vu0", "vv0",
              "red_du", "red_dv", "red_r", "blue_du", "blue_dv", "blue_r")
C = {c: i for i, c in enumerate(TRACK_COLS)}


def _vel(p: np.ndarray, end: bool, n: int = 4) -> tuple[float, float]:
    q = p[-n:] if end else p[:n]
    if len(q) < 2:
        return 0.0, 0.0
    dt = q[-1, 0] - q[0, 0]
    if dt <= 0:
        return 0.0, 0.0
    return float((q[-1, 1] - q[0, 1]) / dt), float((q[-1, 2] - q[0, 2]) / dt)


def opening_frames(cam_track, alliance: str) -> tuple[np.ndarray, np.ndarray]:
    """Per pose time of a CameraTrack: (times, [(cu, cv, r)]) of the hub opening in that view (nan if behind)."""
    ring = fieldmod.hub_opening_3d(alliance, 12)
    times = np.asarray(getattr(cam_track, "times", [0.0]), dtype=float)
    out = np.full((len(times), 3), np.nan)
    for i, t in enumerate(times):
        c = cam_track.at(float(t))
        pc = (c.R @ (ring - c.center_world()).T).T
        if (pc[:, 2] <= 0.1).any():
            continue
        pr = c.project(ring)
        out[i] = (pr[:, 0].mean(), pr[:, 1].mean(), max((pr[:, 0].max() - pr[:, 0].min()) / 2, 3.0))
    return times, out


def _frame_at(times: np.ndarray, fr: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(times) == 1:
        return np.repeat(fr, len(t), 0)
    return np.stack([np.interp(t, times, fr[:, k]) for k in range(3)], 1)


def track_ends(tracks: list, cam_track) -> np.ndarray:
    """Table (TRACK_COLS) of one view's flight tracks: start / end, end velocity, end position relative to each
    hub opening (du, dv in units of the opening's image radius r)."""
    rows = []
    for tr in tracks:
        p = np.asarray(tr.pts, dtype=float)
        if len(p) < MIN_PTS:
            continue
        vu1, vv1 = _vel(p, True)
        vu0, vv0 = _vel(p, False)
        rows.append((p[0, 0], p[-1, 0], p[0, 1], p[0, 2], p[-1, 1], p[-1, 2], len(p), vu1, vv1, vu0, vv0))
    T = np.full((len(rows), len(TRACK_COLS)), np.nan)
    if not rows:
        return T
    T[:, :11] = np.array(rows)
    if cam_track is None:
        return T
    for k, al in enumerate(("red", "blue")):
        times, fr = opening_frames(cam_track, al)
        ok = np.isfinite(fr[:, 0])
        if not ok.any():
            continue
        f = _frame_at(times[ok], fr[ok], T[:, C["t1"]])
        T[:, 11 + 3 * k] = (T[:, C["u1"]] - f[:, 0]) / f[:, 2]
        T[:, 12 + 3 * k] = (T[:, C["v1"]] - f[:, 1]) / f[:, 2]
        T[:, 13 + 3 * k] = f[:, 2]
    return T


def is_entry(T: np.ndarray, alliance: str, du: float = OPEN_DU, dv: tuple = OPEN_DV) -> np.ndarray:
    """Tracks whose last point drops into the alliance's hub opening (descending at the end)."""
    if not len(T):
        return np.zeros(0, bool)
    a, b = T[:, C[f"{alliance}_du"]], T[:, C[f"{alliance}_dv"]]
    down = T[:, C["vv1"]] > 0
    with np.errstate(invalid="ignore"):
        return (np.abs(a) <= du) & (b >= dv[0]) & (b <= dv[1]) & down


# ---------------------------------------------------------------- entry evidence -> per-ball entry times
D_MIN, D_MAX = 1.0, 7.0    # s: entry -> counter delay bounds (sensor + display at least ~1.3 s; queues add more)
D_TYP = 1.7                # s: the typical (median) delay at a non-busy hub (lag_study: 1.75 Chezy, 1.66 FIRST)
BURST_GAP = 2.0            # s: counter jumps further apart than this start a new burst
BG_WIN = 6.0               # s: background (idle) activity level taken over this long before the burst
READ_GAP = 1 / 3.0         # s: the overlay is read 3x/s; a jump happened in the 1/3 s before its reading


@dataclass
class Retimed:
    count_t: float   # video time of the counter reading that counted the ball(s)
    entry_t: float   # estimated video time the ball(s) dropped into the hub
    n: float         # balls (fractional when a reading's balls are spread over several entry times)
    conf: float      # 0..1: how much entry evidence backed this assignment (0 = fell back to count_t - D_TYP)


def _bursts(inc: list[tuple[float, float]], gap: float = BURST_GAP) -> list[list[tuple[float, float]]]:
    out, cur = [], []
    for t, n in inc:
        if n <= 0:
            continue
        if cur and t - cur[-1][0] > gap:
            out.append(cur)
            cur = []
        cur.append((t, n))
    if cur:
        out.append(cur)
    return out


def retime(inc: list[tuple[float, float]], rate_t: np.ndarray, rate: np.ndarray, d_min: float = D_MIN,
           d_max: float = D_MAX, d_typ: float = D_TYP, split: int = 4, min_mass: float = 0.5) -> list[Retimed]:
    """Counter increments [(video_t, n)] -> per-ball entry times, first in first out.

    rate_t / rate: entry evidence on a regular grid (any scale). Per burst of counter jumps, the evidence in
    [first jump - d_max, last jump - d_min] (minus its idle background) is normalised to the burst's ball count,
    and the k-th counted ball is placed where the cumulative evidence reaches k (each reading's balls are cut
    into `split` pieces so a big jump can cover a spread of entries). Each placement is then clipped so its
    delay stays in [d_min, d_max]. Bursts with too little evidence fall back to count time - d_typ.
    """
    out: list[Retimed] = []
    if not len(rate_t):
        return [Retimed(t, t - d_typ, n, 0.0) for t, n in inc if n > 0]
    dt = float(np.median(np.diff(rate_t))) if len(rate_t) > 1 else 0.1
    for b in _bursts(inc):
        ta, tb = b[0][0] - READ_GAP, b[-1][0]
        nb = sum(n for _, n in b)
        m = (rate_t >= ta - d_max) & (rate_t <= tb - d_min)
        bgm = (rate_t >= ta - d_max - BG_WIN) & (rate_t < ta - d_max)
        bg = float(np.median(rate[bgm])) if bgm.any() else 0.0
        ev = np.clip(rate[m] - bg, 0, None)
        if m.sum() < 2 or ev.sum() * dt < min_mass * 1e-3 or ev.sum() <= 0:
            out += [Retimed(t, t - d_typ, n, 0.0) for t, n in b]
            continue
        et = rate_t[m]
        cum = np.concatenate([[0.0], np.cumsum(ev)])
        cum = cum / cum[-1] * nb
        edges = np.concatenate([et - dt / 2, [et[-1] + dt / 2]])
        k = 0.0
        # evidence share in the burst window: how peaked vs flat (flat -> low confidence)
        conf = float(min(1.0, ev.sum() / max(1e-9, rate[m].sum())))
        for t, n in b:
            piece = n / split
            for s in range(split):
                mid = k + piece / 2
                te = float(np.interp(mid, cum, edges))
                te = min(max(te, t - READ_GAP - d_max), t - d_min)
                out.append(Retimed(t, te, piece, conf))
                k += piece
    return out


# ---------------------------------------------------------------- credit
FLIGHT = 0.8        # s: launch -> entry, typical (lag_study: launch -> hub-top activity 0.73-0.95 s)
FLIGHT_LO, FLIGHT_HI = 0.2, 1.9   # s: kernel support of launch -> entry
KERNEL_AREA = 2.35  # s: area of attribution.kernel (0.8 / 2.8 / 5.5 triangle): narrow kernel rescaled to it


def entry_kernel(dt: np.ndarray, lo: float = FLIGHT_LO, peak: float = FLIGHT, hi: float = FLIGHT_HI,
                 area: float | None = KERNEL_AREA) -> np.ndarray:
    """Triangular weight over entry - launch delay dt (s); scaled to `area` so launch evidence keeps the scale
    the credit model was fitted on (None: peak 1)."""
    w = np.where(dt < peak, (dt - lo) / (peak - lo), (hi - dt) / (hi - peak))
    w = np.clip(w, 0, 1)
    if area:
        w = w * area / (0.5 * (hi - lo))
    return w


# ---------------------------------------------------------------- entry evidence series
EV_BIN = 0.1
EV_SMOOTH = 3             # bins: box smoothing of the sparse entry-track events
BLOB_SHIFT = 0.2          # s: blobs over the opening lead the ball's disappearance into it by about this much


def _hist(times, edges, w=None) -> np.ndarray:
    h, _ = np.histogram(times, edges, weights=w)
    return h.astype(float)


def entry_series(tables: dict, hub_feats: dict, alliance: str, t0: float, t1: float, w_track: float = 1.0,
                 w_blob: float = 1.0, bin_s: float = EV_BIN) -> tuple[np.ndarray, np.ndarray]:
    """Entry evidence for one hub on a regular grid: entry tracks (is_entry) in the main and the alliance's own end
    view, plus moving-FUEL blobs over the opening (hub_feats "top") in the same views. Each source is scaled to
    mean 1 over the match, so the result is a relative rate (retime() only needs its shape inside a burst).

    tables: {view: track_ends table}; hub_feats: {(view, alliance, "top"|"exit"): [(t, n, area)]}.
    """
    edges = np.arange(t0, t1 + bin_s, bin_s)
    tc = edges[:-1] + bin_s / 2
    rate = np.zeros(len(tc))
    k = np.ones(EV_SMOOTH) / EV_SMOOTH
    for v in ("main", f"{alliance}_end"):
        T = tables.get(v)
        if w_track and T is not None and len(T):
            e = is_entry(T, alliance)
            h = np.convolve(_hist(T[e, C["t1"]], edges), k, mode="same")
            if h.sum() > 0:
                rate += w_track * h / h.mean()
        arr = hub_feats.get((v, alliance, "top"))
        if w_blob and arr is not None and len(arr):
            arr = np.asarray(arr, dtype=float).reshape(-1, 3)
            n = _hist(arr[:, 0] + BLOB_SHIFT, edges, arr[:, 1]) / np.maximum(_hist(arr[:, 0] + BLOB_SHIFT, edges), 1)
            if n.sum() > 0:
                rate += w_blob * n / n.mean()
    return tc, rate


def retime_queue(inc: list[tuple[float, float]], rate_t: np.ndarray, rate: np.ndarray, d_min: float = 0.6,
                 d_max: float = D_MAX, d_first: tuple = (0.8, 2.6), d_typ: float = 1.35, gap: float = 1.0,
                 split: int = 4, min_ev: float = 0.3) -> list[Retimed]:
    """Like retime(), with the hub's queue structure: the counter pauses only when the hub's queue is empty, so a
    burst (counts no more than `gap` s apart) starts with a ball that entered d_first s earlier (no backlog), and
    no ball of a burst entered before the previous burst's last ball (first in, first out). Bursts whose window
    holds less than `min_ev` (mean-1 units x s) of evidence above background fall back to count time - d_typ.
    """
    out: list[Retimed] = []
    dt = float(np.median(np.diff(rate_t))) if len(rate_t) > 1 else EV_BIN
    floor = -1e9
    for b in _bursts(inc, gap):
        ta, tb = b[0][0] - READ_GAP, b[-1][0]
        nb = sum(n for _, n in b)
        lo = max(ta - d_first[1], floor)
        hi = tb - d_min
        m = (rate_t >= lo) & (rate_t <= hi)
        bgm = (rate_t >= lo - BG_WIN) & (rate_t < lo)
        bg = float(np.median(rate[bgm])) if bgm.any() else 0.0
        ev = np.clip(rate[m] - bg, 0, None) if m.any() else np.zeros(0)
        if m.sum() < 2 or ev.sum() * dt < min_ev:
            for t, n in b:
                te = t - READ_GAP / 2 - d_typ
                out.append(Retimed(t, te, n, 0.0))
            floor = max(floor, b[-1][0] - READ_GAP / 2 - d_typ)
            continue
        et = rate_t[m]
        cum = np.concatenate([[0.0], np.cumsum(ev)])
        cum = cum / cum[-1] * nb
        edges = np.concatenate([et - dt / 2, [et[-1] + dt / 2]])
        conf = float(min(1.0, ev.sum() * dt / max(nb * 0.1, 1e-9)))  # evidence per ball, saturating
        k = 0.0
        last = floor
        for t, n in b:
            piece = n / split
            for s in range(split):
                te = float(np.interp(k + piece / 2, cum, edges))
                te = min(max(te, t - READ_GAP - d_max, ta - d_first[1]), t - d_min)
                out.append(Retimed(t, te, piece, conf))
                last = max(last, te)
                k += piece
        floor = last
    return out


# ---------------------------------------------------------------- credit split on re-timed balls
def _weights(model: dict | None, teams: list, te: float, gate: bool, kern, lo: float, hi: float) -> np.ndarray:
    """Per-robot credit weights for ball(s) that entered the hub at te: attribution's learned model (or the hand
    rule when model is None) with launch evidence s_r = sum kern(te - launch) * conf over the robot's launches and
    the zone / visibility / distance windows taken over [lo, hi] (when its launches could have left)."""
    from . import attribution as A

    n = len(teams)
    if model is not None:
        lw = np.zeros(n)
        for i, ts in enumerate(teams):
            z = ts.zone_frac(lo, hi) if gate else 1.0
            s = float((kern(te - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
            vis = ts.vis_frac(lo, hi)
            pres = 1.0 - ts.absent_frac(lo, hi) if ts.absent else 1.0
            prior = 0.0 if ts.prior_off else model["prior"]
            hid = 0.0 if ts.no_hidden else model["hidden"] * (1.0 - vis)
            e = s + (prior * vis + hid) * pres
            d = ts.dist_mean(lo, hi)
            dz = 0.0 if np.isnan(d) else (d - model["dist_center"]) / model["dist_scale"]
            lw[i] = model["b"] * np.log(e + 1e-9) + model["gz"] * np.log(z + model["zone_eps"]) + model["g_dist"] * dz
        w = np.exp(lw - lw.max())
        return w / w.sum()
    w = np.zeros(n)
    for i, ts in enumerate(teams):
        z = ts.zone_frac(lo, hi) if gate else 1.0
        s = float((kern(te - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
        pres = 1.0 - ts.absent_frac(lo, hi) if ts.absent else 1.0
        prior = 0.0 if ts.prior_off else A.PRIOR
        vis = ts.vis_frac(lo, hi) if A.VIS_AWARE else 1.0
        hid = 0.0 if (ts.no_hidden or not A.VIS_AWARE) else A.HIDDEN * (1.0 - vis)
        w[i] = z * (s + (prior * vis + hid) * pres)
    if w.sum() <= 1e-9:
        w = np.ones(n)
    return w / w.sum()


def split_entries(increments: list[tuple[float, float]], teams: list, to_match_time, entries: list[Retimed],
                  log: list | None = None, opr: dict | None = None, model: dict | None = "default",
                  kern=None, flight: tuple = (FLIGHT_LO, FLIGHT_HI), fallback_conf: float = 0.0,
                  mix: float = 0.0) -> dict:
    """attribution.split_increments on re-timed balls: same output ({team: {period: balls}}), same log stamps
    (counter time - 2 s) and the same period / zone-gate rules, but each counter reading's balls are split over
    their entry times (entries from retime / retime_queue) and credited to launches one flight time earlier.
    Pieces with confidence <= fallback_conf use attribution's broad counter-time kernel instead. With mix > 0 the
    launch evidence is (1 - mix) * entry kernel + mix * attribution's kernel at counter time (windows: their union)."""
    from . import attribution as A

    if model == "default":
        model = A.load_model()
    kern = kern or entry_kernel
    by_t: dict[float, list[Retimed]] = {}
    for e in entries:
        by_t.setdefault(e.count_t, []).append(e)
    out = {ts.team: {} for ts in teams}
    logged = []
    for t, n in increments:
        if n <= 0:
            continue
        mt = to_match_time(t - 2.0)
        per = A.period_of(mt) if mt is not None else "unknown"
        gate = A.AUTO_ZONE_GATE or per != "auto"
        alloc = np.zeros(len(teams))
        pieces = by_t.get(t) or [Retimed(t, t, n, 0.0)]
        tot = sum(p.n for p in pieces)
        for p in pieces:
            if p.conf <= fallback_conf:
                w = _weights(model, teams, t, gate, A.kernel, t - A.LAG_HI, t - A.LAG_LO)
            elif mix > 0:
                off = t - p.entry_t

                def kmix(dt, off=off):
                    return (1 - mix) * kern(dt) + mix * A.kernel(dt + off)
                w = _weights(model, teams, p.entry_t, gate, kmix, min(p.entry_t - flight[1], t - A.LAG_HI),
                             max(p.entry_t - flight[0], t - A.LAG_LO))
            else:
                w = _weights(model, teams, p.entry_t, gate, kern, p.entry_t - flight[1], p.entry_t - flight[0])
            alloc += (n * p.n / tot) * w
        for i, ts in enumerate(teams):
            out[ts.team][per] = out[ts.team].get(per, 0.0) + alloc[i]
            if mt is not None:
                logged.append((mt, ts.team, float(alloc[i])))
    f = A._opr_factors(model, out, opr) if model is not None else None
    if f:
        out = {tm: {p: v * f[tm] for p, v in d.items()} for tm, d in out.items()}
        logged = [(mt, tm, b * f[tm]) for mt, tm, b in logged]
    if log is not None:
        log.extend((round(mt, 2), tm, round(b, 3)) for mt, tm, b in logged)
    return out
