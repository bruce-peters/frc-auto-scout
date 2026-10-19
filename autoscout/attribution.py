"""Split each alliance's scored FUEL among its three robots.

The alliance-level scoring timeline comes from (in order of preference) the live overlay
counter, or TBA's per-period hub counts, or the vision hub-entry counter. Each scoring
increment at video time t was launched roughly 1-4 s earlier (hub sensor lag ~1.3 s, up to
3 s, plus flight time). It is shared among the alliance's robots in proportion to

    w_r = zone_r * (shots_r + prior)

where shots_r is the (time-kernel weighted) number of launches detected from robot r in that
window, zone_r is the fraction of the window robot r spent in its alliance zone (a robot may
only shoot into its hub from there, G407), and `prior` keeps a robot that is in the zone but
whose shots are hidden from every camera from getting exactly zero.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import AUTO_LEN, ROOT, TELEOP_PERIODS

# s from launch to counter increment: ~1.1 s flight, then the hub counts a ball ~1.3 s after it drops into an idle
# hub but at most ~15 balls/s, so in a hopper dump balls queue inside the hub (Chezy: 1.25 s for a burst's first 5
# balls, 2.1 s median for balls 60+, 90th percentile 4.3 s). The measured launch-to-count peak is 2.4 s; the window
# is wide for the queue (data/exp_hubentry/RESULTS_section.md; autoscout/hubentry.py re-times counted balls to
# their hub entries)
LAG_LO, LAG_HI, LAG_PEAK = 0.8, 5.5, 2.8
PRIOR = 0.12
GRACE = 3.0
# Occlusion: a robot no camera can see cannot show launch arcs, so "no arcs seen" is weak
# evidence for it. It gets HIDDEN pseudo-shots per window, scaled by how much of the window it
# was hidden; a robot that is in plain view and launching nothing keeps only PRIOR.
VIS_AWARE = True
HIDDEN = 2.0  # tuned on 2026casnv+joh (share corr vs fuel OPR 0.648 -> 0.686), test events 0.636 -> 0.671
# Zone gating in AUTO: robots tracked in the neutral zone do score in auto (or auto positions are off).
# Gating auto credit by alliance-zone occupancy made the auto split worse than an even split (share corr
# vs an auto-only OPR 0.205, MAE 0.183 vs 0.144 for 1/3 each). Without it, on all 76 v2 matches:
# total share corr vs fuel OPR 0.682 -> 0.718 (paired bootstrap 95 % CI +0.019..+0.057), top scorer
# 83 -> 91 of 150, corr vs 5026's scouting 0.856 -> 0.870 (failure-finder cycle 7). Teleop stays gated
# (ungating it hurts: 0.598). Empirical choice: whether the zone rule binds in AUTO was not verified.
AUTO_ZONE_GATE = False
# Learned credit model (opt-in; data/exp_attrmodel/RESULTS_section.md): a JSON of weights fitted on FRC 5026's
# human scouting of the Chezy Champs quals (leave-one-match-out CV). Per scoreboard jump a robot's weight is
#   (zone + eps)^gz * (s + prior * vis + hidden * (1 - vis))^b * exp(g_dist * (dist_to_own_hub - 3.5 m) / 2 m)
# and each team's match total is then multiplied by its fuel-OPR share^g_opr (alliance total kept exact).
# Default: models/attr_model.json when present (adopted 2026-10-18: vs scouting, leave-one-match-out, share
# error 0.153 -> 0.134 and fuel MAE 52.1 -> 44.5; other events' share corr vs OPR 0.635 -> 0.696). Results keep
# the pure-video split as `fuel_video`. The environment variable AUTOSCOUT_ATTR_MODEL overrides the path;
# set it to "none" for the hand-tuned rule above.
_env_model = os.environ.get("AUTOSCOUT_ATTR_MODEL")
ATTR_MODEL: str | None = (None if (_env_model or "").lower() == "none" else _env_model) or (
    "models/attr_model.json" if (Path(__file__).resolve().parent.parent / "models" / "attr_model.json").exists()
    and (_env_model or "").lower() != "none" else None)
_MODELS: dict[str, dict] = {}


def load_model(path: str | None = None) -> dict | None:
    """The learned credit model ATTR_MODEL points at (cached), or None for the hand-tuned rule."""
    path = path if path is not None else ATTR_MODEL
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    key = str(p)
    if key not in _MODELS:
        _MODELS[key] = json.loads(p.read_text(encoding="utf-8"))
    return _MODELS[key]


def period_of(match_t: float) -> str:
    """Scoring period for a ball that *entered the hub* at match time `match_t`."""
    if match_t < AUTO_LEN + GRACE:
        return "auto"
    tt = match_t - AUTO_LEN
    for name, a, b in TELEOP_PERIODS:
        if a <= tt < b:
            return name
    return "endgame"


def kernel(dt: np.ndarray) -> np.ndarray:
    """Triangular weight over launch->count delay dt (s)."""
    w = np.where(dt < LAG_PEAK, (dt - LAG_LO) / (LAG_PEAK - LAG_LO), (LAG_HI - dt) / (LAG_HI - LAG_PEAK))
    return np.clip(w, 0, 1)


@dataclass
class TeamSignals:
    team: int
    shots: np.ndarray = field(default_factory=lambda: np.zeros(0))  # video times of launches
    shot_w: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per-shot confidence
    zone_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # sample times
    zone_v: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 1 in zone, 0 out, nan unknown
    vis_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # sample times
    vis_v: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 1 seen by some camera, 0 hidden
    # Absent / dead robots (data/exp_identity/RESULTS_section.md; all empty / False unless enabled in Options):
    absent: list = field(default_factory=list)  # [(t0, t1)] stretches the identity model says it is absent
    prior_off: bool = False  # parked all match: no PRIOR credit
    no_hidden: bool = False  # dead / absent all match: no HIDDEN credit
    dist_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # sample times (learned model only)
    dist_v: np.ndarray = field(default_factory=lambda: np.zeros(0))  # distance to own hub (m), nan unknown
    # shooting signals on the same grid (learned model version 2 only; autoscout/shooting.py): DINOv2 "is shooting"
    # (expected balls per 0.5 s cell) and hopper drain, both moved to launch time; empty = factor 1
    dinoc_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dinoc_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drain_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drain_v: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def dist_mean(self, t0: float, t1: float) -> float:
        """Mean distance to its own hub over [t0, t1] where the position is known (nan if never)."""
        if len(self.dist_t) == 0:
            return float("nan")
        m = (self.dist_t >= t0) & (self.dist_t <= t1)
        v = self.dist_v[m]
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")

    def vis_frac(self, t0: float, t1: float) -> float:
        """Fraction of [t0, t1] the robot was seen by at least one camera (1.0 if unknown)."""
        if len(self.vis_t) == 0:
            return 1.0
        m = (self.vis_t >= t0) & (self.vis_t <= t1)
        return float(self.vis_v[m].mean()) if m.any() else 1.0

    def absent_frac(self, t0: float, t1: float) -> float:
        """Fraction of [t0, t1] covered by the robot's absent stretches."""
        tot = 0.0
        for a, b in self.absent:
            tot += max(0.0, min(t1, b) - max(t0, a))
        return min(1.0, tot / max(t1 - t0, 1e-9))

    def zone_frac(self, t0: float, t1: float) -> float:
        if len(self.zone_t) == 0:
            return 0.5
        m = (self.zone_t >= t0) & (self.zone_t <= t1)
        v = self.zone_v[m]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            # nearest known sample within 3 s
            known = ~np.isnan(self.zone_v)
            if not known.any():
                return 0.5
            idx = np.argmin(np.abs(self.zone_t[known] - (t0 + t1) / 2))
            if abs(self.zone_t[known][idx] - (t0 + t1) / 2) < 3.0:
                return float(self.zone_v[known][idx]) * 0.8 + 0.1
            return 0.5
        return float(v.mean())


def _model_weights(model: dict, teams: list[TeamSignals], t: float, gate: bool) -> np.ndarray:
    """Learned credit weights (ATTR_MODEL) of an alliance's robots for a scoreboard jump at video time t."""
    lo, hi = t - LAG_HI, t - LAG_LO
    lw = np.zeros(len(teams))
    for i, ts in enumerate(teams):
        z = ts.zone_frac(lo, hi) if gate else 1.0
        s = float((kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
        vis = ts.vis_frac(lo, hi)
        pres = 1.0 - ts.absent_frac(lo, hi) if ts.absent else 1.0
        prior = 0.0 if ts.prior_off else model["prior"]
        hid = 0.0 if ts.no_hidden else model["hidden"] * (1.0 - vis)
        e = s + (prior * vis + hid) * pres
        d = ts.dist_mean(lo, hi)
        dz = 0.0 if np.isnan(d) else (d - model["dist_center"]) / model["dist_scale"]
        lw[i] = model["b"] * np.log(e + 1e-9) + model["gz"] * np.log(z + model["zone_eps"]) + model["g_dist"] * dz
        if model.get("g_dinoc") or model.get("g_drain"):  # version 2: + g * log1p(S) per shooting signal
            from .shooting import window_mean
            form = (model.get("shooting") or {}).get("form", "window")
            lw[i] += model.get("g_dinoc", 0.0) * np.log1p(window_mean(ts.dinoc_t, ts.dinoc_v, t, form))
            lw[i] += model.get("g_drain", 0.0) * np.log1p(window_mean(ts.drain_t, ts.drain_v, t, form))
    w = np.exp(lw - lw.max())
    return w / w.sum()


def _opr_factors(model: dict, out: dict[int, dict[str, float]], opr: dict | None) -> dict[int, float] | None:
    """Per-team factors applying the learned model's match-level fuel-OPR prior (alliance total unchanged)."""
    g = model.get("g_opr", 0.0)
    teams = list(out)
    tot = np.array([sum(out[tm].values()) for tm in teams])
    if not g or not opr or tot.sum() <= 0:
        return None
    o = np.array([opr.get(tm, np.nan) for tm in teams], float)
    if not (np.all(np.isfinite(o)) and o.sum() > 0):
        return None
    sh = tot / tot.sum()
    lw = np.log(sh + 1e-9) + g * np.log(np.maximum(o / o.sum(), model.get("opr_floor", 0.02)))
    new = np.exp(lw - lw.max())
    new = new / new.sum()
    return {tm: float(new[i] / sh[i]) if sh[i] > 0 else 1.0 for i, tm in enumerate(teams)}


def split_increments(increments: list[tuple[float, float]], teams: list[TeamSignals],
                     to_match_time, log: list | None = None,
                     opr: dict[int, float] | None = None) -> dict[int, dict[str, float]]:
    """increments: [(video_t, n_balls)] -> {team: {period: balls}}.

    If `log` is given, (match_time, team, balls) allocations are appended to it. A robot inside one of its
    absent stretches gets no PRIOR / HIDDEN credit (its launches still count). With the learned model
    (ATTR_MODEL) the weights come from _model_weights, and `opr` ({team: fuel OPR}, optional) feeds its
    match-level prior; `opr` is ignored by the hand-tuned rule.
    """
    model = load_model()
    out = {ts.team: {} for ts in teams}
    entries = []
    for t, n in increments:
        if n <= 0:
            continue
        mt = to_match_time(t - 2.0)
        per = period_of(mt) if mt is not None else "unknown"
        gate = AUTO_ZONE_GATE or per != "auto"
        if model is not None:
            w = _model_weights(model, teams, t, gate)
        else:
            w = np.zeros(len(teams))
            for i, ts in enumerate(teams):
                z = ts.zone_frac(t - LAG_HI, t - LAG_LO) if gate else 1.0
                s = float((kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                if ts.absent or ts.prior_off or ts.no_hidden:
                    pres = 1.0 - ts.absent_frac(t - LAG_HI, t - LAG_LO) if ts.absent else 1.0
                    prior = 0.0 if ts.prior_off else PRIOR
                    if VIS_AWARE:
                        vis = ts.vis_frac(t - LAG_HI, t - LAG_LO)
                        hid = 0.0 if ts.no_hidden else HIDDEN * (1.0 - vis)
                        w[i] = z * (s + (prior * vis + hid) * pres)
                    else:
                        w[i] = z * (s + prior * pres)
                elif VIS_AWARE:
                    vis = ts.vis_frac(t - LAG_HI, t - LAG_LO)
                    w[i] = z * (s + PRIOR * vis + HIDDEN * (1.0 - vis))
                else:
                    w[i] = z * (s + PRIOR)
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
        for i, ts in enumerate(teams):
            out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
            if mt is not None:
                entries.append((mt, ts.team, float(n * w[i])))
    f = _opr_factors(model, out, opr) if model is not None else None
    if f:
        out = {tm: {p: v * f[tm] for p, v in d.items()} for tm, d in out.items()}
        entries = [(mt, tm, b * f[tm]) for mt, tm, b in entries]
    if log is not None:
        log.extend((round(mt, 2), tm, round(b, 3)) for mt, tm, b in entries)
    return out


def increments_from_counts(counts: list[tuple[float, int]]) -> list[tuple[float, float]]:
    out = []
    prev = 0
    for t, c in counts:
        if c > prev:
            out.append((t, float(c - prev)))
        prev = max(prev, c)
    return out


def increments_from_periods(period_counts: dict[str, int], to_video_time, step: float = 0.5,
                            rate=None, to_match_time=None) -> list[tuple[float, float]]:
    """Spread each TBA period total over its period (used when there is no overlay).

    With `rate` = (times, balls/s) from the vision counter, each period's total is spread in
    proportion to the vision rate (so the timing - and therefore who gets credit - follows
    when the hub was actually busy); otherwise uniformly.
    """
    spans = {"auto": (0.0, AUTO_LEN + GRACE)}
    for name, a, b in TELEOP_PERIODS:
        spans[name] = (AUTO_LEN + a, AUTO_LEN + b + (GRACE if name == "endgame" else 0.0))
    out = []
    for name, n in period_counts.items():
        if not n or name not in spans:
            continue
        a, b = spans[name]
        ts = np.arange(a + 2.0, b + 2.0, step)  # counter lags entry by ~2 s
        vts = [to_video_time(mt) for mt in ts]
        keep = [(mt, vt) for mt, vt in zip(ts, vts) if vt is not None]
        if not keep:
            continue
        w = np.ones(len(keep))
        if rate is not None:
            rt, rv = rate
            w = np.interp([vt for _, vt in keep], rt, rv, left=0, right=0) + 0.05 * (rv.mean() if len(rv) else 0)
            if w.sum() <= 0:
                w = np.ones(len(keep))
        w = w / w.sum()
        for (mt, vt), wi in zip(keep, w):
            out.append((vt, n * wi))
    return out


def split_nnls(increments: list[tuple[float, float]], teams: list[TeamSignals], to_match_time,
               bin_s: float = 1.0, ridge: float = 4.0, log: list | None = None) -> dict[int, dict[str, float]]:
    """Per-robot gain model: counter increments ~ sum_r beta_r * (kernel-spread, zone-gated launches_r)
    + gamma_r * (in-zone indicator_r). beta/gamma >= 0 with ridge shrinkage toward a pooled value,
    then each increment is shared by the fitted per-robot contributions (exact total preserved).
    Compensates for robots whose launches are seen less often (camera angle, occlusion).
    """
    from scipy.optimize import nnls

    if not increments:
        return {ts.team: {} for ts in teams}
    t0 = min(t for t, _ in increments) - 8
    t1 = max(t for t, _ in increments) + 1
    edges = np.arange(t0, t1 + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    y = np.zeros(len(centers))
    for t, n in increments:
        i = int((t - t0) / bin_s)
        if 0 <= i < len(y):
            y[i] += n
    cols = []
    for ts in teams:
        x = np.zeros(len(centers))
        for st, sw in zip(ts.shots, ts.shot_w):
            k = kernel(centers - st)
            if k.sum() > 0:
                z = ts.zone_frac(st - 0.5, st + 0.5)
                x += sw * z * k / k.sum()
        zc = np.array([ts.zone_frac(c - LAG_HI, c - LAG_LO) for c in centers])
        cols.append(x)
        cols.append(zc * 0.25)
    X = np.stack(cols, axis=1)
    # pooled single-gain fit gives the shrinkage target
    xs = X[:, 0::2].sum(1)
    zs = X[:, 1::2].sum(1)
    g, _ = nnls(np.stack([xs, zs], 1), y)
    target = np.tile(g, len(teams))
    lam = np.sqrt(ridge * max(1.0, (X ** 2).sum() / X.shape[1]))
    Xa = np.vstack([X, lam * np.eye(X.shape[1])])
    ya = np.concatenate([y, lam * target])
    coef, _ = nnls(Xa, ya)
    contrib = X * coef[None, :]
    per_team = contrib[:, 0::2] + contrib[:, 1::2]
    out = {ts.team: {} for ts in teams}
    for t, n in increments:
        i = int((t - t0) / bin_s)
        w = per_team[i] if 0 <= i < len(per_team) else np.zeros(len(teams))
        if w.sum() <= 1e-9:
            w = np.array([ts.zone_frac(t - LAG_HI, t - LAG_LO) for ts in teams]) + 1e-6
        w = w / w.sum()
        mt = to_match_time(t - 2.0)
        per = period_of(mt) if mt is not None else "unknown"
        for j, ts in enumerate(teams):
            out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[j]
            if log is not None and mt is not None:
                log.append((round(mt, 2), ts.team, round(float(n * w[j]), 3)))
    return out
