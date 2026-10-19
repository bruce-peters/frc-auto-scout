"""Which robot launched a ball: rebuild the ball's flight in 3-D from two camera views at once (experimental).

The broadcast frame holds three calibrated views (main, blue_end, red_end) sampled on the same frame grid, so a
ball tracked in two views at the same instants is seen along two rays. Where the rays meet is the ball. A
drag-free ballistic path (gravity known, 6 free numbers: position and velocity) is fitted to every sighting of
the ball in both views (image residuals, px). Followed back to the launch height it gives the field point the
ball left from, with an uncertainty from the fit, and the robot standing there is the shooter. The robot does
not need a box in either view: its fused field track (cameras combined) is enough.

Main entry points:
  MultiView(md)                          per-match index of every view's tracks, poses and rays
  MultiView.candidates(va, i, vb)        tracks of view vb whose rays meet track i's at common instants
  MultiView.fit_pair(va, i, vb, ib, tc)  joint ballistic fit of one candidate pair (fit_linear) + launch point
  MultiView.pairs_for(va, i)             the best partner in any other view if it fits to FIT_RMS_PX, else None
  launch_point(fit)                      launch time, field point (x, y) and 2x2 covariance at LAUNCH_Z
  pick_robot(launch, robots)             the robot whose position explains the launch point (gated), or None

Status (data/exp_multiview/RESULTS_section.md): not used by the pipeline. With the tag-only Chezy calibrations the
views disagree about points in the air by 0.1-0.5 m, so true pairs fit at 7-15 px and are hard to tell from
chance pairs; launch points land a median 1.4-1.6 m from the scouted shooter, and picks are not better than the
2-D link rules.

Zero-context: everything is per match (its own camera poses, its own tracks and robot tracks); the constants are
global.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import config

G = 9.81
LAUNCH_Z = 0.55          # m: where balls leave a robot
SIGMA_LAUNCH_Z = 0.2     # m: spread of real launch heights about LAUNCH_Z (folded into the launch-point cov)
PAIR_GAP_M = 1.0         # m: median closest distance between the two views' rays at common instants
PAIR_MIN_COMMON = 3      # instants seen in both views
PAIR_MAX_CANDS = 12      # joint fits tried per (anchor, other view)
PAIR_MARGIN_S = 0.1      # s: the fit uses the anchor up to the end of the overlap + this, the partner overlap +- this
FIT_RMS_PX = 6.0         # px: pairs_for accepts a pair only if the joint fit explains each view to this rms
FIT_F_SCALE = 3.0        # px: soft-L1 scale of the joint fit
SIGMA_PX_FLOOR = 1.5     # px: floor on the per-point noise used for the fit covariance
MAX_LEAD = 1.0           # s: the launch may be at most this long before the first sighting
MAX_LAG = 0.3            # s: ... or this long after it (first seen below LAUNCH_Z)
FIELD_BOX = (-1.0, config.FIELD_LENGTH + 1.0, -1.0, config.FIELD_WIDTH + 1.0, -0.3, 8.0)
SIGMA_ROBOT = 0.45       # m: shooter exit vs the robot's fused centre (robot half-size + track error)
GATE_CHI2 = 9.21         # chi^2(2) 99 %: the picked robot must explain the launch point at least this well
MARGIN_CHI2 = 2.0        # ... and beat the next robot by this much (log-likelihood ratio x 2)


def _ez():
    return np.array([0.0, 0.0, 1.0])


class ViewIndex:
    """One view's tracks as normalised (undistorted) image coordinates, per-point camera poses and world rays."""

    def __init__(self, name: str, cam_track, tracks: list):
        self.name = name
        self.cam = cam_track
        self.K = np.asarray(cam_track.K, dtype=np.float64)
        self.D = np.asarray(cam_track.dist, dtype=np.float64)
        self.f = float(0.5 * (self.K[0, 0] + self.K[1, 1]))
        self.tracks = tracks
        self._pose: dict = {}
        self._cache: dict = {}
        # time -> [(track index, point index)]
        self.by_t: dict = {}
        for i, tr in enumerate(tracks):
            for j, p in enumerate(tr.pts):
                self.by_t.setdefault(round(p[0], 3), []).append((i, j))

    def pose(self, t: float):
        k = round(t, 3)
        p = self._pose.get(k)
        if p is None:
            c = self.cam.at(t)
            R = cv2.Rodrigues(np.asarray(c.rvec, dtype=np.float64))[0]
            tv = np.asarray(c.tvec, dtype=np.float64).reshape(3)
            p = self._pose[k] = (R, tv, -R.T @ tv)
        return p

    def obs(self, i: int) -> dict:
        """Track i: t (n,), nrm (n, 2) normalised image coords, R (n, 3, 3), tv (n, 3), C (n, 3), d (n, 3) rays."""
        o = self._cache.get(i)
        if o is not None:
            return o
        p = np.asarray(self.tracks[i].pts, dtype=np.float64)
        t = p[:, 0]
        nrm = cv2.undistortPoints(p[:, 1:3].reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        P = [self.pose(x) for x in t]
        R = np.stack([q[0] for q in P])
        tv = np.stack([q[1] for q in P])
        C = np.stack([q[2] for q in P])
        d = np.einsum("nji,nj->ni", R, np.hstack([nrm, np.ones((len(t), 1))]))  # R^T [x, y, 1]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = self._cache[i] = {"t": t, "nrm": nrm, "R": R, "tv": tv, "C": C, "d": d, "f": self.f, "view": self.name,
                              "track": i, "uv": p[:, 1:3].copy(), "h": float(self.cam.size[1])}
        return o


def ray_gap(C1, d1, C2, d2):
    """Closest distance between rays (C1 + a d1) and (C2 + b d2), the midpoint, and the depths a, b (arrays)."""
    w = C1 - C2
    b_ = np.sum(d1 * d2, axis=-1)
    d_ = np.sum(d1 * w, axis=-1)
    e_ = np.sum(d2 * w, axis=-1)
    den = np.maximum(1.0 - b_ * b_, 1e-9)
    a = (b_ * e_ - d_) / den
    b = (e_ - b_ * d_) / den
    p1 = C1 + a[..., None] * d1
    p2 = C2 + b[..., None] * d2
    return np.linalg.norm(p1 - p2, axis=-1), 0.5 * (p1 + p2), a, b


def path(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Ball positions (n, 3) at times s after the reference instant; x = [P0 (3), V0 (3)]."""
    s = np.asarray(s, dtype=np.float64)[:, None]
    return x[:3] + x[3:6] * s - 0.5 * G * s * s * _ez()


def _residuals(x, obs_list, t_ref, dts=None):
    out = []
    for k, o in enumerate(obs_list):
        s = o["t"] - t_ref + (dts[k] if dts is not None else 0.0)
        X = path(x, s)
        Xc = np.einsum("nij,nj->ni", o["R"], X) + o["tv"]
        z = Xc[:, 2]
        bad = z < 0.2
        z = np.where(bad, 1.0, z)
        r = o["f"] * (Xc[:, :2] / z[:, None] - o["nrm"])
        r[bad] = 1e3
        out.append(r.ravel())
    return np.concatenate(out)


def fit_linear(obs_list: list, t_ref: float, dts=None, iters: int = 4, f_scale: float = None) -> dict | None:
    """Closed-form ballistic fit: each sighting gives two equations linear in x = [P0, V0] (the projection
    constraint Xc_xy - nrm * Xc_z = 0 with X = P0 + V0 s - g s^2 / 2), weighted by f / depth so the residuals are
    image px. Re-weighted `iters` times (depth from the previous solution; soft-L1 weights on px residuals).
    Same output as fit_path (x, t_ref, per-view rms px, parameter covariance)."""
    f_scale = FIT_F_SCALE if f_scale is None else f_scale
    rows_A, rows_b, meta = [], [], []
    for k, o in enumerate(obs_list):
        s = o["t"] - t_ref + (dts[k] if dts is not None else 0.0)
        R, tv, nrm = o["R"], o["tv"], o["nrm"]
        n = len(s)
        for c in range(2):
            a = R[:, c, :] - nrm[:, c:c + 1] * R[:, 2, :]                # (n, 3): row of (R[c] - x R[2])
            b0 = tv[:, c] - nrm[:, c] * tv[:, 2]                          # (n,)
            A = np.hstack([a, a * s[:, None]])                           # (n, 6)
            b = -b0 + a[:, 2] * 0.5 * G * s * s                          # move the gravity term over
            rows_A.append(A)
            rows_b.append(b)
        meta.append((k, n))
    A = np.vstack(rows_A)
    b = np.concatenate(rows_b)
    # depth of each equation's point (same order as A's rows) for the px weighting
    Rz = np.concatenate([np.concatenate([o["R"][:, 2, :]] * 2) for o in obs_list])
    tz = np.concatenate([np.concatenate([o["tv"][:, 2]] * 2) for o in obs_list])
    s_all = np.concatenate([np.concatenate([o["t"] - t_ref + (dts[k] if dts is not None else 0.0)] * 2)
                            for k, o in enumerate(obs_list)])
    fpx = np.concatenate([np.full(2 * len(o["t"]), o["f"]) for o in obs_list])
    w = np.ones(len(b))
    x = None
    for it in range(iters + 1):
        Aw, bw = A * w[:, None], b * w
        try:
            x = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        X = x[:3] + np.outer(s_all, x[3:]) - 0.5 * G * (s_all * s_all)[:, None] * _ez()
        z = np.sum(Rz * X, axis=1) + tz
        if np.any(z < 0.2):
            z = np.maximum(z, 0.2)
        r_px = (A @ x - b) * fpx / z                                      # px residual of each equation
        rob = 1.0 / np.sqrt(np.sqrt(1.0 + (r_px / f_scale) ** 2))        # sqrt of the soft-L1 IRLS weight
        w = fpx / z * (rob if it >= 1 else 1.0)
    rms, k0 = [], 0
    # residual rows are ordered per view: [u rows (n), v rows (n)]
    for k, n in meta:
        ru, rv = r_px[k0:k0 + n], r_px[k0 + n:k0 + 2 * n]
        rms.append(float(np.sqrt(np.mean(ru * ru + rv * rv))))
        k0 += 2 * n
    Jw = A * (fpx / z)[:, None]
    s2 = max(float(np.sum(np.minimum(r_px * r_px, 25 * f_scale ** 2)) / max(len(r_px) - 6, 1)), SIGMA_PX_FLOOR ** 2)
    try:
        cov = np.linalg.inv(Jw.T @ Jw) * s2
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    return {"x": x, "t_ref": t_ref, "rms": rms, "views": [o["view"] for o in obs_list],
            "tracks": [o["track"] for o in obs_list], "n": [len(o["t"]) for o in obs_list], "cov": cov,
            "cost": float(np.sum(np.minimum(r_px * r_px, 25 * f_scale ** 2))), "n_pts": len(r_px) // 2,
            "min_depth": float(z.min())}


def fit_path(obs_list: list, mids_t: np.ndarray, mids: np.ndarray, dts=None) -> dict | None:
    """Reference non-linear version of fit_linear (scipy least_squares, soft-L1), started from triangulated
    midpoints; ~80x slower, within a few px of fit_linear on real pairs. Not used by MultiView."""
    t_ref = float(mids_t[0])
    s = mids_t - t_ref
    A = np.zeros((3 * len(s), 6))
    b = (mids + 0.5 * G * (s * s)[:, None] * _ez()).ravel()
    for i, si in enumerate(s):
        A[3 * i:3 * i + 3, :3] = np.eye(3)
        A[3 * i:3 * i + 3, 3:] = si * np.eye(3)
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    try:
        sol = least_squares(_residuals, x0, args=(obs_list, t_ref, dts), loss="soft_l1", f_scale=FIT_F_SCALE,
                            max_nfev=100)
    except ValueError:
        return None
    x = sol.x
    r = _residuals(x, obs_list, t_ref, dts)
    n_per = [2 * len(o["t"]) for o in obs_list]
    rms, k = [], 0
    for n in n_per:
        rr = r[k:k + n].reshape(-1, 2)
        rms.append(float(np.sqrt(np.mean(np.sum(rr * rr, axis=1)))))
        k += n
    J = sol.jac
    N = len(r) // 2
    s2 = max(float(np.sum(np.minimum(r * r, 25 * FIT_F_SCALE ** 2)) / max(len(r) - 6, 1)), SIGMA_PX_FLOOR ** 2)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    return {"x": x, "t_ref": t_ref, "rms": rms, "views": [o["view"] for o in obs_list],
            "tracks": [o["track"] for o in obs_list], "n": [len(o["t"]) for o in obs_list], "cov": cov,
            "cost": float(sol.cost), "n_pts": N}


def _launch_s(x, z: float):
    """Earlier time s (rising branch) at which the path is at height z, or None."""
    a, b, c = -0.5 * G, x[5], x[2] - z
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return float(min((-b + r) / (2 * a), (-b - r) / (2 * a)))


def launch_point(fit: dict, z: float = LAUNCH_Z, sigma_z: float = SIGMA_LAUNCH_Z) -> dict | None:
    """Where (field x, y) and when the fitted path left height z on its way up, with a 2x2 covariance from the fit
    (finite differences) plus the spread of launch heights. None if the path never rose through z, or the launch
    would be more than MAX_LEAD s before the first sighting or after it."""
    x, t_ref = fit["x"], fit["t_ref"]
    s = _launch_s(x, z)
    if s is None:
        return None
    vz = x[5] - G * s
    if vz <= 0.3:
        return None
    if s < -MAX_LEAD or s > MAX_LAG:  # t_ref = the anchor's first sighting
        return None
    xy = x[:2] + x[3:5] * s

    def lp(xx, zz=z):
        ss = _launch_s(xx, zz)
        if ss is None:
            return np.array([np.nan, np.nan])
        return xx[:2] + xx[3:5] * ss

    Jl = np.zeros((2, 6))
    for i in range(6):
        h = 1e-4 * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        Jl[:, i] = (lp(xp) - lp(xm)) / (2 * h)
    cov = Jl @ fit["cov"] @ Jl.T
    dz = (lp(x, z + 0.01) - lp(x, z - 0.01)) / 0.02
    cov = cov + sigma_z ** 2 * np.outer(dz, dz)
    if not np.all(np.isfinite(cov)):
        return None
    return {"t": t_ref + s, "s": s, "xy": xy, "cov": cov, "vz": float(vz), "vxy": x[3:5].copy(),
            "dz": dz}


def _sub(o: dict, m: np.ndarray) -> dict:
    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(o["t"]) else v) for k, v in o.items()}


class MultiView:
    """Per-match index of every calibrated view's flight tracks (md.tracks_all, md.cams)."""

    def __init__(self, md, views: list | None = None, tracks_all: dict | None = None):
        ta = tracks_all if tracks_all is not None else md.tracks_all
        self.v = {}
        for name, trs in ta.items():
            if views is not None and name not in views:
                continue
            cam = md.cams.get(name)
            if cam is None or not trs:
                continue
            self.v[name] = ViewIndex(name, cam, trs)

    def candidates(self, va: str, i: int, vb: str) -> list:
        """Tracks of view vb seen at >= PAIR_MIN_COMMON of track i's instants whose rays pass within PAIR_GAP_M
        (median) of track i's, in front of both cameras and inside FIELD_BOX: [(median gap, track index, common
        times, midpoints)], best first, at most PAIR_MAX_CANDS."""
        A, B = self.v[va], self.v[vb]
        oa = A.obs(i)
        acc: dict = {}
        for ja, t in enumerate(oa["t"]):
            hits = B.by_t.get(round(t, 3))
            if not hits:
                continue
            for ib, jb in hits:
                acc.setdefault(ib, []).append((ja, jb))
        out = []
        for ib, prs in acc.items():
            if len(prs) < PAIR_MIN_COMMON:
                continue
            ob = B.obs(ib)
            ja = np.array([p[0] for p in prs])
            jb = np.array([p[1] for p in prs])
            gap, mid, a, b = ray_gap(oa["C"][ja], oa["d"][ja], ob["C"][jb], ob["d"][jb])
            g = float(np.median(gap))
            if g > PAIR_GAP_M or np.any(a <= 0) or np.any(b <= 0):
                continue
            x0, x1, y0, y1, z0, z1 = FIELD_BOX
            m = mid
            if not (np.all((m[:, 0] > x0) & (m[:, 0] < x1) & (m[:, 1] > y0) & (m[:, 1] < y1))
                    and np.all((m[:, 2] > z0) & (m[:, 2] < z1))):
                continue
            out.append((g, ib, oa["t"][ja], mid))
        out.sort(key=lambda r: r[0])
        return out[:PAIR_MAX_CANDS]

    def fit_pair(self, va: str, i: int, vb: str, ib: int, tc: np.ndarray) -> dict | None:
        """fit_linear on the anchor's points up to the end of the overlap + PAIR_MARGIN_S and the partner's points
        inside the overlap +- PAIR_MARGIN_S (tc: the common instants from candidates); t_ref = the anchor's first
        sighting. Adds "launch" (launch_point or None) and "n_common"."""
        oa, ob = self.v[va].obs(i), self.v[vb].obs(ib)
        lo, hi = float(tc.min()) - PAIR_MARGIN_S - 1e-3, float(tc.max()) + PAIR_MARGIN_S + 1e-3
        sa = _sub(oa, oa["t"] <= hi)
        sb = _sub(ob, (ob["t"] >= lo) & (ob["t"] <= hi))
        f = fit_linear([sa, sb], float(sa["t"][0]))
        if f is None:
            return None
        f["launch"] = launch_point(f)
        f["n_common"] = int(len(tc))
        return f

    def pairs_for(self, va: str, i: int, others: list | None = None) -> dict | None:
        """The anchor's best partner fit over the other views (lowest max-view rms, launch point required) if it
        explains every view to FIT_RMS_PX, else None."""
        best = None
        for vb in (others or [v for v in self.v if v != va]):
            if vb == va or vb not in self.v:
                continue
            for g, ib, tc, mid in self.candidates(va, i, vb):
                f = self.fit_pair(va, i, vb, ib, tc)
                if f is None or f["launch"] is None:
                    continue
                f["gap"], f["partner"] = g, (vb, int(ib))
                if best is None or max(f["rms"]) < max(best["rms"]):
                    best = f
        return best if best is not None and max(best["rms"]) <= FIT_RMS_PX else None


def pick_robot(launch: dict, robots: dict, sigma_robot: float = SIGMA_ROBOT, gate: float = GATE_CHI2,
               margin: float = MARGIN_CHI2) -> dict:
    """Which robot the launch point belongs to. robots: {team: (x, y)} positions at the launch time (fused field
    tracks; robots with no box in any view are fine). Returns {"team": the pick or None, "d2": Mahalanobis^2 per
    team, "best", "second"}: the pick must pass the gate (d2 <= gate) and beat the next robot by margin."""
    S = launch["cov"] + sigma_robot ** 2 * np.eye(2)
    Si = np.linalg.inv(S)
    d2 = {}
    for tm, xy in robots.items():
        if xy is None:
            continue
        r = np.asarray(xy[:2], dtype=np.float64) - launch["xy"]
        d2[tm] = float(r @ Si @ r)
    if not d2:
        return {"team": None, "d2": d2}
    order = sorted(d2, key=d2.get)
    best = order[0]
    second = d2[order[1]] if len(order) > 1 else np.inf
    ok = d2[best] <= gate and second - d2[best] >= margin
    return {"team": best if ok else None, "best": best, "d2": d2, "second": second}
