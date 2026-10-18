"""Which robot launched a fuel arc: a 3-D ballistic check in one camera view.

For each candidate robot the ball is taken to leave LAUNCH_Z above the robot's field position and to drop into
its alliance's hub opening. That leaves one free parameter, the flight time T to the opening (equivalently the
apex height), plus the unknown launch time. Each candidate's best trajectory is projected into the view, and the
image residual says how well that candidate explains the arc. Use the residuals to compare candidates (a
likelihood ratio), not as a 3-D reconstruction: a short arc seen from one camera does not pin down depth by itself.

Depth shows up in two ways: where the candidate's launch point projects, and how fast the arc curves in the
image (gravity is f * g / depth px/s^2), so arcs of 0.2 s or more separate a near robot from one behind it.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

G = 9.81
LAUNCH_Z = 0.55        # m: where balls leave a robot (top of the bumper-to-shooter stack)
T_MIN, T_MAX = 0.55, 2.6  # s: flight time to the hub opening (dropping into it needs T > ~0.51 s)
LAUNCH_BACK = 0.5      # s: the ball left the robot at most this long before its first sighting
F_SCALE = 6.0          # px: robust (soft-L1) loss scale; tracker mislinks and blob jitter beyond this count less
_T_GRID = np.linspace(T_MIN, T_MAX, 28)
_L_GRID = np.linspace(0.0, LAUNCH_BACK, 16)


def _project(cam, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Image points (N, 2) of world points (N, 3) and whether each lies in front of the camera."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    R = cv2.Rodrigues(np.asarray(cam.rvec, dtype=np.float64))[0]
    z = P @ R[2] + float(np.asarray(cam.tvec, dtype=np.float64).reshape(3)[2])
    uv, _ = cv2.projectPoints(P, np.asarray(cam.rvec, dtype=np.float64), np.asarray(cam.tvec, dtype=np.float64),
                              np.asarray(cam.K, dtype=np.float64), np.asarray(cam.dist, dtype=np.float64))
    return uv.reshape(-1, 2), z > 0.2


def trajectory(launch: np.ndarray, target: np.ndarray, T, s) -> np.ndarray:
    """Positions (..., 3) at times s after launch of the drag-free ball that reaches `target` T s after leaving
    `launch`. T and s broadcast against each other."""
    T = np.asarray(T, dtype=np.float64)[..., None]
    s = np.asarray(s, dtype=np.float64)[..., None]
    v = (target - launch) / T
    v = v + np.array([0.0, 0.0, 0.5 * G]) * T
    return launch + v * s - np.array([0.0, 0.0, 0.5 * G]) * s * s


def _fit_one(t: np.ndarray, uv: np.ndarray, cam, launch: np.ndarray, target: np.ndarray) -> dict:
    # coarse grid over (flight time T, launch lead L = first sighting - launch time), then refine
    TT, LL = np.meshgrid(_T_GRID, _L_GRID, indexing="ij")
    s = (t[None, None, :] - t[0]) + LL[..., None]                    # (nT, nL, n) time since launch
    ok_t = s < TT[..., None]                                         # still in the air
    P = trajectory(launch, target, TT[..., None], s)                 # (nT, nL, n, 3)
    pj, front = _project(cam, P.reshape(-1, 3))
    d = np.linalg.norm(pj.reshape(*s.shape, 2) - uv[None, None], axis=-1)
    d = np.where(front.reshape(s.shape) & ok_t, d, 1e4)
    rho = 2 * F_SCALE ** 2 * (np.sqrt(1 + (d / F_SCALE) ** 2) - 1)  # soft-L1 on the point distance
    cost = rho.mean(-1)
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)

    def res(x):
        T, L = x
        P = trajectory(launch, target, T, (t - t[0]) + L)
        pj, front = _project(cam, P)
        r = pj - uv
        r[~front | ((t - t[0]) + L >= T)] = 1e3
        return r.ravel()

    x0 = np.array([_T_GRID[i], _L_GRID[j]])
    try:
        sol = least_squares(res, x0, bounds=([T_MIN, 0.0], [T_MAX, LAUNCH_BACK]), loss="soft_l1", f_scale=F_SCALE,
                            max_nfev=60)
        x, r = sol.x, sol.fun.reshape(-1, 2)
    except ValueError:
        x, r = x0, res(x0).reshape(-1, 2)
    d = np.linalg.norm(r, axis=1)
    rho = 2 * F_SCALE ** 2 * (np.sqrt(1 + (d / F_SCALE) ** 2) - 1)
    return {"cost": float(rho.mean()), "rms": float(np.sqrt(np.mean(d * d))), "med": float(np.median(d)),
            "T": float(x[0]), "lead": float(x[1])}


def _fit_soft(t: np.ndarray, uv: np.ndarray, cam, launch: np.ndarray, target: np.ndarray, prior: dict) -> dict:
    """MAP fit with the launch point and the landing point free inside Gaussian priors around the robot and the
    hub (soft-L1 on everything): x = [T, lead, dx, dy, dz, hx, hy]. "cost" = image term + prior term."""
    sp, sxy, sz, sh = prior["sigma_px"], prior["sigma_xy"], prior["sigma_z"], prior["sigma_hub"]
    g = _fit_one(t, uv, cam, launch, target)  # one-parameter fit as the start
    s0 = t - t[0]

    def res(x):
        T, L, dx, dy, dz, hx, hy = x
        P = trajectory(launch + np.array([dx, dy, dz]), target + np.array([hx, hy, 0.0]), T, s0 + L)
        pj, front = _project(cam, P)
        r = (pj - uv) / sp
        r[~front | (s0 + L >= T)] = 1e3
        return np.concatenate([r.ravel(), [dx / sxy, dy / sxy, dz / sz, hx / sh, hy / sh]])

    lo = [T_MIN, 0.0, -3 * sxy, -3 * sxy, -min(3 * sz, launch[2] - 0.1), -3 * sh, -3 * sh]
    hi = [T_MAX, LAUNCH_BACK, 3 * sxy, 3 * sxy, 3 * sz, 3 * sh, 3 * sh]
    x0 = np.clip([g["T"], g["lead"], 0, 0, 0, 0, 0], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    try:
        sol = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=200)
        x, r = sol.x, sol.fun
    except ValueError:
        x, r = x0, res(x0)
    n = len(t)
    rho = 2 * (np.sqrt(1 + r * r) - 1)
    img, pri = float(rho[:2 * n].sum()), float(rho[2 * n:].sum())
    d = np.linalg.norm(r[:2 * n].reshape(-1, 2), axis=1) * sp
    return {"cost": img + pri, "img": img, "prior": pri, "rms": float(np.sqrt(np.mean(d * d))),
            "med": float(np.median(d)), "T": float(x[0]), "lead": float(x[1]), "offset": [float(v) for v in x[2:5]],
            "hub_offset": [float(v) for v in x[5:7]], "cost1": g["cost"]}


SOFT_PRIOR = {"sigma_px": 4.0, "sigma_xy": 0.5, "sigma_z": 0.25, "sigma_hub": 0.4}


def arc_fit(track_pts, cam, cand_xy, hub_xyz, launch_z: float = LAUNCH_Z, max_pts: int | None = None,
            prior: dict | None = None) -> list[dict | None]:
    """Image residual of each candidate's best hub-bound trajectory for one launch arc.

    track_pts: (t, u, v[, ...]) rows of the arc in view-local px (fuel.FlightTrack.pts).
    cam: the view's field.Camera at the launch (CameraTrack.at(t)).
    cand_xy: each candidate robot's field position (x, y) in m at the launch (None if unknown).
    hub_xyz: the target (x, y, z) in m, or one per candidate (each robot's own alliance hub).
    Returns one dict per candidate (None where cand_xy is None): "cost" (mean soft-L1 point cost, px^2; compare
    across candidates), "rms" / "med" (px), "T" (flight time, s) and "lead" (launch-to-first-sighting time, s).
    prior: instead of the fixed launch and landing points, let both move inside Gaussian priors (keys of
    SOFT_PRIOR); "cost" is then the MAP cost in units of sigma^2 (image + prior), summed over points.
    """
    p = np.asarray(track_pts, dtype=np.float64)
    if max_pts:
        p = p[:max_pts]
    t, uv = p[:, 0], p[:, 1:3]
    hubs = np.asarray(hub_xyz, dtype=np.float64)
    if hubs.ndim == 1:
        hubs = np.repeat(hubs[None], len(cand_xy), axis=0)
    out = []
    for xy, hub in zip(cand_xy, hubs):
        if xy is None or not np.all(np.isfinite(xy)):
            out.append(None)
            continue
        L0 = np.array([xy[0], xy[1], launch_z], dtype=np.float64)
        out.append(_fit_soft(t, uv, cam, L0, hub, prior) if prior else _fit_one(t, uv, cam, L0, hub))
    return out


def log_lr(fits: list[dict | None], sigma2: float) -> np.ndarray:
    """Log-likelihood of each candidate relative to the best one, from arc_fit costs: -(cost - min cost) / sigma2
    (0 for the best, nan where there is no fit)."""
    c = np.array([f["cost"] if f is not None else np.nan for f in fits], dtype=np.float64)
    if not np.isfinite(c).any():
        return np.full(len(fits), np.nan)
    return -(c - np.nanmin(c)) / sigma2
