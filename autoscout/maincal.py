"""Main-camera calibration for views whose per-window tag fits disagree (failure-finder cycle 8).

The Chezy Champs main camera sees only tags 2, 11, 21 and 24: one vertical plane, one height, ~12 px wide,
20-30 m away. Two things went wrong there:

* its rotation is ~180 deg, where a rotation has two Rodrigues vectors; the per-window fits returned either
  and CameraTrack.at blended across them (30 % of main-view boxes off the field). Fixed in field.CameraTrack.at;
* each 4 s window is fitted on its own, so the camera centre wanders 0.5-1.6 m between windows and the focal
  length / distance differ from match to match (the tags fix the image of a thin strip and little else).

This module, used by pipeline.calibrate / pipeline.accept_main_cam and scripts/recalibrate_main.py only for main
views whose window centres scatter by more than MAIN_SCATTER_M (centre_scatter; no other event's main view does):

* **static camera** (fit_static): one focal length, k1 and camera centre for every window of a match, one
  rotation per window (a pan/tilt head turns, it does not move), started from the stored fit and from the
  planar-PnP solutions (ippe_start: cv2.solvePnPGeneric SOLVEPNP_IPPE, physically impossible ones dropped);
  rvecs kept on one branch (unwrap_rvecs). Optional terms (experiments): robot boxes on the field, a far-wall
  coverage quantile, fixed landmark points;
* **event prior** (load_main_prior / update_main_prior): the median K, distortion and centre of the event's
  static fits (data/cache/calib_<event>_main.json), or a camera checked against hand-picked field landmarks
  (calib_<event>_main_landmarks.json, trusted); static_from_prior poses a match by rotation only about it;
* **floor guard** (guard): a window pose that puts >= GUARD_SHARE of its own boxes off the field is dropped and
  its time interpolated from the neighbours; a match where most poses are bad is flagged.

The tags cannot fix the far side of the Chezy field (compressed by ~1.5 m in every tag-only fit; the low end
cameras share it): see data/exp_maincal/RESULTS_section.md.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from . import config, field as F, robots

L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
FLOOR_Z = 0.03  # box bottom-centre height used by pipeline.analyse
FLOOR_CAP = 2.0  # m: a floor residual is capped here (people, robots on the cart, wrong boxes)
W_FLOOR = 2.0  # px of tag reprojection per metre of floor violation
FLOOR_MAX_BOXES = 4000  # boxes used by one fit (evenly subsampled)
GUARD_SHARE = 0.20  # a window pose putting this share of its own boxes off the field is dropped ...
GUARD_MIN_N = 10  # ... if it has at least this many boxes
FLAG_SHARE = 0.5  # a match is flagged when more than this share of its poses are dropped
OFF_MARGIN = 0.4  # robots.in_field margin (as analyse)


# ---------------------------------------------------------------- floor points, as pipeline.analyse
def floor_xy(cam: F.Camera, uv: np.ndarray, half_depth: float) -> np.ndarray:
    """(N, 2) field points of box bottom-centres `uv` (view pixels), NaN where the ray misses the floor."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    if not len(uv):
        return np.zeros((0, 2))
    xy = cam.backproject(uv, z=FLOOR_Z)
    c = cam.center_world()[:2]
    dv = xy - c
    rng = np.hypot(dv[:, 0], dv[:, 1])
    return xy + half_depth * dv / np.maximum(rng, 1e-6)[:, None]


def off_field(cam, boxes: np.ndarray, half_depth: float) -> np.ndarray:
    """Bool per box (rows t, u, v): no field position under `cam` (a Camera or CameraTrack), as analyse."""
    boxes = np.asarray(boxes, float).reshape(-1, 3)
    out = np.ones(len(boxes), bool)
    if not len(boxes):
        return out
    ts, inv = np.unique(boxes[:, 0], return_inverse=True)
    for k, t in enumerate(ts):
        sel = np.flatnonzero(inv == k)
        xy = floor_xy(cam.at(float(t)), boxes[sel, 1:], half_depth)
        ok = np.all(np.isfinite(xy), axis=1)
        ok[ok] = [robots.in_field(p, OFF_MARGIN) for p in xy[ok]]
        out[sel] = ~ok
    return out


def window_off(track, boxes: np.ndarray, half_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Per pose of `track`: the boxes nearest to it in time, and how many of them that pose itself (not the
    blend with a neighbour) puts off the field."""
    ts = list(getattr(track, "times", [0.0]))
    n, o = np.zeros(len(ts), int), np.zeros(len(ts), int)
    boxes = np.asarray(boxes, float).reshape(-1, 3)
    if not len(boxes):
        return n, o
    idx = np.abs(boxes[:, :1] - np.asarray(ts)[None, :]).argmin(1)
    for i, t in enumerate(ts):
        sel = idx == i
        if sel.any():
            xy = floor_xy(track.at(float(t)), boxes[sel, 1:], half_depth)
            ok = np.all(np.isfinite(xy), axis=1)
            ok[ok] = [robots.in_field(p, OFF_MARGIN) for p in xy[ok]]
            n[i], o[i] = int(sel.sum()), int((~ok).sum())
    return n, o


def unwrap_rvecs(rvecs) -> list[list[float]]:
    """Each rvec on the branch nearest the previous one (r and r * (1 - 2 pi / |r|) are the same rotation)."""
    out = []
    for r in rvecs:
        r = np.asarray(r, float).ravel()
        th = float(np.linalg.norm(r))
        if out and th > np.pi / 2:
            alt = r * (1 - 2 * np.pi / th)
            if np.linalg.norm(alt - out[-1]) < np.linalg.norm(r - out[-1]):
                r = alt
        out.append(r)
    return [r.tolist() for r in out]


def _subtrack(track, keep: list[int]):
    return F.CameraTrack(track.K, track.dist, [track.times[i] for i in keep], [track.rvecs[i] for i in keep],
                         [track.tvecs[i] for i in keep], track.size, track.rms, track.n_tags)


def guard(track, boxes: np.ndarray, half_depth: float, share: float = GUARD_SHARE, min_n: int = GUARD_MIN_N):
    """Safety net: drop window poses that put >= `share` of their own boxes off the field (their times are
    then interpolated from the neighbouring poses), re-check once. Returns (track, report); report["flag"]
    is set when more than FLAG_SHARE of the poses had to go (the match's main camera is not trustworthy)."""
    if not hasattr(track, "times") or len(track.times) < 2:
        return track, {"dropped": 0, "bad": 0, "poses": 1, "flag": False}
    keep = list(range(len(track.times)))
    bad_all: set = set()
    for _ in range(2):
        tr = _subtrack(track, keep)
        n, o = window_off(tr, boxes, half_depth)
        bad = {keep[j] for j in range(len(keep)) if n[j] >= min_n and o[j] >= share * n[j]}
        bad_all |= bad
        if not bad or len(keep) - len(bad) < 2:
            break
        keep = [i for i in keep if i not in bad]
    dropped = len(track.times) - len(keep)
    out = track if not dropped else _subtrack(track, keep)
    for a in ("recovered", "static"):
        if hasattr(track, a):
            setattr(out, a, getattr(track, a))
    rep = {"dropped": dropped, "bad": len(bad_all), "poses": len(track.times),
           "flag": len(bad_all) > FLAG_SHARE * len(track.times)}
    return out, rep


# ---------------------------------------------------------------- tags
def tag_observations(crops: list[np.ndarray], times: list[float], masks) -> list[tuple[float, dict]]:
    return [(float(t), d) for c, t in zip(crops, times) if (d := F._detect_masked(c, masks))]


def _stack(det: dict, tags: dict):
    ids = sorted(det)
    return (np.concatenate([F.tag_corners(tags[i]) for i in ids]).astype(np.float64),
            np.concatenate([np.asarray(det[i], np.float64) for i in ids]))


def _coplanar(obj: np.ndarray) -> bool:
    s = np.linalg.svd(obj - obj.mean(0), compute_uv=False)
    return bool(s[-1] < 1e-3 * s[0])


def ippe_centres(obs, tags, K, D) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]]:
    """Per frame: both planar PnP solutions (rvec, tvec, centre, rms px) of its tags, when they are coplanar."""
    out = []
    for _, det in obs:
        obj, img = _stack(det, tags)
        sols = []
        if len(obj) >= 4 and _coplanar(obj):
            try:
                n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                n = 0
            for k in range(n):
                rv, tv = rvs[k].ravel(), tvs[k].ravel()
                sols.append((rv, tv, -cv2.Rodrigues(rv)[0].T @ tv, F._proj_err(obj, img, rv, tv, K, D)))
        out.append(sols)
    return out


def _tags_in_front(obj, rv, C) -> bool:
    """Camera centre on the visible side of every tag (the tags face it) and above the floor."""
    return bool(C[2] > 0.3 and np.all(((cv2.Rodrigues(np.asarray(rv, float))[0] @ (obj - C).T)[2]) > 0))


def ippe_start(obs, tags, K, D) -> np.ndarray | None:
    """Camera centre to start a static fit from: the median of the per-frame IPPE solutions that are physically
    possible, choosing per frame the solution closest to the other frames' (two rounds)."""
    sols = ippe_centres(obs, tags, K, D)
    cand = []
    for (_, det), s in zip(obs, sols):
        obj, _ = _stack(det, tags)
        cand.append([c for rv, tv, c, e in s if e < 3.0 and _tags_in_front(obj, rv, c)])
    allc = [c for cs in cand for c in cs]
    if not allc:
        return None
    m = np.median(allc, axis=0)
    for _ in range(2):
        pick = [min(cs, key=lambda c: np.linalg.norm(c - m)) for cs in cand if cs]
        m = np.median(pick, axis=0)
    return m


# ---------------------------------------------------------------- static fit
def _rot_about(det, tags, K, D, C, rv0):
    obj, img = _stack(det, tags)
    return F._pan_refine(obj, img, K, D, C, rv0)


def fit_static(obs, tags, size, K0, D0, C0, rvs0=None, boxes: np.ndarray | None = None, half_depth: float = 0.4,
               w_floor: float = W_FLOOR, fix_intrinsics: bool = False, fix_centre: bool = False,
               max_nfev: int = 200, points=None, w_points: float = 1.0, far_target: float | None = None,
               w_far: float = 20.0, far_q: float = 0.005, fix_k1: bool = False) -> F.CameraTrack | None:
    """Joint fit over the windows of one match: shared focal length, k1 and camera centre, one rotation per
    window, robust least squares on the tag corners (px) plus, with `boxes` (rows t, u, v: robot box
    bottom-centres in view pixels), w_floor * the distance (m, capped at FLOOR_CAP) by which each box's floor
    point lies outside the field. `points` ((N, 3) field points, (N, 2) pixels) are extra fixed features seen
    in every window (weight w_points per px). With `far_target` (m) the far_q quantile of the boxes' distance
    from the far side wall (the side away from the camera) is pulled to it (w_far px per m): robots do reach
    the far wall, which the tags alone cannot tell. Returns a CameraTrack (its rms is the tag rms, px)
    marked `static`."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    obs = [(t, d) for t, d in obs if d]
    if len(obs) < 2:
        return None
    w, h = size
    cx, cy = float(K0[0][2]), float(K0[1][2])
    K0, D0 = np.asarray(K0, float), np.asarray(D0, float).ravel()
    C0 = np.asarray(C0, float)
    nW = len(obs)
    times = np.array([t for t, _ in obs])
    stacks = [_stack(d, tags) for _, d in obs]
    if rvs0 is None:
        rvs0 = [F._rot_from_bearings(o, m, K0, D0, C0) for o, m in stacks]
    rvs0 = [np.asarray(_rot_about(d, tags, K0, D0, C0, rv)[0]) for (_, d), rv in zip(obs, rvs0)]
    bx = np.zeros((0, 2))
    bwin = np.zeros(0, int)
    if boxes is not None and len(boxes):
        b = np.asarray(boxes, float)
        if len(b) > FLOOR_MAX_BOXES:
            b = b[np.linspace(0, len(b) - 1, FLOOR_MAX_BOXES).round().astype(int)]
        bx = b[:, 1:3]
        bwin = np.abs(b[:, :1] - times[None, :]).argmin(1)
    n_tag = sum(2 * len(o) for o, _ in stacks)
    pobj = np.zeros((0, 3)) if points is None else np.asarray(points[0], float).reshape(-1, 3)
    pimg = np.zeros((0, 2)) if points is None else np.asarray(points[1], float).reshape(-1, 2)
    free_i = [] if fix_intrinsics else ([0] if fix_k1 else [0, 1])
    free_c = [] if fix_centre else [2, 3, 4]
    glob = free_i + free_c
    x_fixed = np.concatenate([[np.log(K0[0, 0]), D0[0]], C0])
    far_y0 = bool(C0[1] > W / 2)  # camera beyond the y = W side: the far wall is y = 0

    def unpack(x):
        g = x_fixed.copy()
        g[glob] = x[:len(glob)]
        return np.exp(g[0]), g[1], g[2:5], x[len(glob):].reshape(nW, 3)

    def res(x):
        f, k1, C, R = unpack(x)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        D = np.array([k1, 0, 0, 0, 0.0])
        out = []
        Rm = np.stack([cv2.Rodrigues(r)[0] for r in R])
        for i, (o, m) in enumerate(stacks):
            p, _ = cv2.projectPoints(o, R[i], -Rm[i] @ C, K, D)
            out.append((p.reshape(-1, 2) - m).ravel())
        if len(bx):
            n = cv2.undistortPoints(bx.reshape(-1, 1, 2), K, D).reshape(-1, 2)
            rc = np.hstack([n, np.ones((len(n), 1))])
            rw = np.einsum("nji,nj->ni", Rm[bwin], rc)  # R.T @ ray
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (FLOOR_Z - C[2]) / rw[:, 2]
            xy = C[None, :2] + s[:, None] * rw[:, :2]
            dv = xy - C[None, :2]
            xy = xy + half_depth * dv / np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-6)[:, None]
            dx = np.maximum(0.0, -xy[:, 0]) + np.maximum(0.0, xy[:, 0] - L)
            dy = np.maximum(0.0, -xy[:, 1]) + np.maximum(0.0, xy[:, 1] - W)
            e = np.minimum(np.hypot(dx, dy), FLOOR_CAP)
            e[~(s > 0) | ~np.isfinite(e)] = FLOOR_CAP
            out.append(w_floor * e)
        for i in range(nW if len(pobj) else 0):
            p, _ = cv2.projectPoints(pobj, R[i], -Rm[i] @ C, K, D)
            out.append(w_points * (p.reshape(-1, 2) - pimg).ravel())
        if far_target is not None and len(bx):
            far = xy[:, 1] if far_y0 else W - xy[:, 1]
            v = far[(s > 0) & np.isfinite(far)]
            out.append(np.array([w_far * (np.quantile(v, far_q) - far_target) if len(v) else 0.0]))
        return np.concatenate(out)

    ng = len(glob)
    n_pt = 2 * len(pobj)
    n_far = int(far_target is not None and len(bx) > 0)
    J = lil_matrix((n_tag + len(bx) + nW * n_pt + n_far, ng + 3 * nW), dtype=int)
    J[:, :ng] = 1
    r = 0
    for i, (o, _) in enumerate(stacks):
        J[r:r + 2 * len(o), ng + 3 * i:ng + 3 * i + 3] = 1
        r += 2 * len(o)
    for j, i in enumerate(bwin):
        J[n_tag + j, ng + 3 * i:ng + 3 * i + 3] = 1
    for i in range(nW if n_pt else 0):
        r = n_tag + len(bx) + i * n_pt
        J[r:r + n_pt, ng + 3 * i:ng + 3 * i + 3] = 1
    if n_far:
        J[-1, :] = 1
    x0 = np.concatenate([x_fixed[glob], np.concatenate(rvs0)])
    try:
        s = least_squares(res, x0, jac_sparsity=J, loss="soft_l1", f_scale=1.0, max_nfev=max_nfev, x_scale="jac")
    except (ValueError, cv2.error):
        return None
    f, k1, C, R = unpack(s.x)
    if not (np.isfinite(f) and f > 0 and C[2] > 0.3):
        return None
    K = [[float(f), 0.0, cx], [0.0, float(f), cy], [0.0, 0.0, 1.0]]
    D = [float(k1), 0.0, 0.0, 0.0, 0.0]
    errs = [F._proj_err(o, m, R[i], -cv2.Rodrigues(R[i])[0] @ C, np.array(K), np.array(D))
            for i, (o, m) in enumerate(stacks)]
    tr = F.CameraTrack(K, D, times.tolist(), unwrap_rvecs(R),
                       [(-cv2.Rodrigues(r_)[0] @ C).tolist() for r_ in R], tuple(size),
                       float(np.sqrt(np.mean(np.square(errs)))), len({i for _, d in obs for i in d}))
    tr.static = {"centre": np.round(C, 3).tolist(), "floor_boxes": int(len(bx)), "cost": float(s.cost)}
    return tr


def static_from_prior(obs, tags, prior: dict, max_frame_err: float = 3.0) -> F.CameraTrack | None:
    """Event K, distortion and camera centre held fixed; each window's rotation from its tags (tag consensus,
    field._pan_pose). For a camera that does not move between matches."""
    K, D = np.array(prior["K"], float), np.array(prior["dist"], float)
    C = np.asarray(prior["centre"], float)
    rows = []
    for t, det in obs:
        r = F._pan_pose(det, tags, K, D, C, max_frame_err)
        if r is not None:
            rows.append((t, r[0], r[1], r[2]))
    if len(rows) < 2:
        return None
    tr = F.CameraTrack(K.tolist(), D.tolist(), [float(r[0]) for r in rows], unwrap_rvecs([r[1] for r in rows]),
                       [(-cv2.Rodrigues(np.asarray(r[1], float))[0] @ C).tolist() for r in rows], tuple(prior["size"]),
                       float(np.sqrt(np.mean([r[3] ** 2 for r in rows]))), len({i for r in rows for i in r[2]}))
    tr.static = {"centre": np.round(C, 3).tolist(), "from": "landmarks" if prior.get("landmarks") else "event prior",
                 "n": prior.get("n")}
    return tr


def track_start(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K, dist, median centre) of an existing camera or track, to start a static fit from."""
    return np.asarray(cam.K, float), np.asarray(cam.dist, float).ravel(), F.track_centre(cam)


def main_candidates(obs, tags, size, plain, prior: dict | None = None) -> list[tuple[str, F.CameraTrack]]:
    """Candidates for a match's main camera from its tag sightings (for pipeline.accept_main_cam): the static
    fit (tags only), and with an event prior the rotation-only track about the event centre."""
    if plain is None or not obs or size is None:
        return []
    try:
        return static_candidates(obs, tags, size, plain=plain, prior=prior)
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return []


def static_candidates(obs, tags, size, plain=None, prior: dict | None = None, boxes=None,
                      half_depth: float = 0.4) -> list[tuple[str, F.CameraTrack]]:
    """Candidate main cameras for one match: the static fit started from the plain fit and from the IPPE
    centre (with the floor term when `boxes` are given), and the event-prior track."""
    out = []
    starts = []
    if plain is not None:
        starts.append(("plain", *track_start(plain)))
    if starts:
        K, D = starts[0][1], starts[0][2]
        c = ippe_start(obs, tags, K, D)
        if c is not None:
            starts.append(("ippe", K, D, c))
    fits = []
    for name, K, D, C in starts:
        tr = fit_static(obs, tags, size, K, D, C, boxes=boxes, half_depth=half_depth)
        if tr is not None and tr.rms < F.MAX_RMS:
            fits.append((tr.static["cost"], name, tr))
    if fits:
        cost, name, tr = min(fits, key=lambda x: x[0])
        tr.static["start"] = name
        out.append(("static_floor" if boxes is not None and len(boxes) else "static", tr))
    if prior is not None:
        tr = static_from_prior(obs, tags, prior)
        if tr is not None and tr.rms < F.MAX_RMS:
            out.append(("event_prior", tr))
    return out


# ---------------------------------------------------------------- when to refit, and the event prior
MAIN_SCATTER_M = 0.4  # m: refit a main view whose window centres scatter more than this (centre_scatter)


def centre_scatter(cam) -> float:
    """Median distance (m) of a track's window camera centres from their median: how much the per-window fits
    disagree about where the camera is. Chezy main views 0.5-1.6 m; every other main view checked <= 0.28 m."""
    ts = list(getattr(cam, "times", None) or [])
    if len(ts) < 3:
        return 0.0
    c = np.array([cam.at(float(t)).center_world() for t in ts])
    return float(np.median(np.linalg.norm(c - np.median(c, axis=0), axis=1)))


def landmark_prior_path(event: str):
    return F.config.CACHE / f"calib_{event}_main_landmarks.json"


def load_main_prior(event: str | None, size=None, exclude: str | None = None, path=None) -> tuple[dict | None, bool]:
    """(event camera, trusted) for a main view that does not move between matches.

    Trusted: a camera fitted to hand-picked field landmarks besides the tags (`path`, else
    data/cache/calib_<event>_main_landmarks.json; K, dist, centre, rvec, size and "landmarks" rows X, Y, Z, u, v;
    built by data/exp_maincal/tools/m19_prior_lm.py). Otherwise the median K, distortion and centre of the
    event's per-match static fits (data/cache/calib_<event>_main.json, field.load_calib_prior, >= 3 matches,
    `exclude` left out): consistent across matches, but it keeps the tags' far-side compression."""
    d = F._read_json(path) if path is not None else (F._read_json(landmark_prior_path(event)) if event else None)
    if d and d.get("landmarks") and all(k in d for k in ("K", "dist", "centre", "size")) and             (size is None or tuple(d["size"]) == tuple(size)):
        return d, True
    if event is None or path is not None:
        return None, False
    return F.load_calib_prior(event, "main", size, exclude=exclude), False


def update_main_prior(event: str, match: str, cam) -> bool:
    """Record a match's static main fit (tags only, fit_static) in data/cache/calib_<event>_main.json, which
    field.load_calib_prior reads. Plain per-window fits of a view that needed a refit, and tracks built from
    the prior itself, never feed it. Returns True if added."""
    path = F.calib_prior_path(event, "main")
    d = F._read_json(path) or {"event": event, "view": "main", "matches": {}, "seen": []}
    st = getattr(cam, "static", None)
    added = False
    if st is not None and "from" not in st and np.isfinite(cam.rms) and cam.rms < F.MAX_RMS:
        mid = cam.at(float(np.median(cam.times)))
        d["matches"][match] = {"K": np.asarray(cam.K, float).tolist(), "dist": np.asarray(cam.dist, float).ravel().tolist(),
                               "centre": np.round(np.asarray(st["centre"], float), 4).tolist(),
                               "rvec": np.asarray(mid.rvec, float).ravel().tolist(), "size": list(cam.size),
                               "rms": round(float(cam.rms), 3), "n_tags": int(cam.n_tags)}
        added = True
    if match not in d["seen"]:
        d["seen"].append(match)
    try:  # best effort: parallel batch shards may write the same file
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
    return added
