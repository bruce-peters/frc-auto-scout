"""Field model (WPILib blue-origin frame, metres) and per-view camera calibration.

Each camera view is calibrated automatically from the field's AprilTags: we know every tag's
3-D corner positions (official layout), detect them in many frames of the match, and fit a
pinhole camera (focal length, one radial term, pose) with OpenCV. The calibrated camera
gives the floor homography (robot positions) and projects 3-D field geometry such as the
hub openings into the image.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

IN = 0.0254
TAG_SIZE = 6.5 * IN  # black square edge (36h11 "6.5in" variant)
ASSETS = Path(__file__).resolve().parent / "assets"


def load_tags(kind: str = "welded") -> dict[int, dict]:
    tags = {}
    with open(ASSETS / f"tags_{kind}.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tags[int(row["ID"])] = {
                "c": np.array([float(row["X"]), float(row["Y"]), float(row["Z"])]) * IN,
                "yaw": np.deg2rad(float(row["Z-Rotation"])),
            }
    return tags


def tag_corners(tag: dict, size: float = TAG_SIZE) -> np.ndarray:
    """3-D corners in the order OpenCV's AprilTag-36h11 detector reports them.

    Checked empirically against broadcast frames: the detector's first corner is the
    tag's bottom-right as seen from the front (fit RMS 1.4 px vs 22 px for TL-first).
    """
    c, th = tag["c"], tag["yaw"]
    right = np.array([-np.sin(th), np.cos(th), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    h = size / 2
    return np.array([c + right * h - up * h, c - right * h - up * h,
                     c - right * h + up * h, c + right * h + up * h])


# ---------------------------------------------------------------- field geometry
L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
HUB_HALF = 47 * IN / 2
ALLIANCE_ZONE_DEPTH = 158.6 * IN


def hub_center(alliance: str) -> np.ndarray:
    return np.array(config.BLUE_HUB if alliance == "blue" else config.RED_HUB)


def hub_opening_3d(alliance: str, n: int = 6) -> np.ndarray:
    """Hexagonal top opening (41.7 in across) at 72 in height, as an n-gon."""
    cx, cy = hub_center(alliance)
    r = 41.7 * IN / 2
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a),
                     np.full(n, config.HUB_OPENING_HEIGHT)], axis=1)


def hub_box_3d(alliance: str) -> np.ndarray:
    cx, cy = hub_center(alliance)
    h = HUB_HALF
    pts = []
    for z in (0.0, config.HUB_OPENING_HEIGHT):
        for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h)):
            pts.append((cx + dx, cy + dy, z))
    return np.array(pts)


def hub_region_3d(alliance: str, kind: str) -> np.ndarray:
    """8 corners of a 3-D box used as a counting region around a hub.

    top  - the airspace over the opening (incoming / dropping balls)
    exit - just outside the four exits at the base, on the neutral-zone side
    """
    cx, cy = hub_center(alliance)
    if kind == "top":
        xs, ys, zs = (cx - 0.95, cx + 0.95), (cy - 0.95, cy + 0.95), (1.25, 3.0)
    else:
        sgn = 1 if alliance == "blue" else -1
        xa, xb = cx + sgn * 0.62, cx + sgn * 1.9
        xs, ys, zs = (min(xa, xb), max(xa, xb)), (cy - 0.8, cy + 0.8), (0.0, 0.6)
    return np.array([(x, y, z) for x in xs for y in ys for z in zs])


def region_mask(cam: "Camera", pts3: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled convex hull of projected 3-D points, as a uint8 mask of `shape` (h, w)."""
    cw = cam.center_world()
    pc = (cam.R @ (pts3 - cw).T).T
    m = np.zeros(shape, np.uint8)
    if (pc[:, 2] <= 0.1).any():
        return m
    p = cam.project(pts3).astype(np.float32)
    hull = cv2.convexHull(p).astype(np.int32)
    cv2.fillPoly(m, [hull], 1)
    return m


def in_alliance_zone(alliance: str, x: float, margin: float = 0.6) -> bool:
    """Robot centre within the zone, or close enough that its bumpers overlap it (G407)."""
    if alliance == "blue":
        return x <= ALLIANCE_ZONE_DEPTH + margin
    return x >= L - ALLIANCE_ZONE_DEPTH - margin


def field_lines() -> list[np.ndarray]:
    """Polylines (z=0) for drawing: perimeter, centre line, alliance-zone lines, hubs."""
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]])]
    lines.append(np.array([[L / 2, 0], [L / 2, W]]))
    for x in (ALLIANCE_ZONE_DEPTH, L - ALLIANCE_ZONE_DEPTH):
        lines.append(np.array([[x, 0], [x, W]]))
    for al in ("blue", "red"):
        cx, cy = hub_center(al)
        h = HUB_HALF
        lines.append(np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h],
                               [cx - h, cy + h], [cx - h, cy - h]]))
    return [np.hstack([ln, np.zeros((len(ln), 1))]) for ln in lines]


# ---------------------------------------------------------------- camera
@dataclass
class Camera:
    K: list  # 3x3
    dist: list  # k1, k2, p1, p2, k3
    rvec: list
    tvec: list
    size: tuple  # view (w, h)
    rms: float
    n_tags: int

    @property
    def Km(self):
        return np.array(self.K, dtype=np.float64)

    @property
    def D(self):
        return np.array(self.dist, dtype=np.float64)

    @property
    def R(self):
        return cv2.Rodrigues(np.array(self.rvec, dtype=np.float64))[0]

    @property
    def t(self):
        return np.array(self.tvec, dtype=np.float64).reshape(3)

    def project(self, pts3: np.ndarray) -> np.ndarray:
        pts3 = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
        p, _ = cv2.projectPoints(pts3, np.array(self.rvec), np.array(self.tvec), self.Km, self.D)
        return p.reshape(-1, 2)

    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t

    def backproject(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height z. Returns (N, 2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.Km, self.D).reshape(-1, 2)
        rays_c = np.hstack([n, np.ones((len(n), 1))])
        rays_w = rays_c @ self.R  # == (R.T @ ray).T
        c = self.center_world()
        s = (z - c[2]) / rays_w[:, 2]
        pts = c[None, :] + s[:, None] * rays_w
        out = pts[:, :2]
        out[s <= 0] = np.nan
        return out

    def at(self, t: float) -> "Camera":
        return self

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: Path) -> "Camera":
        d = json.loads(Path(path).read_text())
        d["size"] = tuple(d["size"])
        return Camera(**d)


_DETECTOR = None


def tag_detector():
    global _DETECTOR
    if _DETECTOR is None:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.minMarkerPerimeterRate = 0.005
        _DETECTOR = cv2.aruco.ArucoDetector(d, p)
    return _DETECTOR


def detect_tags(img: np.ndarray, upscale: float = 2.0) -> dict[int, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if upscale != 1:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = tag_detector().detectMarkers(g)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        if 1 <= int(i) <= 32:
            out[int(i)] = c.reshape(4, 2) / upscale
    return out


def _inits(w: int, prior: dict | None) -> list[tuple[float, np.ndarray]]:
    """(focal / view width, distortion) starting points; with an event prior its K comes first."""
    out = [(f0, np.zeros(5)) for f0 in (0.6, 0.9, 1.3, 2.0, 3.0)]
    if prior is not None:
        out.insert(0, (prior["f"] / w, np.asarray(prior["dist"], dtype=np.float64)))
    return out


def calibrate_view(frames: list[np.ndarray], tags: dict[int, dict] | None = None,
                   masks: list[tuple[int, int, int, int]] = (), prior: dict | None = None) -> Camera | None:
    """Fit a camera to a static view from AprilTag detections over several frames.

    With an event `prior` (load_calib_prior) the fit also starts from the event K, and fits whose
    focal length is more than PRIOR_F_TOL off the event median are rejected."""
    tags = tags or load_tags()
    h, w = frames[0].shape[:2]
    seen: dict[int, list[np.ndarray]] = {}
    for f in frames:
        for i, c in detect_tags(f).items():
            cx, cy = c.mean(0)
            if any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in masks):
                continue
            seen.setdefault(i, []).append(c)
    # median over frames; keep tags seen consistently (static camera)
    obs = {}
    for i, cs in seen.items():
        if len(cs) < max(2, len(frames) // 5):
            continue
        arr = np.stack(cs)
        med = np.median(arr, axis=0)
        spread = np.median(np.linalg.norm(arr - med, axis=2))
        if spread < 3.0:
            obs[i] = med
    if len(obs) < 2:
        return None
    obj = np.concatenate([tag_corners(tags[i]) for i in obs]).astype(np.float32)
    img = np.concatenate([obs[i] for i in obs]).astype(np.float32)
    best = None
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    for f0, d0 in _inits(w, prior):
        K0 = np.array([[f0 * w, 0, w / 2], [0, f0 * w, h / 2], [0, 0, 1]], dtype=np.float64)
        try:
            rms, K, D, rv, tv = cv2.calibrateCamera([obj], [img], (w, h), K0, d0.copy(), flags=flags)
        except cv2.error:
            continue
        cam = Camera(K.tolist(), D.ravel().tolist(), rv[0].ravel().tolist(),
                     tv[0].ravel().tolist(), (w, h), float(rms), len(obs))
        # camera must be in front of the tags and above the floor
        if cam.center_world()[2] < 0.3 or not focal_ok(cam, prior):
            continue
        if best is None or rms < best.rms:
            best = cam
    if best is not None and len(obs) < 4:
        # too few tags to trust a free focal length + distortion: refit with k1 fixed at 0
        pass
    return best


def draw_field(img: np.ndarray, cam: Camera, color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    for ln in field_lines():
        dense = []
        for a, b in zip(ln[:-1], ln[1:]):
            for s in np.linspace(0, 1, 30):
                dense.append(a + (b - a) * s)
        dense = np.array(dense)
        cw = cam.center_world()
        # keep points in front of the camera
        pc = (cam.R @ (dense - cw).T).T
        p = cam.project(dense)
        ok = pc[:, 2] > 0.1
        for i in range(len(p) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(out, tuple(np.int32(p[i])), tuple(np.int32(p[i + 1])), color, 2)
    for al, col in (("blue", (255, 120, 0)), ("red", (0, 0, 255))):
        hex_ = cam.project(hub_opening_3d(al, 12))
        cv2.polylines(out, [np.int32(hex_)], True, col, 2)
    return out


@dataclass
class CameraTrack:
    """A camera whose intrinsics are fixed but whose pose drifts (pans / tilts) over the match.

    Poses are fitted on sampled frames and linearly interpolated in time.
    """
    K: list
    dist: list
    times: list
    rvecs: list
    tvecs: list
    size: tuple
    rms: float
    n_tags: int

    def at(self, t: float) -> Camera:
        ts = np.asarray(self.times)
        if len(ts) == 1 or t <= ts[0]:
            i = 0 if (len(ts) == 1 or t <= ts[0]) else len(ts) - 1
            return Camera(self.K, self.dist, self.rvecs[i], self.tvecs[i], self.size, self.rms, self.n_tags)
        if t >= ts[-1]:
            return Camera(self.K, self.dist, self.rvecs[-1], self.tvecs[-1], self.size, self.rms, self.n_tags)
        j = int(np.searchsorted(ts, t))
        a = (t - ts[j - 1]) / max(1e-9, ts[j] - ts[j - 1])
        r0, r1 = np.asarray(self.rvecs[j - 1]), np.asarray(self.rvecs[j])
        th = float(np.linalg.norm(r1))
        if th > np.pi / 2:
            # a rotation by ~180 deg (Chezy's main camera looks back along -y) has two rvecs, r and
            # r * (1 - 2 pi / |r|); per-window fits return either, and blending across the two gave garbage
            # poses between windows (30 % of cc main boxes off the field). Blend on r0's branch.
            alt = r1 * (1 - 2 * np.pi / th)
            if np.linalg.norm(alt - r0) < np.linalg.norm(r1 - r0):
                r1 = alt
        rv = (1 - a) * r0 + a * r1
        tv = (1 - a) * np.asarray(self.tvecs[j - 1]) + a * np.asarray(self.tvecs[j])
        return Camera(self.K, self.dist, rv.tolist(), tv.tolist(), self.size, self.rms, self.n_tags)

    def pose_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.times) - t)))

    # Camera-like conveniences (pose at the middle of the match)
    def _mid(self) -> Camera:
        return self.at(float(np.median(self.times)))

    def project(self, pts3):
        return self._mid().project(pts3)

    def backproject(self, uv, z: float = 0.0):
        return self._mid().backproject(uv, z)

    def center_world(self):
        return self._mid().center_world()

    @property
    def R(self):
        return self._mid().R


def calibrate_view_dynamic(frames: list[np.ndarray], times: list[float], tags: dict[int, dict] | None = None,
                           masks: list[tuple[int, int, int, int]] = (), max_frame_err: float = 3.0,
                           prior: dict | None = None) -> CameraTrack | None:
    """Shared intrinsics + one pose per frame (multi-view cv2.calibrateCamera).

    `prior` (an event prior from load_calib_prior): also start from the event K and reject fits whose
    focal length is more than PRIOR_F_TOL off the event median. Without it the fit is unchanged."""
    tags = tags or load_tags()
    h, w = frames[0].shape[:2]
    obj_l, img_l, t_l, ntag = [], [], [], set()
    for f, t in zip(frames, times):
        det = {}
        for i, c in detect_tags(f).items():
            cx, cy = c.mean(0)
            if any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in masks):
                continue
            det[i] = c
        if len(det) < 2:
            continue
        obj_l.append(np.concatenate([tag_corners(tags[i]) for i in det]).astype(np.float32))
        img_l.append(np.concatenate([det[i] for i in det]).astype(np.float32))
        t_l.append(t)
        ntag |= set(det)
    if len(obj_l) < 2:
        return None
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    best = None
    all_obj, all_img, all_t = obj_l, img_l, t_l
    for f0, d0 in _inits(w, prior):
        obj_l, img_l, t_l = all_obj, all_img, all_t
        K0 = np.array([[f0 * w, 0, w / 2], [0, f0 * w, h / 2], [0, 0, 1]], dtype=np.float64)
        for _ in range(2):  # fit, drop frames with large error, refit
            try:
                rms, K, D, rv, tv = cv2.calibrateCamera(obj_l, img_l, (w, h), K0, d0.copy(), flags=flags)
            except cv2.error:
                rms = None
                break
            errs = []
            for o, im, r_, t_ in zip(obj_l, img_l, rv, tv):
                p, _ = cv2.projectPoints(o, r_, t_, K, D)
                errs.append(float(np.sqrt(((p.reshape(-1, 2) - im) ** 2).sum(1).mean())))
            keep = [i for i, e in enumerate(errs) if e <= max_frame_err]
            if len(keep) == len(errs) or len(keep) < 2:
                break
            obj_l = [obj_l[i] for i in keep]
            img_l = [img_l[i] for i in keep]
            t_l = [t_l[i] for i in keep]
        if rms is None:
            continue
        tr = CameraTrack(K.tolist(), D.ravel().tolist(), list(t_l), [r.ravel().tolist() for r in rv],
                         [t.ravel().tolist() for t in tv], (w, h), float(rms), len(ntag))
        if tr.at(t_l[0]).center_world()[2] < 0.3 or not focal_ok(tr, prior):
            continue
        # prefer the fit that explains the most frames, then the lowest error
        if best is None or (len(tr.times), -tr.rms) > (len(best.times), -best.rms):
            best = tr
    # alternative: intrinsics from a short (static) window, then per-frame solvePnP poses
    alt = _poses_from_window_intrinsics(frames, times, tags, masks, max_frame_err, prior)
    if alt is not None and (best is None or (len(alt.times), -alt.rms) > (len(best.times), -best.rms)):
        best = alt
    return best


def _poses_from_window_intrinsics(frames, times, tags, masks, max_frame_err, prior=None) -> CameraTrack | None:
    n = len(frames)
    cands = []
    for a in range(0, max(1, n - 4), max(1, n // 6)):
        cam = calibrate_view(frames[a:a + 5], tags, masks, prior)
        if cam is not None and cam.rms < 2.5:
            cands.append(cam)
    if not cands:
        return None
    base = max(cands, key=lambda c: (c.n_tags, -c.rms))
    K, D = base.Km, base.D
    t_l, rv_l, tv_l, errs, ntag = [], [], [], [], set()
    for f, t in zip(frames, times):
        det = {}
        for i, c in detect_tags(f).items():
            cx, cy = c.mean(0)
            if any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in masks):
                continue
            det[i] = c
        if len(det) < 2:
            continue
        obj = np.concatenate([tag_corners(tags[i]) for i in det]).astype(np.float64)
        img = np.concatenate([det[i] for i in det]).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(obj, img, K, D, np.array(base.rvec), np.array(base.tvec),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        p, _ = cv2.projectPoints(obj, rv, tv, K, D)
        e = float(np.sqrt(((p.reshape(-1, 2) - img) ** 2).sum(1).mean()))
        if e > max_frame_err:
            continue
        t_l.append(t)
        rv_l.append(rv.ravel().tolist())
        tv_l.append(tv.ravel().tolist())
        errs.append(e)
        ntag |= set(det)
    if not t_l:
        return None
    return CameraTrack(base.K, base.dist, t_l, rv_l, tv_l, base.size, float(np.sqrt(np.mean(np.square(errs)))),
                       len(ntag))


# ---------------------------------------------------------------- end cameras: event prior + fallback
# An end camera that cannot be fitted on its own (only 1-2 tags in view, an optimiser minimum with an
# absurd focal length, or panned away from most tags) borrows the intrinsics and mount position of the
# same view at the event's other matches: end cameras are pan/tilt heads on fixed mounts.
MAX_RMS = 4.0  # px: pipeline.calibrate keeps a plain fit only below this
PRIOR_F_TOL = 0.30  # fits whose focal length is further than this from the event median are rejected
PRIOR_MIN_MATCHES = 3  # calibrated matches needed before an event prior is used
PRIOR_MAX_SHIFT = 3.0  # m: a match's own camera-centre estimate may differ this much from the event's


def calib_prior_path(event: str, view: str) -> Path:
    return config.CACHE / f"calib_{event}_{view}.json"


def focal_ok(cam, prior: dict | None) -> bool:
    return prior is None or abs(float(cam.K[0][0]) / prior["f"] - 1.0) <= PRIOR_F_TOL


def track_centre(cam) -> np.ndarray:
    """Median camera centre over a track's poses."""
    ts = list(getattr(cam, "times", None) or [0.0])
    return np.median([cam.at(t).center_world() for t in ts], axis=0)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def update_calib_prior(event: str, view: str, match: str, cam) -> bool:
    """Record one match's calibration of `view` in the event prior (data/cache/calib_<event>_<view>.json).

    Only plain tag fits count (rms < MAX_RMS, centre near the field, not a recovered camera). The match is
    marked as seen either way, so seeding from saved scans skips it next time. Returns True if added."""
    path = calib_prior_path(event, view)
    d = _read_json(path) or {"event": event, "view": view, "matches": {}, "seen": []}
    added = False
    if cam is not None and getattr(cam, "recovered", None) is None and np.isfinite(cam.rms) and cam.rms < MAX_RMS:
        c = track_centre(cam)
        if -4 < c[0] < L + 4 and -4 < c[1] < W + 4 and 0.2 < c[2] < 6 and cam.K[0][0] > 0:
            mid = cam.at(float(np.median(list(getattr(cam, "times", None) or [0.0]))))
            d["matches"][match] = {"K": np.asarray(cam.K, float).tolist(),
                                   "dist": np.asarray(cam.dist, float).ravel().tolist(),
                                   "centre": c.round(4).tolist(), "rvec": np.asarray(mid.rvec, float).ravel().tolist(),
                                   "size": list(cam.size), "rms": round(float(cam.rms), 3), "n_tags": int(cam.n_tags)}
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


def load_calib_prior(event: str, view: str, size: tuple | None = None, exclude: str | None = None) -> dict | None:
    """Event median K, distortion and camera centre of `view` (None with fewer than PRIOR_MIN_MATCHES)."""
    d = _read_json(calib_prior_path(event, view))
    if not d:
        return None
    ent = [e for k, e in d.get("matches", {}).items() if k != exclude]
    if size is None and ent:  # the production's view size (all of an event's matches share it)
        sizes = [tuple(e["size"]) for e in ent]
        size = max(set(sizes), key=sizes.count)
    ent = [e for e in ent if tuple(e["size"]) == tuple(size)]
    if len(ent) < PRIOR_MIN_MATCHES:
        return None
    f = float(np.median([e["K"][0][0] for e in ent]))
    cx, cy = (float(np.median([e["K"][i][2] for e in ent])) for i in (0, 1))
    ref = min(ent, key=lambda e: abs(e["K"][0][0] - f))
    return {"event": event, "view": view, "n": len(ent), "size": tuple(size), "f": f,
            "K": [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            "dist": np.median([e["dist"] for e in ent], axis=0).tolist(),
            "centre": np.median([e["centre"] for e in ent], axis=0).tolist(), "rvec": ref["rvec"]}


def _detect_masked(img: np.ndarray, masks) -> dict[int, np.ndarray]:
    out = {}
    for i, c in detect_tags(img).items():
        cx, cy = c.mean(0)
        if not any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in masks):
            out[i] = c.astype(np.float64)
    return out


def _proj_err(obj, img, rv, tv, K, D) -> float:
    p, _ = cv2.projectPoints(obj, np.asarray(rv, float), np.asarray(tv, float), K, D)
    return float(np.sqrt(((p.reshape(-1, 2) - img) ** 2).sum(1).mean()))


def _rot_from_bearings(obj, img, K, D, C) -> np.ndarray:
    """rvec turning the world directions from camera centre C to `obj` onto the pixels' rays (Kabsch)."""
    n = cv2.undistortPoints(img.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    b = np.hstack([n, np.ones((len(n), 1))])
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    a = obj - C
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(a.T @ b)
    s = np.sign(np.linalg.det(Vt.T @ U.T))
    return cv2.Rodrigues(Vt.T @ np.diag([1.0, 1.0, s]) @ U.T)[0].ravel()


def _pan_refine(obj, img, K, D, C, rv0):
    """Rotation-only reprojection fit about a known camera centre. (rvec, rms px)"""
    from scipy.optimize import least_squares

    def res(rv):
        p, _ = cv2.projectPoints(obj, rv, -cv2.Rodrigues(rv)[0] @ C, K, D)
        return (p.reshape(-1, 2) - img).ravel()

    s = least_squares(res, np.asarray(rv0, float), method="lm")
    return s.x, float(np.sqrt((s.fun.reshape(-1, 2) ** 2).sum(1).mean()))


def _pan_pose(det, tags, K, D, C, thr):
    """Rotation about C from the tags that agree. One tag alone fixes the rotation (8 equations, 3
    unknowns); the hypothesis the most other tags agree with wins, which drops a misplaced tag.
    Returns (rvec, tag ids, rms px) or None."""
    corners = {i: tag_corners(tags[i]) for i in det}

    def stack(ids):
        return np.concatenate([corners[i] for i in ids]), np.concatenate([det[i] for i in ids])

    best = None
    for h in det:
        inl = [h]
        rv, e = _pan_refine(*stack(inl), K, D, C, _rot_from_bearings(corners[h], det[h], K, D, C))
        for _ in range(3):
            tv = -cv2.Rodrigues(rv)[0] @ C
            new = sorted(i for i in det if _proj_err(corners[i], det[i], rv, tv, K, D) <= thr)
            if not new or new == sorted(inl):
                break
            inl = new
            rv, e = _pan_refine(*stack(inl), K, D, C, rv)
        if e <= thr and (best is None or (len(inl), -e) > (len(best[1]), -best[2])):
            best = (rv, list(inl), e)
    return best


def _pnp_ransac(det, tags, K, D, thr):
    """Free pose from >= 2 tags, RANSAC over their corners; a tag stays if >= 3 of its corners are inliers.
    Returns (rvec, tvec, tag ids, rms px) or None."""
    ids = list(det)
    corners = {i: tag_corners(tags[i]) for i in ids}
    obj = np.concatenate([corners[i] for i in ids])
    img = np.concatenate([det[i] for i in ids])
    try:
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, D, iterationsCount=500, reprojectionError=thr,
                                             confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok or inl is None:
        return None
    inl = {int(x) for x in np.ravel(inl)}
    keep = [i for k, i in enumerate(ids) if sum((4 * k + j) in inl for j in range(4)) >= 3]
    for _ in range(3):
        if len(keep) < 2:
            return None
        o = np.concatenate([corners[i] for i in keep])
        m = np.concatenate([det[i] for i in keep])
        ok, rv, tv = cv2.solvePnP(o, m, K, D, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        new = [i for i in keep if _proj_err(corners[i], det[i], rv, tv, K, D) <= thr]
        if new == keep:
            break
        keep = new
    else:
        return None  # the inlier set did not settle
    return rv.ravel(), tv.ravel(), keep, _proj_err(o, m, rv, tv, K, D)


def _pan_bundle(obs, K, D, C0, rvs0):
    """Shared camera centre + one rotation per frame, intrinsics fixed (robust least squares).
    obs: [(obj (N,3), img (N,2))] per frame. Returns (C, [rvec])."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    n = len(obs)

    def res(x):
        C = x[:3]
        out = []
        for k, (o, m) in enumerate(obs):
            rv = x[3 + 3 * k:6 + 3 * k]
            p, _ = cv2.projectPoints(o, rv, -cv2.Rodrigues(rv)[0] @ C, K, D)
            out.append((p.reshape(-1, 2) - m).ravel())
        return np.concatenate(out)

    J = lil_matrix((2 * sum(len(o) for o, _ in obs), 3 + 3 * n), dtype=int)
    r = 0
    for k, (o, _) in enumerate(obs):
        J[r:r + 2 * len(o), :3] = 1
        J[r:r + 2 * len(o), 3 + 3 * k:6 + 3 * k] = 1
        r += 2 * len(o)
    x0 = np.concatenate([np.asarray(C0, float)] + [np.asarray(v, float) for v in rvs0])
    s = least_squares(res, x0, jac_sparsity=J, loss="huber", f_scale=1.0, max_nfev=100)
    return s.x[:3], [s.x[3 + 3 * k:6 + 3 * k] for k in range(n)]


def _rot_angle(a, b) -> float:
    """Degrees between two rvecs."""
    Ra, Rb = cv2.Rodrigues(np.asarray(a, float))[0], cv2.Rodrigues(np.asarray(b, float))[0]
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ra.T @ Rb)[0])))


def calibrate_view_fixed(frames: list[np.ndarray], times: list[float], tags: dict[int, dict] | None,
                         masks, prior: dict, max_frame_err: float = 3.0, mode: str = "pan",
                         diag: dict | None = None) -> CameraTrack | None:
    """Fallback for an end view without a fit of its own: the event K and distortion held fixed and one
    pose per sampled frame, interpolated in time like any CameraTrack.

    1. frames with >= 2 tags: free pose by solvePnPRansac (drops an outlier tag); where >= 3 tags agree the
       frame is an anchor
    2. every frame with >= 1 tag: rotation about the event camera centre (one tag fixes it), tag consensus
    3. with >= 3 anchors this match's centre is estimated too: the centre and all rotations are refined
       jointly, started from the event centre and from the anchors' median centre, and the start whose
       result has the lower truncated tag cost wins (free PnP alone can settle in a wrong basin)
    mode "pnp": frames with >= 2 agreeing tags keep their free solvePnPRansac pose instead (translation free).
    Frames whose pose jumps away from both neighbours (which agree with each other) are dropped.
    """
    tags = tags or load_tags()
    K, D = np.array(prior["K"], float), np.array(prior["dist"], float)
    h, w = frames[0].shape[:2]
    obs = [(t, _detect_masked(f, masks)) for f, t in zip(frames, times)]
    obs = [(t, d) for t, d in obs if d]
    C_ev = np.asarray(prior["centre"], float)
    info = {"frames": len(frames), "frames_with_tags": len(obs), "mode": mode,
            "tags_per_frame": np.bincount([len(d) for _, d in obs], minlength=3).tolist() if obs else [],
            "centre_from": "event", "centre": C_ev.round(3).tolist()}
    if diag is not None:
        diag.update(info)
    if not obs:
        return None
    pnp = {}
    for t, d in obs:
        if len(d) >= 2:
            r = _pnp_ransac(d, tags, K, D, max_frame_err)
            if r is not None:
                pnp[t] = r
    anchors = [-cv2.Rodrigues(r[0])[0].T @ r[1] for r in pnp.values() if len(r[2]) >= 3]

    def pan_all(C):
        return {t: r for t, d in obs if (r := _pan_pose(d, tags, K, D, C, max_frame_err)) is not None}

    def cost(C, poses):  # truncated quadratic over every tag sighting: inliers and outliers both count
        c = 0.0
        for t, d in obs:
            for i in d:
                e = max_frame_err
                if t in poses:
                    e = min(e, _proj_err(tag_corners(tags[i]), d[i], poses[t][0], -cv2.Rodrigues(poses[t][0])[0] @ C, K, D))
                c += e ** 2
        return c

    def ok(C):
        return np.linalg.norm(C - C_ev) < PRIOR_MAX_SHIFT and C[2] > 0.2

    C, poses = C_ev, pan_all(C_ev)
    if len(anchors) >= 3:
        best = (cost(C, poses), "event", C, poses)
        Ca = np.median(anchors, axis=0)
        for name, C0 in (("event+refined", C_ev), (f"{len(anchors)} anchors+refined", Ca)):
            if not ok(C0):
                continue
            p0 = pan_all(C0)
            if len(p0) < 3:
                continue
            ts = sorted(p0)
            fo = [(np.concatenate([tag_corners(tags[i]) for i in p0[t][1]]),
                   np.concatenate([dict(obs)[t][i] for i in p0[t][1]])) for t in ts]
            Cb, _ = _pan_bundle(fo, K, D, C0, [p0[t][0] for t in ts])
            if ok(Cb):
                pb = pan_all(Cb)
                cb = cost(Cb, pb)
                info[f"cost_{name.split()[-1] if 'anchors' in name else name}"] = round(cb, 1)
                if cb < best[0]:
                    best = (cb, name, Cb, pb)
        info["cost_event"] = round(cost(C_ev, poses), 1)
        _, info["centre_from"], C, poses = best
    info["centre"] = np.round(C, 3).tolist()
    rows = []  # (t, rvec, tvec, err, tag ids, how)
    for t, d in obs:
        if mode == "pnp" and t in pnp:
            rv, tv, ids, e = pnp[t]
            rows.append((t, rv, tv, e, ids, "pnp"))
        elif t in poses:
            rv, ids, e = poses[t]
            rows.append((t, rv, -cv2.Rodrigues(rv)[0] @ C, e, ids, "pan"))
    rows = [r for r in rows if r[3] <= max_frame_err]
    keep = [True] * len(rows)
    for i in range(1, len(rows) - 1):
        a, b = _rot_angle(rows[i - 1][1], rows[i][1]), _rot_angle(rows[i][1], rows[i + 1][1])
        if a > 5.0 and b > 5.0 and _rot_angle(rows[i - 1][1], rows[i + 1][1]) < 2.5:
            keep[i] = False
    rows = [r for r, k in zip(rows, keep) if k]
    info.update(poses=len(rows), dropped_jumps=keep.count(False), pnp_frames=len(pnp), anchors=len(anchors))
    if diag is not None:
        diag.update(info)
    if len(rows) < 2:
        return None
    errs = np.array([r[3] for r in rows])
    track = CameraTrack(K.tolist(), D.tolist(), [float(r[0]) for r in rows],
                        [np.asarray(r[1], float).ravel().tolist() for r in rows],
                        [np.asarray(r[2], float).ravel().tolist() for r in rows], (w, h),
                        float(np.sqrt(np.mean(errs ** 2))), len({i for r in rows for i in r[4]}))
    if diag is not None:
        diag["rms"] = round(track.rms, 3)
    return track


def recover_view(frames: list[np.ndarray], times: list[float], tags: dict[int, dict] | None, masks,
                 prior: dict, max_frame_err: float = 3.0, diag: dict | None = None) -> list[tuple[str, CameraTrack]]:
    """Candidate cameras for an end view whose plain fit failed, for pipeline.accept_recovered_cams (which
    keeps one only if the view's robots then agree with the main view's): the tag fit started from the
    event K, and the fixed-intrinsics poses (calibrate_view_fixed)."""
    out = []
    cam = calibrate_view_dynamic(frames, times, tags, masks, max_frame_err, prior=prior)
    if cam is not None and cam.rms < MAX_RMS:
        out.append(("event_K_fit", cam))
    d = {}
    fb = calibrate_view_fixed(frames, times, tags, masks, prior, max_frame_err, diag=d)
    if fb is not None:
        out.append(("fixed_K", fb))
    if diag is not None:
        diag["event_K_fit"] = None if cam is None else {"rms": round(cam.rms, 3), "f": round(cam.K[0][0], 1),
                                                         "poses": len(cam.times)}
        diag["fixed_K"] = d
    return out
