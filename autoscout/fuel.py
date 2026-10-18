"""Fuel in flight: moving-yellow blob detection and ballistic tracking at full frame rate.

Flying FUEL is small, fast and motion-blurred, and arrives in streams of up to ~20 balls/s.
A per-frame detector is not needed to find it: within the field airspace, anything yellow
that differs from both the previous and next frames (three-frame differencing) is a moving
ball. Blobs are linked into tracks with a constant-acceleration predictor; a track that rises
and then curves down (projectile) is a shot. Its start point is traced back to the launching
robot and its end point tells whether it went into a hub opening.

Everything works in a single view's pixel coordinates; `attribution.py` combines views.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# HSV range for FUEL yellow (OpenCV hue 0-180)
H_LO, H_HI, S_MIN, V_MIN = 17, 42, 80, 100


def yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= H_LO) & (h <= H_HI) & (s >= S_MIN) & (v >= V_MIN)).astype(np.uint8)


@dataclass
class Blob:
    u: float
    v: float
    area: float
    w: float
    h: float
    x0: float = 0.0  # bounding box top-left (same pixel frame as u, v)
    y0: float = 0.0
    src: str = "motion"  # "motion" (frame differencing) or "det" (flying-fuel YOLO)


def view_blobs(cur_g: np.ndarray, prev_g: np.ndarray, next_g: np.ndarray, cur_y: np.ndarray,
               valid: np.ndarray | None = None, diff_thr: int = 22, min_area: int = 6,
               max_area: int = 2500) -> list[Blob]:
    """Yellow pixels in the current frame that changed vs both neighbours (removes static +
    ghosting). Inputs are one view's crops of the grey frames and of the current yellow mask."""
    mov = (cv2.absdiff(cur_g, prev_g) > diff_thr) & (cv2.absdiff(cur_g, next_g) > diff_thr)
    m = (mov & (cur_y > 0)).astype(np.uint8)
    if valid is not None:
        m &= valid
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [Blob(float(cen[j, 0]), float(cen[j, 1]), float(st[j, 4]), float(st[j, 2]), float(st[j, 3]),
                 float(st[j, 0]), float(st[j, 1]))
            for j in range(1, n) if min_area <= st[j, 4] <= max_area]


def view_detections(dets: np.ndarray, view_box: tuple, valid: np.ndarray) -> np.ndarray:
    """Full-frame detections (x0, y0, x1, y1, conf) -> those centred inside this view's valid area,
    in view-local pixels."""
    x0, y0, x1, y1 = view_box
    if not len(dets):
        return dets.reshape(0, 5)
    d = dets.copy()
    d[:, [0, 2]] -= x0
    d[:, [1, 3]] -= y0
    cu = ((d[:, 0] + d[:, 2]) / 2).astype(int)
    cv = ((d[:, 1] + d[:, 3]) / 2).astype(int)
    ok = (cu >= 0) & (cu < x1 - x0) & (cv >= 0) & (cv < y1 - y0)
    ok[ok] = valid[cv[ok], cu[ok]] > 0
    return d[ok]


def merge_detections(blobs: list[Blob], dets: np.ndarray | None, conf: float = 0.25) -> list[Blob]:
    """Union of motion blobs and flying-fuel detections (view pixels; rows x0, y0, x1, y1, conf).

    A detection whose box holds a motion blob's centre (or whose centre lies in a blob's box) is
    the same ball and the motion blob is kept; a motion blob that holds two or more detection
    centres is a merged stream and is replaced by those detections; every other detection
    becomes a new blob.
    """
    if dets is None or not len(dets):
        return blobs
    d = dets[dets[:, 4] >= conf]
    if not len(d):
        return blobs
    dcu, dcv = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
    if blobs:
        B = np.array([(b.u, b.v, b.x0, b.y0, b.x0 + b.w, b.y0 + b.h) for b in blobs])
        # det centre inside blob box (n_blob x n_det), blob centre inside det box (+2 px)
        c_in_b = ((dcu[None] >= B[:, 2:3] - 1) & (dcu[None] <= B[:, 4:5] + 1) &
                  (dcv[None] >= B[:, 3:4] - 1) & (dcv[None] <= B[:, 5:6] + 1))
        b_in_d = ((B[:, 0:1] >= d[None, :, 0] - 2) & (B[:, 0:1] <= d[None, :, 2] + 2) &
                  (B[:, 1:2] >= d[None, :, 1] - 2) & (B[:, 1:2] <= d[None, :, 3] + 2))
        same = c_in_b | b_in_d
    else:
        c_in_b = same = np.zeros((0, len(d)), bool)
    out, used = [], np.zeros(len(d), bool)
    for i, b in enumerate(blobs):
        js = np.nonzero(c_in_b[i] & ~used)[0]
        if len(js) >= 2:
            used[js] = True
            out += [_det_blob(d[j]) for j in js]
        else:
            out.append(b)
    dup = same.any(axis=0) if len(blobs) else np.zeros(len(d), bool)
    out += [_det_blob(d[j]) for j in range(len(d)) if not used[j] and not dup[j]]
    return out


def _det_blob(r: np.ndarray) -> Blob:
    w, h = float(r[2] - r[0]), float(r[3] - r[1])
    return Blob(float(r[0] + r[2]) / 2, float(r[1] + r[3]) / 2, 0.785 * w * h, w, h, float(r[0]), float(r[1]), "det")


def moving_blobs(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, valid: np.ndarray | None = None,
                 diff_thr: int = 22, min_area: int = 6, max_area: int = 2500) -> list[Blob]:
    """`view_blobs` on BGR frames."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt)]
    return view_blobs(g[1], g[0], g[2], yellow_mask(cur), valid, diff_thr, min_area, max_area)


@dataclass
class FlightTrack:
    id: int
    pts: list = field(default_factory=list)  # (t, u, v, area)
    misses: int = 0
    done: bool = False

    def predict(self, t: float) -> tuple[float, float]:
        if len(self.pts) == 1:
            return self.pts[-1][1], self.pts[-1][2]
        (t1, u1, v1, _), (t2, u2, v2, _) = self.pts[-2], self.pts[-1]
        dt = t2 - t1 if t2 > t1 else 1 / 60
        vu, vv = (u2 - u1) / dt, (v2 - v1) / dt
        if len(self.pts) >= 3:
            t0, u0, v0, _ = self.pts[-3]
            dt0 = t1 - t0 if t1 > t0 else 1 / 60
            av = (vv - (v1 - v0) / dt0) / ((dt + dt0) / 2)
            av = float(np.clip(av, -2000, 6000))
        else:
            av = 0.0
        h = t - t2
        return u2 + vu * h, v2 + vv * h + 0.5 * av * h * h

    @property
    def duration(self) -> float:
        return self.pts[-1][0] - self.pts[0][0] if self.pts else 0.0

    def velocity0(self, n: int = 4) -> tuple[float, float]:
        """Initial image velocity (px/s) from the first n points."""
        p = np.array(self.pts[:max(2, min(n, len(self.pts)))])
        if len(p) < 2:
            return 0.0, 0.0
        A = np.vstack([p[:, 0] - p[0, 0], np.ones(len(p))]).T
        vu = np.linalg.lstsq(A, p[:, 1], rcond=None)[0][0]
        vv = np.linalg.lstsq(A, p[:, 2], rcond=None)[0][0]
        return float(vu), float(vv)

    def fit(self) -> dict:
        """Least-squares projectile fit u = a0 + a1 t, v = b0 + b1 t + b2 t^2."""
        p = np.array(self.pts)
        t = p[:, 0] - p[0, 0]
        A2 = np.vstack([t * t, t, np.ones_like(t)]).T
        A1 = np.vstack([t, np.ones_like(t)]).T
        bu, ru, *_ = np.linalg.lstsq(A1, p[:, 1], rcond=None)
        bv, rv, *_ = np.linalg.lstsq(A2, p[:, 2], rcond=None)
        n = len(p)
        res = float(np.sqrt(((ru[0] if len(ru) else 0) + (rv[0] if len(rv) else 0)) / max(1, n)))
        return {"au": bu, "bv": bv, "res": res, "speed0": float(np.hypot(bu[0], bv[1]))}


class FlightTracker:
    def __init__(self, gate: float = 45.0, max_misses: int = 2, min_len: int = 4):
        self.gate = gate
        self.max_misses = max_misses
        self.min_len = min_len
        self.active: list[FlightTrack] = []
        self.finished: list[FlightTrack] = []
        self._next = 0

    def update(self, t: float, blobs: list[Blob]):
        """Associate blobs to tracks: vectorised distances, greedy matching under the gate."""
        na, nb = len(self.active), len(blobs)
        used_t, used_b = set(), set()
        if na and nb:
            P = np.array([tr.predict(t) for tr in self.active])
            B = np.array([(b.u, b.v) for b in blobs])
            D = np.sqrt(((P[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            gates = np.array([self.gate if len(tr.pts) > 1 else self.gate * 1.6 for tr in self.active])
            ii, jj = np.nonzero(D < gates[:, None])
            order = np.argsort(D[ii, jj])
            for k in order:
                i, j = int(ii[k]), int(jj[k])
                if i in used_t or j in used_b:
                    continue
                b = blobs[j]
                self.active[i].pts.append((t, b.u, b.v, b.area))
                self.active[i].misses = 0
                used_t.add(i)
                used_b.add(j)
        keep = []
        for i, tr in enumerate(self.active):
            if i not in used_t:
                tr.misses += 1
            if tr.misses > self.max_misses:
                tr.done = True
                if len(tr.pts) >= self.min_len:
                    self.finished.append(tr)
            else:
                keep.append(tr)
        self.active = keep
        for j, b in enumerate(blobs):
            if j not in used_b:
                self.active.append(FlightTrack(self._next, [(t, b.u, b.v, b.area)]))
                self._next += 1

    def flush(self) -> list[FlightTrack]:
        for tr in self.active:
            if len(tr.pts) >= self.min_len:
                self.finished.append(tr)
        self.active = []
        return self.finished


class BallisticTracker:
    """Flight tracker for streams of balls (volleys of up to ~20 balls/s). Not the default.

    In a stream the greedy FlightTracker loses most arcs: a new one-point track predicts zero
    velocity and grabs the *following* ball (which has just moved into almost the same spot), and
    tracks hop to a neighbouring ball inside the wide fixed gate, so the arcs stop being smooth
    projectiles (failure-finder cycle 17). Here:
    * gates scale with the track's own step: a ball moving d px per frame is matched only within
      gate_min + gate_k * d of its prediction (x1.5 while the acceleration is unknown);
    * one optimal assignment per frame (Hungarian) instead of greedy nearest pairs, with unseeded
      one-point tracks penalised so they never win a blob that another track predicts;
    * a new track is seeded from a two-frame look-ahead: the blob at t+1 whose constant-velocity
      continuation is confirmed by a blob at t+2, moving >= seed_rmin px per frame (rejects the
      follower alias); ties prefer rising motion. Without a confirmed continuation it starts unseeded.
    Same update(t, blobs) / flush() interface as FlightTracker; frames are processed two frames late.
    """

    def __init__(self, gate_min: float = 8.0, gate_k: float = 0.6, gate_unseeded: float = 40.0,
                 seed_rmin: float = 3.0, seed_rmax: float = 90.0, seed_tol: float = 4.0, seed_tol_k: float = 0.2,
                 unseeded_penalty: float = 20.0, max_misses: int = 2, min_len: int = 4):
        from collections import deque

        self.gate_min, self.gate_k, self.gate_unseeded = gate_min, gate_k, gate_unseeded
        self.seed_rmin, self.seed_rmax, self.seed_tol, self.seed_tol_k = seed_rmin, seed_rmax, seed_tol, seed_tol_k
        self.unseeded_penalty = unseeded_penalty
        self.max_misses, self.min_len = max_misses, min_len
        self.queue = deque()
        self.active: list[FlightTrack] = []
        self.finished: list[FlightTrack] = []
        self._next = 0

    def update(self, t: float, blobs: list[Blob]):
        self.queue.append((t, blobs))
        if len(self.queue) == 3:
            t0, b0 = self.queue.popleft()
            self._step(t0, b0, self.queue[0], self.queue[1])

    def flush(self) -> list[FlightTrack]:
        while self.queue:
            t0, b0 = self.queue.popleft()
            q = list(self.queue)
            self._step(t0, b0, q[0] if q else None, q[1] if len(q) > 1 else None)
        for tr in self.active:
            if len(tr.pts) >= self.min_len:
                self.finished.append(tr)
        self.active = []
        return self.finished

    @staticmethod
    def _seed_v(tr: FlightTrack):
        return getattr(tr, "v_seed", None)

    def _predict(self, tr: FlightTrack, t: float) -> tuple[float, float]:
        if len(tr.pts) == 1:
            t0, u, v, _ = tr.pts[0]
            vs = self._seed_v(tr)
            return (u, v) if vs is None else (u + vs[0] * (t - t0), v + vs[1] * (t - t0))
        return tr.predict(t)

    def _gate(self, tr: FlightTrack, t: float) -> float:
        h = t - tr.pts[-1][0]
        if len(tr.pts) >= 2:
            (t1, u1, v1, _), (t2, u2, v2, _) = tr.pts[-2], tr.pts[-1]
            sp, k = np.hypot(u2 - u1, v2 - v1) / max(t2 - t1, 1e-6), (1.0 if len(tr.pts) >= 3 else 1.5)
        elif self._seed_v(tr) is not None:
            sp, k = float(np.hypot(*self._seed_v(tr))), 1.5
        else:
            return self.gate_unseeded
        return k * (self.gate_min + self.gate_k * sp * h)

    def _step(self, t: float, blobs: list[Blob], nxt1, nxt2):
        from scipy.spatial import cKDTree

        na, nb = len(self.active), len(blobs)
        used = np.zeros(nb, bool)
        matched = np.zeros(na, bool)
        B = np.array([(b.u, b.v) for b in blobs]).reshape(-1, 2)
        if na and nb:
            P = np.array([self._predict(tr, t) for tr in self.active])
            G = np.array([self._gate(tr, t) for tr in self.active])
            D = np.sqrt(((P[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            pen = np.array([self.unseeded_penalty if len(tr.pts) == 1 and self._seed_v(tr) is None else 0.0
                            for tr in self.active])
            C = D + pen[:, None]
            C[D > G[:, None]] = 1e9
            ri = np.nonzero((C < 1e9).any(1))[0]
            ci = np.nonzero((C < 1e9).any(0))[0]
            if len(ri):
                rr, cc = linear_sum_assignment(C[np.ix_(ri, ci)])
                for a, b in zip(rr, cc):
                    i, j = int(ri[a]), int(ci[b])
                    if C[i, j] >= 1e9:
                        continue
                    bl = blobs[j]
                    self.active[i].pts.append((t, bl.u, bl.v, bl.area))
                    self.active[i].misses = 0
                    matched[i] = used[j] = True
        keep = []
        for i, tr in enumerate(self.active):
            if not matched[i]:
                tr.misses += 1
            if tr.misses > self.max_misses:
                if len(tr.pts) >= self.min_len:
                    self.finished.append(tr)
            else:
                keep.append(tr)
        self.active = keep
        free = np.nonzero(~used)[0]
        if not len(free):
            return
        seeds = [None] * len(free)
        if nxt1 is not None and nxt2 is not None and len(nxt1[1]) and len(nxt2[1]):
            (t1, b1), (t2, b2) = nxt1, nxt2
            F1 = np.array([(b.u, b.v) for b in b1])
            F2 = cKDTree(np.array([(b.u, b.v) for b in b2]))
            near = cKDTree(F1).query_ball_point(B[free], self.seed_rmax)
            pairs = [(s, c) for s, cs in enumerate(near) for c in cs]
            if pairs:
                s_idx = np.array([p[0] for p in pairs])
                step = F1[[p[1] for p in pairs]] - B[free][s_idx]
                dstep = np.hypot(step[:, 0], step[:, 1])
                pred = F1[[p[1] for p in pairs]] + step * (t2 - t1) / max(t1 - t, 1e-6)
                d2, _ = F2.query(pred)
                tol = self.seed_tol + self.seed_tol_k * dstep
                score = np.where((dstep >= self.seed_rmin) & (d2 <= tol), d2 / tol + 0.5 * (step[:, 1] > 0), np.inf)
                order = np.lexsort((score, s_idx))  # best candidate first within each seed
                first = np.r_[True, s_idx[order][1:] != s_idx[order][:-1]]
                for k in order[first]:
                    if np.isfinite(score[k]):
                        seeds[s_idx[k]] = (float(step[k, 0] / (t1 - t)), float(step[k, 1] / (t1 - t)))
        for s, j in enumerate(free):
            b = blobs[j]
            tr = FlightTrack(self._next, [(t, b.u, b.v, b.area)])
            tr.v_seed = seeds[s]
            self._next += 1
            self.active.append(tr)


def split_at_jumps(tracks: list[FlightTrack], k: float = 0.5, floor: float = 6.0, min_len: int = 4) -> list[FlightTrack]:
    """Cut flight tracks where a point is far from the constant-acceleration prediction of the points
    before it (the track hopped to another ball): error > max(floor, k * expected step). Pieces with
    >= min_len points are kept."""
    out = []
    nid = max((tr.id for tr in tracks), default=0) + 1
    for tr in tracks:
        pieces, seg = [], [tr.pts[0]]
        for p in tr.pts[1:]:
            if len(seg) >= 2:
                pu, pv = FlightTrack(-1, seg[-3:]).predict(p[0])
                (ta, ua, va, _), (tb, ub, vb, _) = seg[-2], seg[-1]
                step = np.hypot(ub - ua, vb - va) * (p[0] - tb) / max(tb - ta, 1e-6)
                if np.hypot(p[1] - pu, p[2] - pv) > max(floor, k * step):
                    pieces.append(seg)
                    seg = [p]
                    continue
            seg.append(p)
        pieces.append(seg)
        first = True
        for s in pieces:
            if len(s) >= min_len:
                out.append(FlightTrack(tr.id if first else nid, list(s), done=True))
                nid += 0 if first else 1
                first = False
    return out


def is_shot(tr: FlightTrack, px_per_m: float, min_speed_mps: float = 3.5) -> bool:
    """A launched ball: fast, initially rising (image v decreasing), smooth projectile."""
    if len(tr.pts) < 4:
        return False
    vu, vv = tr.velocity0()
    speed = np.hypot(vu, vv) / max(px_per_m, 1e-3)
    if speed < min_speed_mps:
        return False
    if vv > -0.25 * abs(vu) and vv > -1.0 * px_per_m:  # must start going up
        return False
    f = tr.fit()
    return f["res"] < 6.0
