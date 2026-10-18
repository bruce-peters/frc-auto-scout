"""Robot detection, multi-view fusion and tracking in field coordinates.

Pipeline per sampled frame:
  1. YOLO robot detector on each camera view (alliance from the class or bumper colour).
  2. Each box's bottom-centre (bumper on the carpet) is back-projected through that view's
     calibrated camera to field metres.
  3. Same-alliance observations from different views within FUSE_DIST are merged.
  4. A 3-slot-per-alliance tracker (Kalman, Hungarian assignment, speed gating) keeps one
     slot per robot. Slot histories are cut into *segments* whenever identity could have
     been swapped (two same-alliance slots close together, or a slot re-acquired after a
     long gap). `identity.py` then labels segments with team numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config

FUSE_DIST = 1.6  # m (end cameras are low; their far positions are less precise)
DUP_DIST = 0.7  # m: two same-alliance tracks this close for DUP_STEPS steps are one robot
DUP_STEPS = 3
GATE = 1.6  # m, base association gate
MAX_SPEED = 6.0  # m/s (20 ft/s)
SWAP_DIST = 1.1  # m: same-alliance slots closer than this may swap identities
LOST_AFTER = 3.0  # s without observation before a slot is considered lost


@dataclass
class Detection:
    t: float  # video seconds
    view: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in full-frame pixels
    conf: float
    alliance: str  # 'red' | 'blue'
    xy: tuple[float, float] | None = None  # field metres
    crop_id: int | None = None  # index into a crop store for OCR / appearance
    fill: float | None = None  # fraction of FUEL-yellow pixels in the box (hopper fill proxy)


def bumper_alliance(crop: np.ndarray) -> tuple[str | None, float]:
    """Classify alliance from the lower third of a robot box (bumper colour)."""
    if crop.size == 0:
        return None, 0.0
    h = crop.shape[0]
    band = crop[int(h * 0.55):, :]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    s_ok = (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    hue = hsv[..., 0]
    red = s_ok & ((hue < 8) | (hue > 165))
    blue = s_ok & (hue > 95) & (hue < 130)
    r, b = red.mean(), blue.mean()
    if max(r, b) < 0.04:
        return None, 0.0
    return ("red", r / (r + b + 1e-6)) if r > b else ("blue", b / (r + b + 1e-6))


def in_field(xy, margin: float = 0.4) -> bool:
    x, y = xy
    return (-margin <= x <= config.FIELD_LENGTH + margin) and (-margin <= y <= config.FIELD_WIDTH + margin)


FUSION = "cov"  # "cov": uncertainty-weighted multi-view fusion; "greedy": legacy distance merge
CHI2_GATE = 9.21  # 99 % for 2 dof
TRACK_CHI2 = 13.8  # 99.9 %: track <-> observation gate
GATE_NEAR = 0.8  # m: always accept an observation this close to the prediction
PROCESS_Q = 4.0  # Kalman process noise (acceleration variance, m^2/s^4)
FUSE_CAP = 2.5  # m, never merge further apart than this


def fuse_cov(dets: list[Detection]) -> list[dict]:
    """Merge same-alliance detections of one time step across views using their covariances.

    Each detection carries `cov` (2x2, field metres) from its camera geometry: the low end
    cameras are very uncertain along their line of sight, the high main camera much less.
    Pairs from different views are matched by Mahalanobis distance (Hungarian, one detection
    per view per robot) and merged with inverse-covariance weights, so a precise main-camera
    fix dominates while an end camera still corrects the direction it measures well.
    """
    out = []
    for al in ("red", "blue"):
        cand = [d for d in dets if d.alliance == al and d.xy is not None]
        groups: list[list[Detection]] = []
        by_view: dict[str, list[Detection]] = {}
        for d in cand:
            by_view.setdefault(d.view, []).append(d)
        # start from the most precise view, then attach the others one view at a time
        order = sorted(by_view, key=lambda v: np.mean([np.trace(_cov(d)) for d in by_view[v]]))
        for v in order:
            ds = by_view[v]
            if not groups:
                groups = [[d] for d in ds]
                continue
            gx = [_merge(g) for g in groups]
            cost = np.full((len(groups), len(ds)), 1e6)
            for i, (mx, mc) in enumerate(gx):
                for j, d in enumerate(ds):
                    diff = np.asarray(d.xy) - mx
                    if np.hypot(*diff) > FUSE_CAP:
                        continue
                    m2 = float(diff @ np.linalg.solve(mc + _cov(d), diff))
                    if m2 < CHI2_GATE:
                        cost[i, j] = m2
            used = set()
            if len(groups) and len(ds):
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] < 1e6:
                        groups[i].append(ds[j])
                        used.add(j)
            groups += [[d] for j, d in enumerate(ds) if j not in used]
        for g in groups:
            xy, cov = _merge(g)
            out.append({"alliance": al, "xy": xy, "cov": cov, "dets": g,
                        "conf": float(max(d.conf for d in g))})
    return out


def _cov(d: Detection) -> np.ndarray:
    c = getattr(d, "cov", None)
    return np.asarray(c) if c is not None else np.eye(2) * 0.25


def _merge(group: list[Detection]) -> tuple[np.ndarray, np.ndarray]:
    info = np.zeros((2, 2))
    vec = np.zeros(2)
    for d in group:
        ci = np.linalg.inv(_cov(d))
        info += ci
        vec += ci @ np.asarray(d.xy)
    cov = np.linalg.inv(info)
    return cov @ vec, cov


def fuse(dets: list[Detection], view_weight: dict[str, float] | None = None) -> list[dict]:
    """Merge same-alliance detections of one time step across views (greedy by conf)."""
    if FUSION == "cov":
        return fuse_cov(dets)
    out = []
    for al in ("red", "blue"):
        cand = [d for d in dets if d.alliance == al and d.xy is not None]
        cand.sort(key=lambda d: -d.conf)
        used = [False] * len(cand)
        for i, d in enumerate(cand):
            if used[i]:
                continue
            group = [d]
            used[i] = True
            for j in range(i + 1, len(cand)):
                if not used[j] and cand[j].view != d.view and \
                        np.hypot(cand[j].xy[0] - d.xy[0], cand[j].xy[1] - d.xy[1]) < FUSE_DIST:
                    group.append(cand[j])
                    used[j] = True
            w = np.array([g.conf * (view_weight or {}).get(g.view, 1.0) for g in group])
            xy = (np.array([g.xy for g in group]) * w[:, None]).sum(0) / w.sum()
            out.append({"alliance": al, "xy": xy, "dets": group, "conf": float(max(g.conf for g in group))})
    return out


@dataclass
class Slot:
    alliance: str
    idx: int
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))  # x, y, vx, vy
    P: np.ndarray = field(default_factory=lambda: np.eye(4))
    last_t: float = -1e9
    alive: bool = False
    seg: int = -1  # current segment id


@dataclass
class Segment:
    id: int
    alliance: str
    slot: int
    t0: float
    t1: float
    points: list = field(default_factory=list)  # (t, x, y)
    dets: list = field(default_factory=list)  # Detection objects (for OCR / appearance)


class FieldTracker:
    def __init__(self, q: float | None = None, r: float = 0.25):
        self.slots = {al: [Slot(al, i) for i in range(3)] for al in ("red", "blue")}
        self.segments: list[Segment] = []
        self.q, self.r = (PROCESS_Q if q is None else q), r
        self.t = None
        self.close_pairs: set = set()
        self.dup_count: dict = {}

    # ---- Kalman helpers (constant velocity, field frame)
    def _predict(self, s: Slot, dt: float):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        s.x = F @ s.x
        s.P = F @ s.P @ F.T + G @ G.T * self.q
        # cap speed
        v = np.hypot(s.x[2], s.x[3])
        if v > MAX_SPEED:
            s.x[2:] *= MAX_SPEED / v

    def _update(self, s: Slot, z: np.ndarray, r):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        R = np.asarray(r) if np.ndim(r) == 2 else np.eye(2) * r
        S = H @ s.P @ H.T + R
        K = s.P @ H.T @ np.linalg.inv(S)
        s.x = s.x + K @ (z - H @ s.x)
        s.P = (np.eye(4) - K @ H) @ s.P

    def _new_segment(self, s: Slot, t: float) -> Segment:
        seg = Segment(len(self.segments), s.alliance, s.idx, t, t)
        self.segments.append(seg)
        s.seg = seg.id
        return seg

    def step(self, t: float, obs: list[dict]):
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        for al in ("red", "blue"):
            slots = self.slots[al]
            for s in slots:
                if s.alive:
                    self._predict(s, dt)
                    if t - s.last_t > LOST_AFTER:
                        s.alive = False
            cand = [o for o in obs if o["alliance"] == al]
            live = [s for s in slots if s.alive]
            cost = np.full((len(live), len(cand)), 1e6)
            for i, s in enumerate(live):
                gap = t - s.last_t
                gate = GATE + MAX_SPEED * max(0.0, gap - dt)
                for j, o in enumerate(cand):
                    diff = o["xy"] - s.x[:2]
                    d = np.hypot(*diff)
                    if d >= gate:
                        continue
                    if "cov" in o:
                        # statistical distance: an imprecise end-camera fix may sit further away
                        S = s.P[:2, :2] + o["cov"] + np.eye(2) * 0.05
                        m2 = float(diff @ np.linalg.solve(S, diff))
                        if m2 < TRACK_CHI2 or d < GATE_NEAR:
                            cost[i, j] = min(m2, d * d / 0.25)
                    else:
                        cost[i, j] = d
            matched_obs = set()
            if live and cand:
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] < 1e6:
                        s = live[i]
                        r = cand[j]["cov"] + np.eye(2) * 0.01 if "cov" in cand[j] else self.r
                        self._update(s, cand[j]["xy"], r)
                        s.last_t = t
                        seg = self.segments[s.seg]
                        seg.t1 = t
                        seg.points.append((t, float(s.x[0]), float(s.x[1])))
                        seg.dets.extend(cand[j]["dets"])
                        matched_obs.add(j)
            # unmatched observations re-acquire dead slots (new segment = identity unknown)
            for j, o in enumerate(cand):
                if j in matched_obs:
                    continue
                dead = [s for s in slots if not s.alive]
                if not dead:
                    continue
                # prefer the dead slot whose last position is closest (still a new segment)
                s = min(dead, key=lambda s: np.hypot(*(o["xy"] - s.x[:2])) if s.last_t > -1e8 else 1e3)
                s.x = np.array([o["xy"][0], o["xy"][1], 0.0, 0.0])
                s.P = np.diag([0.3, 0.3, 4.0, 4.0])
                s.alive = True
                s.last_t = t
                seg = self._new_segment(s, t)
                seg.points.append((t, float(o["xy"][0]), float(o["xy"][1])))
                seg.dets.extend(o["dets"])
            # duplicates: bumpers make < DUP_DIST centre distance physically impossible, so two
            # slots that stay this close are one robot seen twice -> drop the younger slot
            live = [s for s in slots if s.alive]
            for a in range(len(live)):
                for b in range(a + 1, len(live)):
                    key = (al, live[a].idx, live[b].idx)
                    if np.hypot(*(live[a].x[:2] - live[b].x[:2])) < DUP_DIST:
                        self.dup_count[key] = self.dup_count.get(key, 0) + 1
                        if self.dup_count[key] >= DUP_STEPS:
                            sa, sb = self.segments[live[a].seg], self.segments[live[b].seg]
                            young = live[a] if sa.t0 > sb.t0 else live[b]
                            young.alive = False
                            self.dup_count[key] = 0
                    else:
                        self.dup_count[key] = 0
            # possible swaps: when two live slots come close (and again when they separate)
            # start new segments, isolating the ambiguous stretch
            live = [s for s in slots if s.alive]
            for a in range(len(live)):
                for b in range(a + 1, len(live)):
                    key = (al, live[a].idx, live[b].idx)
                    close = np.hypot(*(live[a].x[:2] - live[b].x[:2])) < SWAP_DIST
                    if close != (key in self.close_pairs):
                        if close:
                            self.close_pairs.add(key)
                        else:
                            self.close_pairs.discard(key)
                        for s in (live[a], live[b]):
                            self._new_segment(s, t)

    def finish(self) -> list[Segment]:
        return [s for s in self.segments if s.points]
