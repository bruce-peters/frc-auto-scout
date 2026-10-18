"""Bumper-only robot detection (PLAN.md 11, Colin's suggestion from the team's Slack thread).

A YOLO model boxes the *visible bumpers* (classes red_bumper / blue_bumper) instead of whole
robots, so it never has to guess the hidden part of a robot behind the hub, a referee or the
polycarbonate wall. Bumpers sit on the carpet, so even a partly hidden bumper gives a floor
point, and the class gives the alliance.

One robot can show several bumper boxes in one view (one per visible face, or a face split by
an intake gap). They are grouped into one observation per robot per view before fusion
(`group_sides`): same colour, touching or overlapping in the image, and together no wider on
the floor than one robot.

Each group becomes an ordinary `robots.Detection` whose box is a robot box synthesised above
the bumpers (bottom edge and width = the bumpers', height = a robot's height at that floor
point through the calibrated camera), so the rest of the pipeline (position from the box's
bottom-centre, shot tracing from the top of the robot, appearance crops, hopper fill) runs
unchanged. The bumper union is kept as `Detection.bumper_box`; OCR reads that crop.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from . import identity, robots

ROBOT_HEIGHT = 0.75  # m: height of the synthesised robot box above the bumper's floor point
GROUP_MAX_M = 1.5  # m: a robot's visible bumpers span at most ~its diagonal on the floor
GROUP_GAP = 0.6  # touching sides: horizontal gap <= this x the taller box's height (+3 px)
GROUP_MAX_PX = 9.0  # without a calibrated camera: group width <= this x the bumper height
DUP_IOU = 0.5  # two boxes of the same bumper (duplicate detections)


class BumperDetector:
    """YOLO bumper model. Returns [(box xyxy full-frame px, conf, 'red' | 'blue')]."""

    def __init__(self, path: str, imgsz: int, conf: float):
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.imgsz = imgsz
        self.conf = conf
        names = {int(k): v.lower() for k, v in self.model.names.items()}
        self.cls_alliance = {k: ("red" if "red" in v else "blue" if "blue" in v else None)
                             for k, v in names.items()}

    def __call__(self, frame: np.ndarray) -> list[tuple[tuple, float, str]]:
        # class-agnostic NMS: one bumper is never both colours
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False, half=True,
                               agnostic_nms=True, max_det=100)[0]
        out = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int)):
            al = self.cls_alliance.get(int(k))
            if al is not None:
                out.append((tuple(float(x) for x in b), float(c), al))
        return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _union(boxes) -> tuple[float, float, float, float]:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())


def group_sides(dets: list[tuple[tuple, float, str]],
                floor_width: Callable[[float, float, float], float | None] | None = None) -> list[list[int]]:
    """Group one view's bumper boxes into robots. Returns lists of indices into `dets`.

    Two same-colour boxes are joined when they are duplicates (IoU > DUP_IOU) or adjacent sides:
    horizontal gap <= GROUP_GAP x the taller box's height and overlapping rows. A join is only
    made if the merged group still fits one robot: `floor_width(x0, x1, y)` (metres between the
    floor points under (x0, y) and (x1, y), or None) <= GROUP_MAX_M, else pixel width <=
    GROUP_MAX_PX x the bumper height. Closest pairs are joined first.
    """
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(n):
        a, _, ca = dets[i]
        ha = a[3] - a[1]
        for j in range(i + 1, n):
            b, _, cb = dets[j]
            if ca != cb:
                continue
            hb = b[3] - b[1]
            if _iou(a, b) > DUP_IOU:
                pairs.append((-1.0, i, j))
                continue
            gap = max(a[0], b[0]) - min(a[2], b[2])  # < 0: the boxes overlap horizontally
            rows = min(a[3], b[3]) - max(a[1], b[1])  # > 0: they share image rows
            if gap <= GROUP_GAP * max(ha, hb) + 3 and rows > 0.25 * min(ha, hb):
                pairs.append((gap / max(ha, hb, 1.0), i, j))
    pairs.sort()
    members = {i: [i] for i in range(n)}
    for score, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        merged = members[ri] + members[rj]
        x0, y0, x1, y1 = _union([dets[k][0] for k in merged])
        if score > -1.0:  # adjacent sides (duplicates always merge): must still be one robot
            wm = floor_width(x0, x1, y1) if floor_width is not None else None
            if wm is not None:
                if wm > GROUP_MAX_M:
                    continue
            elif x1 - x0 > GROUP_MAX_PX * max(dets[k][0][3] - dets[k][0][1] for k in merged):
                continue
        parent[rj] = ri
        members[ri] = merged
        del members[rj]
    return [sorted(m) for m in members.values()]


def robot_box(bumper: tuple, cam, ox: float, oy: float, view_top: float) -> tuple[float, float, float, float]:
    """Whole-robot box (full-frame px) standing on a bumper group's bottom edge."""
    x0, y0, x1, y1 = bumper
    bw, bh = x1 - x0, y1 - y0
    h = None
    if cam is not None:
        u, v = (x0 + x1) / 2 - ox, y1 - oy
        xy = cam.backproject(np.array([[u, v]]), z=0.03)[0]
        if np.all(np.isfinite(xy)):
            top = cam.project(np.array([[xy[0], xy[1], ROBOT_HEIGHT]]))[0]
            if np.all(np.isfinite(top)) and top[1] < v:
                h = v - top[1]
    if h is None:
        h = max(0.9 * bw, 2.5 * bh)
    h = float(np.clip(h, 1.5 * bh, max(1.5 * bh, 3.0 * bw)))
    return x0, max(view_top, y1 - h), x1, y1


def ocr_crop(frame: np.ndarray, bumper: tuple) -> np.ndarray:
    """The bumper itself, slightly padded (digits fill most of its height)."""
    x0, y0, x1, y1 = bumper
    w, h = x1 - x0, y1 - y0
    H, W = frame.shape[:2]
    return frame[max(0, int(y0 - 0.2 * h)):min(H, int(y1 + 0.1 * h) + 1),
                 max(0, int(x0 - 0.05 * w)):min(W, int(x1 + 0.05 * w) + 1)]


def scan_frame(md, frame: np.ndarray, t: float, det: BumperDetector, crop_id: int, take_crops: bool,
               yellow: np.ndarray | None = None, end_view_range: float = 7.5) -> int:
    """Detect bumpers in one sampled frame, group them per robot and view, and append
    `robots.Detection`s (plus OCR / appearance crops) to md.dets / md.crops like the
    whole-robot path of pipeline.scan_video. Returns the next free crop id."""
    from . import views
    from .pipeline import _view_of

    vlist = views.PRODUCTIONS[md.prod]
    H, W = frame.shape[:2]
    by_view: dict[str, list] = {}
    vobj = {}
    for box, conf, al in det(frame):
        v = _view_of(box, vlist, W, H)
        if v is None:
            continue
        by_view.setdefault(v.name, []).append((box, conf, al))
        vobj[v.name] = v
    for vname, bl in by_view.items():
        v = vobj[vname]
        ox, oy = v.offset(W, H)
        vy0 = v.scaled(W, H)[1]
        cam_track = md.cams.get(vname)
        cam = cam_track.at(t) if cam_track is not None else None

        def floor_width(x0, x1, y, cam=cam, ox=ox, oy=oy):
            # trusted only where the floor point lands on the field: elsewhere the pose is off
            # (e.g. Chezy's 4-tag main camera) and footprints come out metres wide
            if cam is None:
                return None
            p = cam.backproject(np.array([[x0 - ox, y - oy], [x1 - ox, y - oy], [(x0 + x1) / 2 - ox, y - oy]]), z=0.03)
            if not np.all(np.isfinite(p)) or not robots.in_field(p[2]):
                return None
            return float(np.hypot(*(p[1] - p[0])))

        for g in group_sides(bl, floor_width):
            bumper = _union([bl[k][0] for k in g])
            conf = max(bl[k][1] for k in g)
            al = bl[g[0]][2]
            box = robot_box(bumper, cam, ox, oy, vy0)
            d = robots.Detection(t, vname, box, conf, al)
            d.bumper_box = bumper
            d.n_sides = len(g)
            x0, y0, x1, y1 = [int(round(c)) for c in box]
            crop = frame[max(0, y0):y1, max(0, x0):x1]
            if yellow is not None:
                ym = yellow[max(0, y0):y1, max(0, x0):x1]
                d.fill = float(ym.mean()) if ym.size else None
            if cam is not None:
                xy = cam.backproject(np.array([[(box[0] + box[2]) / 2 - ox, box[3] - oy]]), z=0.03)[0]
                if np.all(np.isfinite(xy)) and robots.in_field(xy):
                    far = np.hypot(*(xy - cam.center_world()[:2]))
                    if vname == "main" or far < end_view_range:
                        d.xy = (float(xy[0]), float(xy[1]))
            bh = y1 - y0
            if take_crops and bh >= 28:
                d.crop_id = crop_id
                md.crops[crop_id] = (ocr_crop(frame, bumper), identity.appearance_vector(crop), bh,
                                     cv2.resize(crop, (96, 96)) if crop.size else None)
                crop_id += 1
            md.dets.append(d)
    return crop_id
