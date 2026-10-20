"""End-to-end processing of one match video."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import attribution, config, field as fieldmod, fuel, identity, maincal, ocr, overlay, robots, tba, views, vision_count
from .video import Video, download

FIRST_AUTO_START = 6.13  # FIRST's 215 s uploads start the match ~6.1 s in
FIRST_TELEOP_GAP = 25.1
END_VIEW_RANGE = 7.5  # m: use end-camera robot positions only this close to the camera
ROBOT_HALF_DEPTH = 0.40  # m: box bottom-centre -> robot centre, along the camera ray
END_VIEW_RANGE_COV = 12.0  # m: with covariance fusion, far end-camera fixes are just down-weighted
POS_SIGMA_RAY = 0.30  # m: robot-centre uncertainty along the line of sight
POS_SIGMA_FLOOR = 0.08  # m
# Occlusion: between sightings a hidden robot is placed by interpolation (it cannot teleport),
# so its zone occupancy stays known and arcs rising above a wall can still be traced to it.
FILL_GAPS = True
GAP_FILL_MAX = 15.0  # s: interpolate across gaps up to this long
GAP_HOLD = 4.0  # s: hold the first/last sighting this long before/after
VIS_WINDOW = 0.3  # s: "seen" if any camera fixed the robot within this of the sample time
FILLED_SHOT_WEIGHT = 0.7  # confidence factor for arcs traced to an interpolated position
# Identity: a track segment keeps one team for its whole life, so a label decided at its start (while a
# duplicate track or an uncut swap briefly held the right team) overruled long runs of bumper reads.
# Cutting segments into <= SEG_MAX s pieces lets the Viterbi re-decide mid-track at switch_cost
# (failure-finder cycle 4: 53 v2 matches, held-out identity 86.5 -> 88.2 %, share corr vs OPR
# 0.659 -> 0.697). Re-measured with real bumper reads on the 38-match detector A/B (v2 scans): held-out
# identity 87.3 -> 91.4 %, vs 5026's human scouting (51 team-matches) fuel MAE 47.2 -> 43.5, corr
# 0.794 -> 0.824; share corr vs fuel OPR flat (0.765/0.737 -> 0.768/0.720 train/test). 0 = off.
SEG_MAX = 10.0


POST_MATCH_S = 15.0  # counter readings later than this after the end of teleop are not the match


def _match_counts(tl, al: str, hub: dict | None) -> list:
    """The alliance's scoreboard readings, without phantom reads after the match: readings more than
    POST_MATCH_S after teleop ends are dropped, and if the counter still ends > 20 % above TBA's total, a
    trailing reading that jumps past it is dropped too (2026cc_qm12 blue read 902 once, 102 s after the
    match, vs a real 219: one 683-ball increment that was then split evenly and scaled the real ones away)."""
    counts = list(tl.counts.get(al) or [])
    if tl.teleop_start is not None:
        end = tl.teleop_start + config.TELEOP_LEN + POST_MATCH_S
        kept = [(t, c) for t, c in counts if t <= end]
        # if most readings fall "after the match" the match clock is wrong (2026cc_qm14: start not found in a
        # long video), not the readings: keep them rather than zeroing the alliance
        if len(kept) >= 0.8 * len(counts):
            counts = kept
    total = (hub or {}).get("totalCount")
    if total and counts and max(c for _, c in counts) > 1.2 * total:
        counts = [(t, c) for t, c in counts if c <= 1.2 * total]
    return counts


# No hindsight: a match may only use what a live system would have when it is played, i.e. nothing from the
# event's later matches. The credit prior's fuel OPR comes from the quals played before this match ({} before
# the first), not the whole event. False: the whole event's quals (after-the-fact analysis only).
NO_HINDSIGHT = True
# ... and with no hindsight, shrink that in-event OPR toward each team's fuel OPR at its earlier events of the
# season (autoscout.season.prior_oprs; data/exp_season/RESULTS_section.md), which also gives a prior at qm1.
# Unused by default since 2026-10-18: per-team fuel must come from the match's own video only (Bruce), so the
# deployed credit model has no OPR prior (g_opr 0) and prior_opr is never called.
SEASON_PRIOR = True
_EVENT_OPR: dict[str, dict] = {}


def prior_opr(info) -> dict:
    """{team: fuel per match} behind the credit model's match-level prior for this match ({} if unavailable)."""
    live = NO_HINDSIGHT
    k = (info.key if live else info.event_key) + (":season" if live and SEASON_PRIOR else "")
    if k not in _EVENT_OPR:
        try:
            if live and SEASON_PRIOR:
                from . import season

                _EVENT_OPR[k] = season.prior_oprs(info.event_key, before=info.key)
            else:
                from . import opr as oprmod

                _EVENT_OPR[k] = oprmod.fuel_opr(info.event_key, before=info.key if live else None)
        except Exception as e:  # never lose a match to the prior
            print(f"{info.key}: fuel OPR unavailable for the credit prior ({e!r})")
            _EVENT_OPR[k] = {}
    return _EVENT_OPR[k]


def _attr_opr(info) -> dict | None:
    """prior_opr for the learned credit model (attribution.ATTR_MODEL); None when no model or a model without
    that prior is in use."""
    m = attribution.load_model()
    if not m or not m.get("g_opr"):
        return None
    return prior_opr(info)


def _split_segments(segs: list, max_len: float) -> list:
    """Cut every segment longer than max_len s into equal pieces of at most max_len s (same slot);
    points and detections are partitioned by time and ids renumbered."""
    out = []
    for sg in segs:
        if sg.t1 - sg.t0 <= max_len:
            out.append(sg)
            continue
        n = int(np.ceil((sg.t1 - sg.t0) / max_len))
        edges = sg.t0 + (sg.t1 - sg.t0) * np.arange(n + 1) / n
        for a, b in zip(edges[:-1], edges[1:]):
            last = b >= sg.t1 - 1e-9
            pts = [p for p in sg.points if a <= p[0] < b or (last and p[0] == b)]
            dts = [d for d in sg.dets if a <= d.t < b or (last and d.t == b)]
            if pts:
                out.append(robots.Segment(0, sg.alliance, sg.slot, pts[0][0], pts[-1][0], pts, dts))
    for i, sg in enumerate(out):
        sg.id = i
    return out


@dataclass
class Options:
    robot_model: str = str(config.MODELS / "robots.pt")
    fuel_model: str = str(config.MODELS / "fuel.pt")  # optional: ground-fuel map
    fuel_map_hz: float = 1.0
    det_hz: float = 10.0
    det_imgsz: int = 1280
    det_conf: float = 0.30
    fuel_k: int = 2  # frame offset (in source frames) for three-frame differencing
    fuel_step: int = 2  # analyse fuel on every 2nd frame (30 Hz at 60 fps)
    # flying-fuel YOLO (FUEL in the air): an extra blob source for flight tracking, merged with
    # the motion blobs. None = motion only.
    flying_fuel_model: str | None = None
    flying_fuel_conf: float = 0.25
    flying_fuel_imgsz: int = 1920
    # flight tracker: "greedy" (FlightTracker) or "ballistic" (fuel.BallisticTracker: separates the
    # balls of a stream; scripts/stream_experiment.py)
    flight_tracker: str = "greedy"
    # bumper-only robot detector (autoscout/bumpers.py; red_bumper / blue_bumper classes) used
    # instead of robot_model when set. None = whole-robot boxes.
    bumper_model: str | None = None
    bumper_conf: float = 0.30
    # bumper reads (scripts/ocr_ab.py, 6 v2 re-scans: held-out identity 88.9 -> 90.8 % vs 12/seg, 700, 64 px)
    ocr_max_per_seg: int = 12  # per track segment: at least this many ...
    ocr_per_s: float = 0.5     # ... or one per 2 s of a long segment ...
    ocr_seg_cap: int = 40      # ... up to this many
    ocr_budget: int = 2000
    ocr_all_cached: bool = True  # also use every read already cached for a segment's crops
    ocr_holdout: int | None = None  # evaluation: ignore crops with id % 2 == this for identity
    cnn_appearance: bool = True  # ResNet-18 crop embeddings (when thumbnails were stored)
    w_ocr: float = 2.0  # identity evidence weights
    w_app: float = 3.0
    switch_cost: float = 0.6
    dino_appearance: bool = False  # DINOv2 crop embeddings (dino.npz): neutral in tests, only 5 matches have them
    w_cls: float = 0.0  # per-match appearance classifier weight; off: it lowered held-out identity (PLAN.md 11)
    cls_l2: float = 1.0
    cls_min_h: int = 0  # only crops at least this tall (px) train/score the classifier
    galleries: dict | None = None  # {team: unit appearance vector} from other matches
    w_gal: float = 6.0
    # cross-match DINOv2 probe evidence (load_dino_probe, scripts/build_dino_probe.py); 0 = off. A match without
    # a probe file gets no DINO evidence. Only helps together with the absent state below
    # (data/exp_dino/RESULTS_section.md: held-out identity 91.2 -> 97.4 % with both, 91.7 % with the probe alone)
    w_dino: float = 2.0
    # Absent/unknown identity state (identity.assign_absent; data/exp_identity/RESULTS_section.md). None = the
    # 6-permutation model (one segment per team at a time). absent_cost_dead: per-epoch cost of leaving a dead-flagged team
    # (identity.dead_teams: never read while its labelled segments carry teammates' reads) without any segment,
    # so a duplicate track can share a teammate's label; absent_cost: the same for every other team. A team gets
    # no PRIOR/HIDDEN credit in its absent stretches. absent_near_bonus (per epoch) favours one label for two
    # concurrent segments < absent_near_m apart. absent_dead_hidden_s: a dead-flagged team absent at least this
    # many s in total gets no HIDDEN credit all match (None = off).
    absent_cost_dead: float | None = 2.0
    absent_cost: float | None = 2.0
    absent_near_m: float = 1.5
    absent_near_bonus: float = 2.0
    absent_dead_hidden_s: float | None = None
    # Parked robot: median movement extent (m) of its observed positions over 20 s windows below this ->
    # no PRIOR credit all match (0 = off; data/exp_identity/RESULTS_section.md: 5 non-shooter alliances better,
    # 0 worse vs scouting; e.g. 2026cc_qm23 4499, "did not move at all", 0.18 -> 0.07 share)
    park_ext: float = 1.0
    overlay_hz: float = 3.0
    debug: bool = False
    out_dir: Path = config.OUTPUT


class RobotDetector:
    def __init__(self, path: str, imgsz: int, conf: float):
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.imgsz = imgsz
        self.conf = conf
        names = {int(k): v.lower() for k, v in self.model.names.items()}
        self.cls_alliance = {k: ("red" if "red" in v else "blue" if "blue" in v else None)
                             for k, v in names.items()}

    def __call__(self, frame: np.ndarray) -> list[tuple[tuple, float, str | None]]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False, half=True)[0]
        out = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int)):
            out.append((tuple(float(x) for x in b), float(c), self.cls_alliance.get(int(k))))
        return out


class FlyingFuelDetector:
    """YOLO for FUEL in the air. Returns rows (x0, y0, x1, y1, conf) in full-frame pixels."""

    def __init__(self, path: str, imgsz: int, conf: float):
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False, quantize=16, max_det=600)[0]
        if not len(r.boxes):
            return np.zeros((0, 5), np.float32)
        return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)


class FuelScan:
    """Fuel step of the scan: three-frame differencing at 1/fuel_step of the source frame rate
    (plus the optional flying-fuel detector), flight tracking and hub-region activity, per view.

    Call `push` for every decoded frame, then `finish`. Used by `scan_video` and by
    `rescan_fuel` (fuel only, reusing a cached scan's robot detections).
    """

    def __init__(self, md: "MatchData", W: int, H: int, fps: float, opts: Options, keep_blobs: bool = False):
        self.md, self.W, self.H, self.fps = md, W, H, fps
        self.vlist = views.PRODUCTIONS[md.prod]
        # fuel_step / fuel_k are in 60 fps frames: scale them to the video's own rate, so a 30 fps broadcast is still
        # analysed at 30 Hz (every frame) instead of 15 Hz
        scale = (fps / 60.0) if fps and fps > 1 else 1.0
        self.step = max(1, round(opts.fuel_step * scale))
        self.kb = max(1, round(opts.fuel_k * scale) // self.step)  # neighbour offset in buffered (sub-sampled) frames
        self.buf = deque(maxlen=2 * self.kb + 1)  # (idx, frame, gray, yellow) of sub-sampled frames
        gate = 45.0 * (self.step / scale) ** 0.7  # the tracker gate grows with the time between analysed frames
        self.trackers = {v.name: (fuel.BallisticTracker() if opts.flight_tracker == "ballistic"
                                  else fuel.FlightTracker(gate=gate)) for v in self.vlist}
        self.valid = {}
        for v in self.vlist:
            x0, y0, x1, y1 = v.scaled(W, H)
            m = np.ones((y1 - y0, x1 - x0), np.uint8)
            for mx0, my0, mx1, my1 in v.mask_boxes(W, H):
                m[max(0, my0):max(0, my1), max(0, mx0):max(0, mx1)] = 0
            self.valid[v.name] = m
        # counting regions around each hub (view-local masks) for the vision-only counter;
        # recomputed per calibrated pose because cameras drift
        self.region_cache: dict = {}
        self.det = None
        self.det_conf = opts.flying_fuel_conf
        if opts.flying_fuel_model:
            self.det = FlyingFuelDetector(opts.flying_fuel_model, opts.flying_fuel_imgsz, opts.flying_fuel_conf)
        md.flying_dets = {v.name: [] for v in self.vlist} if self.det is not None else None
        # every centre frame's motion blobs and detections, for offline re-tracking experiments
        self.blob_log = {v.name: [] for v in self.vlist} if keep_blobs else None

    def regions_at(self, vname: str, t: float) -> dict:
        cam = self.md.cams.get(vname)
        if cam is None:
            return {}
        pi = cam.pose_index(t) if hasattr(cam, "pose_index") else 0
        key = (vname, pi)
        if key not in self.region_cache:
            v = next(x for x in self.vlist if x.name == vname)
            x0, y0, x1, y1 = v.scaled(self.W, self.H)
            out = {}
            c = cam.at(t)
            for al in ("red", "blue"):
                for kind in ("top", "exit"):
                    m = fieldmod.region_mask(c, fieldmod.hub_region_3d(al, kind), (y1 - y0, x1 - x0))
                    if m.any():
                        out[(vname, al, kind)] = m
                        self.md.hub_feats.setdefault((vname, al, kind), [])
            self.region_cache[key] = out
        return self.region_cache[key]

    def push(self, idx: int, frame: np.ndarray, start: int):
        if (idx - start) % self.step:
            return
        self.buf.append((idx, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fuel.yellow_mask(frame)))
        if len(self.buf) == 2 * self.kb + 1:
            self._centre()

    def _centre(self):
        md, W, H, kb = self.md, self.W, self.H, self.kb
        ci, cf, cg, cy = self.buf[kb]
        pg, ng = self.buf[0][2], self.buf[-1][2]
        tc = ci / self.fps
        dets = self.det(cf) if self.det is not None else None
        for v in self.vlist:
            x0, y0, x1, y1 = v.scaled(W, H)
            blobs = fuel.view_blobs(cg[y0:y1, x0:x1], pg[y0:y1, x0:x1], ng[y0:y1, x0:x1], cy[y0:y1, x0:x1],
                                    self.valid[v.name])
            vd = None
            if dets is not None:
                vd = fuel.view_detections(dets, (x0, y0, x1, y1), self.valid[v.name])
                md.flying_dets[v.name].extend((round(tc, 4), *[round(float(c), 1) for c in r[:4]], round(float(r[4]), 3))
                                              for r in vd)
            if self.blob_log is not None:
                self.blob_log[v.name].append((tc, np.array([(b.u, b.v, b.area, b.w, b.h, b.x0, b.y0) for b in blobs],
                                                           np.float64).reshape(-1, 7),
                                              np.zeros((0, 5), np.float32) if vd is None else vd))
            self.trackers[v.name].update(tc, fuel.merge_detections(blobs, vd, self.det_conf) if vd is not None
                                         else blobs)
            # hub-region activity from the motion blobs only (the vision counter was fitted on them)
            if blobs:
                bu = np.clip(np.array([b.u for b in blobs]).astype(int), 0, x1 - x0 - 1)
                bv = np.clip(np.array([b.v for b in blobs]).astype(int), 0, y1 - y0 - 1)
                ba = np.array([b.area for b in blobs])
            for (rv, al, kind), rm in self.regions_at(v.name, tc).items():
                if blobs:
                    inside = rm[bv, bu] > 0
                    md.hub_feats[(rv, al, kind)].append((tc, int(inside.sum()), float(ba[inside].sum())))
                else:
                    md.hub_feats[(rv, al, kind)].append((tc, 0, 0.0))

    def finish(self):
        for v in self.vlist:
            self.md.tracks_all[v.name] = self.trackers[v.name].flush()


def calibrate(video: Video, prod: str, t0: float, t1: float, step: float = 4.0, event: str | None = None,
              match: str | None = None, fallbacks: dict | None = None) -> dict:
    """Per view: shared intrinsics + a pose every `step` seconds (cameras pan / tilt).

    With `event`, every end-view fit is added to the event prior (field.update_calib_prior), and an end
    view that fails gets recovery candidates (field.recover_view) in `fallbacks`, to be checked against
    the scanned robots by accept_recovered_cams."""
    tags = fieldmod.load_tags()
    times = [float(t) for t in np.arange(t0, t1, step)]
    frames = []
    for t in times:
        f = video.frame_at(t)
        frames.append(f)
    keep = [(t, f) for t, f in zip(times, frames) if f is not None]
    times, frames = [k[0] for k in keep], [k[1] for k in keep]
    cams = {}
    for view in views.PRODUCTIONS[prod]:
        crops = [view.crop(f) for f in frames]
        masks = view.mask_boxes(video.width, video.height)
        cam = fieldmod.calibrate_view_dynamic(crops, times, tags, masks)
        if cam is not None and cam.rms < 4.0:
            cams[view.name] = cam
        if (view.name == "main" and MAIN_STATIC and fallbacks is not None and "main" in cams and crops
                and maincal.centre_scatter(cams["main"]) > maincal.MAIN_SCATTER_M):
            # the window poses disagree about where the camera is (Chezy): static candidates (field/maincal) and,
            # with an event prior, its rotation-only track (preferred); accept_main_cam decides after the scan
            size = crops[0].shape[1::-1]
            prior, trusted = maincal.load_main_prior(event, size, exclude=match)
            cands = maincal.main_candidates(maincal.tag_observations(crops, times, masks), tags, size, cams["main"],
                                            prior=prior)
            if event is not None and dict(cands).get("static") is not None:
                maincal.update_main_prior(event, match or "", dict(cands)["static"])
            fallbacks["main"] = (cands, trusted)
        if event is None or view.name == "main" or not crops:
            continue
        if view.name in cams:
            fieldmod.update_calib_prior(event, view.name, match or "", cam)
        elif fallbacks is not None:
            prior = fieldmod.load_calib_prior(event, view.name, crops[0].shape[1::-1], exclude=match)
            if prior is not None:
                fallbacks[view.name] = fieldmod.recover_view(crops, times, tags, masks, prior)
    return cams


# A recovered end camera is kept only if the view's robots, placed with it, land on the main view's robots.
CALIB_GATE_M = 1.0  # m: median distance to the nearest same-alliance main-view robot at the same instant
CALIB_GATE_MIN_N = 30  # boxes that must be compared


def cross_view_error(md: "MatchData", vname: str, cam) -> dict:
    """Place `vname`'s robot boxes with `cam` (as analyse() does: box bottom-centre on the floor, pushed
    half a robot along the ray, end-view range limit) and measure the distance to the nearest same-alliance
    main-view robot detected at the same instant. Returns median / q75 (m) and the number compared."""
    main = md.cams.get("main")
    if main is None or cam is None:
        return {"median": float("inf"), "q75": float("inf"), "n": 0}
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    max_rng = END_VIEW_RANGE_COV if robots.FUSION == "cov" else END_VIEW_RANGE

    def place(d, c, limit):
        ox, oy = vl[d.view].offset(1920, 1080)
        c = c.at(d.t)
        xy = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
        if not np.all(np.isfinite(xy)):
            return None
        dv = xy - c.center_world()[:2]
        rng = float(np.hypot(*dv))
        xy = xy + ROBOT_HALF_DEPTH * dv / max(rng, 1e-6)
        return xy if robots.in_field(xy) and (limit is None or rng <= limit) else None

    by_t: dict = {}
    for d in md.dets:
        if d.view in ("main", vname):
            by_t.setdefault(round(d.t, 3), []).append(d)
    dist = []
    for ds in by_t.values():
        ends = [d for d in ds if d.view == vname]
        if not ends:
            continue
        mains = [(d.alliance, p) for d in ds if d.view == "main" and (p := place(d, main, None)) is not None]
        for d in ends:
            p = place(d, cam, max_rng)
            q = [np.hypot(*(p - pm)) for al, pm in mains if al == d.alliance] if p is not None else []
            if q:
                dist.append(min(q))
    if not dist:
        return {"median": float("inf"), "q75": float("inf"), "n": 0}
    return {"median": float(np.median(dist)), "q75": float(np.quantile(dist, 0.75)), "n": len(dist)}


def accept_recovered_cams(md: "MatchData", candidates: dict) -> dict:
    """Gate for recovered end cameras ({view: [(how, CameraTrack)]} from field.recover_view): of the
    candidates whose robots agree with the main view (cross_view_error median < CALIB_GATE_M over at least
    CALIB_GATE_MIN_N boxes), the one that agrees best replaces md.cams[view]; otherwise the view stays
    uncalibrated (a wrong camera is worse than none). Returns {view: [per-candidate stats]}."""
    report = {}
    for vname, cands in candidates.items():
        rows, best = [], None
        for how, cam in cands:
            g = cross_view_error(md, vname, cam)
            ok = g["n"] >= CALIB_GATE_MIN_N and g["median"] < CALIB_GATE_M
            rows.append({"how": how, "rms": round(float(cam.rms), 3), "f": round(float(cam.K[0][0]), 1),
                         "poses": len(getattr(cam, "times", [0])), "median_m": round(g["median"], 3),
                         "q75_m": round(g["q75"], 3), "n": g["n"], "pass": ok})
            if ok and (best is None or g["median"] < best[0]):
                best = (g["median"], how, cam)
        if best is not None:
            best[2].recovered = best[1]  # marks it so it never feeds the event prior
            md.cams[vname] = best[2]
        report[vname] = {"accepted": None if best is None else best[1], "candidates": rows}
    return report


# Main camera (failure-finder cycle 8, data/exp_maincal/RESULTS_section.md): when the main view's window poses
# disagree about where the camera is (maincal.centre_scatter > 0.4 m: Chezy, no other event), calibrate() also
# fits a static camera (one centre and focal length per match, a rotation per window) and, with an event prior,
# its rotation-only track; after the scan accept_main_cam keeps one - or the stored camera with its bad window
# poses dropped - only if fewer main-view boxes fall off the field (the event track: not more) and the end
# views agree with it no worse. False = the plain per-window fit only.
MAIN_STATIC = True
MAIN_OFF_DROP = 0.002  # off-field share must drop by more than this
MAIN_XV_TOL = 0.02  # m: mean cross-view median (over the end views compared) may rise at most this much
MAIN_PREFER_RMS = 1.5  # px: tag rms a preferred candidate (accept_main_cam(prefer=...)) must stay under


def _main_boxes(md: "MatchData") -> np.ndarray:
    """(t, u, v) view pixels of the main-view robot box bottom-centres."""
    v = next((x for x in views.PRODUCTIONS[md.prod] if x.name == "main"), None)
    if v is None:
        return np.zeros((0, 3))
    ox, oy = v.offset(1920, 1080)
    return np.array([(d.t, (d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy) for d in md.dets if d.view == "main"],
                    float).reshape(-1, 3)


def accept_main_cam(md: "MatchData", candidates: list, prefer: str | None = None, trusted: bool = False) -> dict:
    """Gate for main cameras ([(how, CameraTrack)], e.g. maincal.main_candidates). Every candidate, and the
    current camera, is cleaned by maincal.guard (window poses putting >= 20 % of their own boxes off the field
    are dropped). The candidate with the fewest boxes off the field replaces md.cams['main'] if that share
    drops by more than MAIN_OFF_DROP and the end views' cross_view_error medians (mean over the views with
    enough boxes) rise by at most MAIN_XV_TOL; otherwise the current camera stays. report['flag'] is set when
    most window poses of the camera kept are bad. The report is also stored as md.main_calib.

    `prefer` names a candidate (the event camera's track, "event_prior") that wins whenever it passes: it may
    leave the off-field share as it is (within MAIN_OFF_DROP), must fit its tags (rms < MAIN_PREFER_RMS), must
    not be flagged, and - unless `trusted` (a camera checked against field landmarks, maincal.load_main_prior)
    - must keep the cross-view rule. Trusted, the end views are only reported: at Chezy the low end cameras
    share the main tags' far-side compression (data/exp_maincal/RESULTS_section.md)."""
    cur = md.cams.get("main")
    rep = {"accepted": None, "flag": False, "candidates": []}
    if cur is None:
        md.main_calib = rep
        return rep
    bx = _main_boxes(md)

    def stats(cam):
        off = float(maincal.off_field(cam, bx, ROBOT_HALF_DEPTH).mean()) if len(bx) else 0.0
        old, xv = md.cams.get("main"), {}
        md.cams["main"] = cam
        for v in ("blue_end", "red_end"):
            if v in md.cams:
                g = cross_view_error(md, v, md.cams[v])
                if g["n"] >= CALIB_GATE_MIN_N:
                    xv[v] = round(g["median"], 3)
        md.cams["main"] = old
        return off, xv

    off0, xv0 = stats(cur)
    _, g0 = maincal.guard(cur, bx, ROBOT_HALF_DEPTH)
    rep["current"] = {"off": round(off0, 4), "xview": xv0, **g0}
    best = None
    for how, cam in [("guard", cur)] + list(candidates):
        gcam, g = maincal.guard(cam, bx, ROBOT_HALF_DEPTH)
        if how == "guard" and not g["dropped"]:
            continue  # the current camera unchanged: nothing to compare
        off, xv = stats(gcam)
        both = [v for v in xv if v in xv0]
        dxv = float(np.mean([xv[v] - xv0[v] for v in both])) if both else 0.0
        ok = off < off0 - MAIN_OFF_DROP and dxv <= MAIN_XV_TOL
        if how == prefer:
            ok = (off <= off0 + MAIN_OFF_DROP and cam.rms < MAIN_PREFER_RMS and not g["flag"]
                  and (trusted or dxv <= MAIN_XV_TOL))
        rep["candidates"].append({"how": how, "off": round(off, 4), "xview": xv, "d_xview": round(dxv, 3),
                                  "rms": round(float(cam.rms), 3), "f": round(float(cam.K[0][0]), 1),
                                  "centre": np.round(fieldmod.track_centre(gcam), 3).tolist(), "pass": ok, **g})
        if ok and (best is None or (how == prefer, -off, -dxv) > (best[2] == prefer, -best[0], -best[1])):
            best = (off, dxv, how, gcam, g)
    if best is not None:
        best[3].main_fix = best[2]
        md.cams["main"] = best[3]
        rep["accepted"], rep["flag"] = best[2], best[4]["flag"]
    else:
        rep["flag"] = g0["flag"]
    md.main_calib = rep
    return rep


def _view_of(box, vlist: list[views.View], w: int, h: int) -> views.View | None:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for v in vlist:
        x0, y0, x1, y1 = v.scaled(w, h)
        if x0 <= cx < x1 and y0 <= cy < y1:
            # ignore detections inside overlay graphics
            ox, oy = v.offset(w, h)
            for mx0, my0, mx1, my1 in v.mask_boxes(w, h):
                if mx0 <= cx - ox < mx1 and my0 <= cy - oy < my1:
                    return None
            return v
    return None


@dataclass
class MatchData:
    key: str
    info: tba.MatchInfo
    prod: str
    cams: dict
    timeline: overlay.Timeline
    dets: list = field(default_factory=list)  # robots.Detection
    crops: dict = field(default_factory=dict)  # crop_id -> (bumper crop, body appearance vec)
    shots: dict = field(default_factory=dict)  # view -> list[FlightTrack]
    tracks_all: dict = field(default_factory=dict)  # view -> n tracks (debug)
    hub_feats: dict = field(default_factory=dict)  # (view, alliance, kind) -> [(t, n, area)]
    fuel_map: object = None  # field grid of fuel sightings (counts summed over samples)
    fuel_map_n: int = 0
    fuel_counts: list = field(default_factory=list)  # (t, balls visible on the field)
    flying_dets: dict | None = None  # view -> [(t, x0, y0, x1, y1, conf)] flying-fuel detections


def scan_video(video: Video, info: tba.MatchInfo, prod: str, cams: dict, timeline: overlay.Timeline,
               t0: float, t1: float, opts: Options) -> MatchData:
    md = MatchData(info.key, info, prod, cams, timeline)
    vlist = views.PRODUCTIONS[prod]
    W, H = video.width, video.height
    bdet = None
    if opts.bumper_model:
        from . import bumpers
        bdet = bumpers.BumperDetector(opts.bumper_model, opts.det_imgsz, opts.bumper_conf)
    else:
        det = RobotDetector(opts.robot_model, opts.det_imgsz, opts.det_conf)
    fuel_det = None
    if opts.fuel_model and Path(opts.fuel_model).exists() and cams.get("main") is not None:
        from ultralytics import YOLO
        fuel_det = YOLO(opts.fuel_model)
        md.fuel_map = np.zeros((int(config.FIELD_LENGTH / 0.25) + 1, int(config.FIELD_WIDTH / 0.25) + 1))
    fuel_every = max(1, int(round(video.fps / opts.fuel_map_hz)))
    main_view = next(v for v in vlist if v.name == "main")
    det_every = max(1, int(round(video.fps / opts.det_hz)))
    fscan = FuelScan(md, W, H, video.fps, opts)
    buf = fscan.buf
    crop_id = 0
    start, end = int(t0 * video.fps), int(t1 * video.fps)
    tic = time.time()
    for idx, frame in video.iter_frames(start, end):
        t = idx / video.fps
        # ---- fuel in flight (buffers this frame; processes the buffer's centre frame)
        fscan.push(idx, frame, start)
        # ---- robots
        if (idx - start) % det_every == 0 and bdet is not None:
            crop_id = bumpers.scan_frame(md, frame, t, bdet, crop_id, (idx // det_every) % 2 == 0,
                                         yellow=buf[-1][3] if buf and buf[-1][0] == idx else None,
                                         end_view_range=END_VIEW_RANGE)
        elif (idx - start) % det_every == 0:
            for box, conf, al_cls in det(frame):
                v = _view_of(box, vlist, W, H)
                if v is None:
                    continue
                x0, y0, x1, y1 = [int(round(c)) for c in box]
                crop = frame[max(0, y0):y1, max(0, x0):x1]
                al_col, col_conf = robots.bumper_alliance(crop)
                al = al_col if (al_col and col_conf > 0.75) else (al_cls or al_col)
                if al is None:
                    continue
                d = robots.Detection(t, v.name, box, conf, al)
                if buf and buf[-1][0] == idx:
                    ym = buf[-1][3][max(0, y0):y1, max(0, x0):x1]
                    d.fill = float(ym.mean()) if ym.size else None
                cam = cams.get(v.name)
                if cam is not None:
                    cam = cam.at(t)
                    ox, oy = v.offset(W, H)
                    xy = cam.backproject(np.array([[(box[0] + box[2]) / 2 - ox, box[3] - oy]]), z=0.03)[0]
                    if np.all(np.isfinite(xy)) and robots.in_field(xy):
                        # low end-cameras: positions of far robots are unreliable
                        far = np.hypot(*(xy - cam.center_world()[:2]))
                        if v.name == "main" or far < END_VIEW_RANGE:
                            d.xy = (float(xy[0]), float(xy[1]))
                bh = y1 - y0
                if bh >= 28 and (idx // det_every) % 2 == 0:
                    d.crop_id = crop_id
                    md.crops[crop_id] = (ocr.bumper_crop(frame, box), identity.appearance_vector(crop), bh,
                                         cv2.resize(crop, (96, 96)) if crop.size else None)
                    crop_id += 1
                md.dets.append(d)
        # ---- ground fuel positions (YOLO fuel model on the main view)
        if fuel_det is not None and (idx - start) % fuel_every == 0:
            crop = main_view.crop(frame)
            r = fuel_det.predict(crop, imgsz=1920, conf=0.3, verbose=False, half=True, max_det=1000)[0]
            if len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                uv = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
                xy = cams["main"].at(t).backproject(uv, z=0.075)
                ok = np.all(np.isfinite(xy), axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < config.FIELD_LENGTH) &                     (xy[:, 1] >= 0) & (xy[:, 1] < config.FIELD_WIDTH)
                gi = (xy[ok] / 0.25).astype(int)
                np.add.at(md.fuel_map, (gi[:, 0], gi[:, 1]), 1)
                md.fuel_counts.append((round(t, 2), int(ok.sum())))
            md.fuel_map_n += 1
        if opts.debug and (idx - start) % (int(video.fps) * 20) == 0:
            print(f"  t={t:6.1f}s  dets={len(md.dets)}  {time.time() - tic:5.0f}s", flush=True)
    fscan.finish()
    return md


def rescan_fuel(md: "MatchData", video: Video, opts: Options, keep_blobs: bool = False) -> "FuelScan":
    """Redo only the fuel step of a cached scan (robot detections, crops and OCR are kept).

    Replaces md.tracks_all / md.hub_feats / md.flying_dets in place and returns the FuelScan
    (its `blob_log` holds every centre frame's blobs when keep_blobs is set).
    """
    md.tracks_all, md.hub_feats = {}, {}
    fscan = FuelScan(md, video.width, video.height, video.fps, opts, keep_blobs=keep_blobs)
    start, end = int(md_t0(md) * video.fps), int(md_t1(md) * video.fps)
    for idx, frame in video.iter_frames(start, end):
        fscan.push(idx, frame, start)
    fscan.finish()
    return fscan


def _track_position(arr: np.ndarray | None, t: float):
    """(x, y, observed) of a team at video time t from its sorted (t, x, y) sightings.

    Observed if a sighting is within 0.6 s. Otherwise, inside a gap of <= GAP_FILL_MAX s the
    position is interpolated between the sightings either side, and within GAP_HOLD s of the
    first/last sighting it is held; beyond that the position is unknown (None).
    """
    if arr is None or len(arr) == 0:
        return None
    j = int(np.searchsorted(arr[:, 0], t))
    near = [arr[k] for k in (j - 1, j) if 0 <= k < len(arr) and abs(arr[k][0] - t) < 0.6]
    if near:
        c = min(near, key=lambda r: abs(r[0] - t))
        return float(c[1]), float(c[2]), True
    if 0 < j < len(arr):
        a, b = arr[j - 1], arr[j]
        if b[0] - a[0] <= GAP_FILL_MAX:
            f = (t - a[0]) / max(b[0] - a[0], 1e-6)
            return float(a[1] + f * (b[1] - a[1])), float(a[2] + f * (b[2] - a[2])), False
        return None
    if j == 0 and arr[0][0] - t <= GAP_HOLD:
        return float(arr[0][1]), float(arr[0][2]), False
    if j == len(arr) and t - arr[-1][0] <= GAP_HOLD:
        return float(arr[-1][1]), float(arr[-1][2]), False
    return None


def _px_per_m(cam: fieldmod.Camera | None, u: float, v: float) -> float:
    if cam is None:
        return 80.0
    xy = cam.backproject(np.array([[u, v]]), z=1.0)[0]
    if not np.all(np.isfinite(xy)):
        return 80.0
    p = cam.project(np.array([[xy[0], xy[1], 1.0], [xy[0] + 0.5, xy[1], 1.0], [xy[0], xy[1] + 0.5, 1.0]]))
    return float(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])) * 2)


# Launch-arc gate (from a frame-by-frame audit of 36 credited launches, 2026-10-17: 8 were not shots
# at all - balls rattling in a clear hopper, rolling or bouncing balls - and 2 were human-player
# throws credited to the robot below them). A real shot leaves from the top of the robot and climbs
# well clear of it. On 6 unseen matches ~83 % of the launches it keeps went to the robot that shot
# (vs ~47 % before); share corr vs fuel OPR 0.686/0.673 -> 0.690/0.688 (train/test), within noise.
SHOT_GATE = True
LAUNCH_ABOVE_MAX = 0.45  # first ball sighting at most this many box heights above the box top
SIDE_MARGIN = 0.2        # ... and horizontally over the robot (this fraction of its width either side)
EXIT_TOP = 0.06          # an arc reaching the top 6 % of its view counts as climbing clear
RISE_ABOVE_MIN = 0.6     # the arc must climb at least this many box heights above the box top
INBOX_MAX = 0.5          # at most this fraction of the arc inside the robot's own box
# Dense volleys: the greedy flight tracker chains one ball onto the next, so a stream becomes a few long zigzag
# tracks that fail the projectile tests (2026cc_qm24 70-74 s, main view: 0 of ~117 tracks near 254 were shots).
SPLIT_TRACKS = True      # cut stored tracks where they hop to another ball (fuel.split_at_jumps) before the tests
# ... and a split piece is first seen mid-air: gate on where the arc, traced back along its initial velocity,
# crossed the box top (within GATE_TRACE_S s) instead of on its first sighting. Both on since 2026-10-18 with the
# credit model refit on the resulting launches (data/exp_split/RESULTS_section.md): sole-shooter recall in 20+
# ball volleys ~0.2 -> ~0.7, and vs 5026's scouting nested-CV fuel MAE 41.9 -> 38.8 (paired CI -6.0 .. -0.04)
GATE_TRACE = True
GATE_TRACE_S = 0.3
# A ball launched while its robot stands outside its own alliance zone in teleop cannot score (G407): a pass or a
# feed. Such launches count LAUNCH_ZONE_WEIGHT as much (1 = off, 0 = dropped); "outside" = robot centre more than
# LAUNCH_ZONE_MARGIN m beyond its zone line (position noise). ~20 % of teleop launches at Chezy are. On since
# 2026-10-18 (Bruce's rule; vs scouting with the evidence-only credit model refit: fuel MAE -0.15, CI -0.7 .. +0.4,
# 35 of 54 matches better; it replaces most of the robot-level zone gate, gz 0.27 -> 0.10)
LAUNCH_ZONE_WEIGHT = 0.0
LAUNCH_ZONE_MARGIN = 1.0
# Stacked robots in one view (one box just below another, e.g. 3847 in front of 254 in the Chezy main camera): an
# arc traced back in time crosses the upper robot's top first and would carry on down to the lower one's. Credit
# the first box top it crosses (within SIDE_MARGIN of that box and GATE_TRACE_S s) instead of the best score.
LINK_FIRST_CROSS = False
LINK_FIRST_CROSS_VIEWS = ("main",)  # the high camera only: in the low end cameras a taller robot BESIDE the shooter
                                    # has the higher top, and the rule would hand it the ball
PROJ_RISE_M = 0.8        # arcs linked by projection (no box in this view) must climb this many metres
TOP_TIEBREAK = 0.5       # between overlapping boxes, prefer the one whose top edge the arc leaves
OVERLAP_ZONE_RULE = True  # ... and in teleop, between robots of both alliances, the one in its own zone
# ... and the box the arc rose out of: |u - box centre| / box width, u = where the arc, traced back, crossed that
# box's top edge (LINK_X_CROSS; else its first point). Side-by-side robots: 54 scouted quals, video-only fuel
# MAE 48.8 -> 46.9 (paired 95 % CI -3.0 .. -0.9), data/exp_audit/tools/link_x_ab.py
LINK_X_TIE = 1.0
LINK_X_CROSS = True
# Shooter's box lost for a moment (against the glass, close under an end camera, behind the hub; the visual QA,
# data/exp_visual_qa/report.md): keep each labelled segment's last box in a view as a link candidate for BOX_HOLD_S
# s after the detector drops it, BOX_HOLD_PEN box-heights behind a box seen in the launch frame
BOX_HOLD_S = 1.5
BOX_HOLD_PEN = 0.3
# Overlapping boxes (one robot in front of another, or side by side at the hub): prefer the robot whose hopper is
# emptying, i.e. whose box fill (yellow fraction) fell from the LINK_FILL_WIN s before the launch to the
# LINK_FILL_WIN s after; score -= LINK_FILL_TIE * drop (visual QA: right in 3 of 3 checked)
LINK_FILL_TIE = 3.0
LINK_FILL_WIN = 1.0
# arcs with no box in the view are linked by projection to the robot whose field position projects within 1 m of
# the launch point; a hidden robot's interpolated position is rougher, so allow it PROJ_HIDDEN_MAX_M
PROJ_HIDDEN_MAX_M = 1.0
# Process of elimination (Bruce). Tracks: two concurrent tracks of one alliance more than IDENT_FAR_M apart cannot
# be the same robot, so the identity Viterbi pays IDENT_FAR_COST per epoch to label them alike (2026cc_qm17: the
# DINO probe called 6800's track 6665 too, so 6800 was "absent" and its launches went to 6665). Boxes: a "?" box (no
# field position, so no track or label: off the calibrated floor at an image edge, far down an end camera) is the
# one team of its alliance not labelled on another box in the frame and not tracked elsewhere at that moment
# (tracked = projects more than ELIM_FAR box widths from the box). Adopted 2026-10-19 with BOX_HOLD_S / LINK_FILL_TIE
# and a refit credit model: 54 scouted Chezy quals, nested CV, fuel MAE 42.4 -> 41.2 (paired -1.2, 95 % CI -3.6 ..
# +0.7), top scorer 64 -> 69 %, the model's hidden-robot pseudo-shots 9.7 -> 6.2 (data/exp_elim/RESULTS_section.md)
IDENT_FAR_M = 2.5
IDENT_FAR_COST = 20.0
ELIM_BOXES = True
ELIM_FAR = 1.5
# ... and an arc with no box or tracked robot near it at all (a robot right under an end camera, behind the hub):
# launched inside an alliance zone, it is that alliance's one robot not tracked more than ELIM_ARC_M from the
# launch point (link confidence ELIM_ARC_WEIGHT)
ELIM_ARCS = True
ELIM_ARC_M = 2.0
ELIM_ARC_WEIGHT = 0.6


def _box_shot_ok(tr, box, view_h: float = 1e9) -> bool:
    """Geometry of the first *observed* ball position against the robot box (the back-extrapolated
    launch point slides down onto whatever is below a high throw, so it is not used here)."""
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    p = np.array([(q[1], q[2]) for q in tr.pts])
    u0, v0 = p[0]
    if GATE_TRACE and (y0 - v0) / bh > LAUNCH_ABOVE_MAX:
        vu, vv = tr.velocity0()
        tau = (y0 - v0) / vv if vv < -1e-6 else -np.inf  # < 0: back in time to the box top
        if tau < -GATE_TRACE_S:
            return False
        u0, v0 = u0 + vu * tau, y0
    if (y0 - v0) / bh > LAUNCH_ABOVE_MAX or not (x0 - SIDE_MARGIN * bw <= u0 <= x1 + SIDE_MARGIN * bw):
        return False
    # must climb clear of the robot - or leave the top of the view (a robot close to an end camera)
    if (y0 - p[:, 1].min()) / bh < RISE_ABOVE_MIN and p[:, 1].min() > EXIT_TOP * view_h:
        return False
    inbox = ((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)).mean()
    return bool(inbox <= INBOX_MAX)


def _proj_shot_ok(tr, ppm: float) -> bool:
    v = np.array([q[2] for q in tr.pts])
    return bool((v[0] - v.min()) / max(ppm, 1e-3) >= PROJ_RISE_M)


def _median_extent(pts: np.ndarray, win: float = 20.0, step: float = 10.0) -> float | None:
    """Median over windows of `win` s (every `step` s) of how far a robot's observed positions spread
    (bounding-box diagonal, m; windows with >= 3 fixes). None without any such window. In the scouted
    2026cc quals parked robots score <= 0.94 m and every other robot >= 3.9 m; a few unscouted robots at
    other events fall in between (data/exp_identity/RESULTS_section.md)."""
    if len(pts) < 3:
        return None
    ext = []
    for a in np.arange(pts[0, 0], pts[-1, 0] - win, step):
        m = (pts[:, 0] >= a) & (pts[:, 0] <= a + win)
        if m.sum() >= 3:
            q = pts[m, 1:3]
            ext.append(float(np.hypot(*(q.max(0) - q.min(0)))))
    return float(np.median(ext)) if ext else None


def analyse(md: MatchData, opts: Options) -> dict:
    info, tl = md.info, md.timeline
    W, H = 1920, 1080
    vlist = {v.name: v for v in views.PRODUCTIONS[md.prod]}

    # ---------------- robot field positions (recomputed here so cached scans get fixes)
    # A box's bottom-centre lies on the robot edge nearest that camera; push it half a robot
    # deeper along the viewing direction so different cameras agree on the robot centre.
    for d in md.dets:
        cam_track = md.cams.get(d.view)
        if cam_track is None:
            d.xy = None
            continue
        cam = cam_track.at(d.t)
        ox, oy = vlist[d.view].offset(W, H)
        u, v = (d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy
        pts = cam.backproject(np.array([[u, v], [u + 1.0, v], [u, v + 1.0]]), z=0.03)
        xy = pts[0]
        if not np.all(np.isfinite(pts)):
            d.xy = None
            continue
        c = cam.center_world()[:2]
        dv = xy - c
        rng = float(np.hypot(*dv))
        ray = dv / max(rng, 1e-6)
        xy = xy + ROBOT_HALF_DEPTH * ray
        max_rng = END_VIEW_RANGE_COV if robots.FUSION == "cov" else END_VIEW_RANGE
        if not robots.in_field(xy) or (d.view != "main" and rng > max_rng):
            d.xy = None
            continue
        d.xy = (float(xy[0]), float(xy[1]))
        # position uncertainty from the camera geometry: pixel noise of the box's bottom edge
        # through the back-projection Jacobian, plus where the robot centre sits behind its
        # visible edge (along the line of sight) and a small isotropic floor
        J = np.stack([pts[1] - pts[0], pts[2] - pts[0]], axis=1)  # metres per pixel
        bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
        S_px = np.diag([(0.10 * bw + 2.0) ** 2, (0.06 * bh + 2.0) ** 2])
        d.cov = (J @ S_px @ J.T + POS_SIGMA_RAY ** 2 * np.outer(ray, ray) + POS_SIGMA_FLOOR ** 2 * np.eye(2))

    # ---------------- robot tracking (10 Hz fused observations)
    tracker = robots.FieldTracker()
    by_t: dict[float, list] = {}
    for d in md.dets:
        by_t.setdefault(round(d.t, 3), []).append(d)
    for t in sorted(by_t):
        tracker.step(t, robots.fuse(by_t[t]))
    segs = tracker.finish()
    if SEG_MAX:
        segs = _split_segments(segs, SEG_MAX)
    det_seg = {}
    for s in segs:
        for d in s.dets:
            det_seg[id(d)] = s.id

    # ---------------- identity evidence
    seg_teams: dict[int, int] = {}
    team_of_seg: dict[int, int] = {}
    ident_debug = {}
    team_absent: dict[int, list] = {}  # absent state (Options.absent_cost*): team -> [(t0, t1)] absent stretches
    team_dead: dict[int, bool] = {}  # ... and its dead-robot flag
    # OCR budget: per segment, the largest bumper crops spread over time (max one per 0.5 s),
    # and at most opts.ocr_budget reads per match overall, biggest crops first
    ocr_reads: dict[int, list] = {}
    pool = []
    for s in segs:
        cands = [d for d in s.dets if d.crop_id is not None]
        cands.sort(key=lambda d: -md.crops[d.crop_id][2])
        taken = []
        cap = min(opts.ocr_seg_cap, max(opts.ocr_max_per_seg, int((s.t1 - s.t0) * opts.ocr_per_s)))
        for d in cands:
            if len(taken) >= cap:
                break
            if all(abs(d.t - x.t) >= 0.5 or d.view != x.view for x in taken):
                taken.append(d)
        pool += taken
    pool.sort(key=lambda d: -md.crops[d.crop_id][2])
    cache = getattr(md, "ocr_cache", None)
    if cache is None:
        cache = md.ocr_cache = {}
    for d in pool[:opts.ocr_budget]:
        if d.crop_id not in cache:
            img = _img(md.crops[d.crop_id][0])
            cache[d.crop_id] = ocr.read_numbers(img) if img is not None else []
        ocr_reads[d.crop_id] = cache[d.crop_id]
    if opts.ocr_all_cached:  # reads made for an earlier segmentation are free evidence too
        for s in segs:
            for d in s.dets:
                if d.crop_id is not None and d.crop_id in cache:
                    ocr_reads[d.crop_id] = cache[d.crop_id]
    # appearance per crop: DINOv2 embedding (scripts/build_dino.py) if available, else ResNet-18 of
    # the stored thumbnail, else colour histogram
    app_vec: dict[int, np.ndarray] = {}
    thumbs = {cid: _img(c[3]) for cid, c in md.crops.items() if len(c) > 3 and c[3] is not None}
    if opts.dino_appearance and getattr(md, "dino_cache", None):
        app_vec = md.dino_cache
    elif opts.cnn_appearance and (thumbs or getattr(md, "cnn_cache", None)):
        if getattr(md, "cnn_cache", None) is None:
            emb = identity.CNNEmbedder()
            ids = list(thumbs)
            feats = emb([thumbs[i] for i in ids])
            md.cnn_cache = dict(zip(ids, feats))
        app_vec = md.cnn_cache
    else:
        app_vec = {cid: c[1] for cid, c in md.crops.items() if c[1] is not None}
    for al in ("red", "blue"):
        teams = info.teams(al)
        asegs = [s for s in segs if s.alliance == al]
        ev = {}
        app = {}
        held = {}
        seg_feats: dict = {}  # view -> segment id -> crop features
        for s in asegs:
            e = np.zeros(3)
            h = np.zeros(3)
            for d in s.dets:
                if d.crop_id in ocr_reads:
                    ev_d = ocr.team_evidence(ocr_reads[d.crop_id], teams)
                    if opts.ocr_holdout is not None and d.crop_id % 2 == opts.ocr_holdout:
                        h += ev_d
                    else:
                        e += ev_d
            held[s.id] = h
            vecs = [app_vec[d.crop_id] for d in s.dets if d.crop_id in app_vec]
            if vecs:
                app[s.id] = np.mean(vecs, axis=0)
            for d in s.dets:
                if d.crop_id in app_vec and md.crops[d.crop_id][2] >= opts.cls_min_h:
                    seg_feats.setdefault(d.view, {}).setdefault(s.id, []).append(app_vec[d.crop_id])
            ev[s.id] = e
        seg_feats = {vw: {sid: np.asarray(f, dtype=np.float32) for sid, f in fs.items()}
                     for vw, fs in seg_feats.items()}
        # per-match, per-camera appearance classifier trained on segments whose bumper reads are clear
        cls_ev = identity.classifier_evidence_by_view(seg_feats, ev, l2=opts.cls_l2) if opts.w_cls > 0 else {}
        # cross-match DINOv2 probe: mean per-crop log q over the alliance's teams, centred, full weight from 5 crops
        dino_ev = {}
        if opts.w_dino > 0:
            if getattr(md, "_dino_lp", None) is None or md._dino_lp[0] != str(DINO_PROBE_DIR):
                md._dino_lp = (str(DINO_PROBE_DIR), load_dino_probe(md.key))
            lp = md._dino_lp[1]
            for s in asegs:
                rows = np.array([[lp[d.crop_id].get(t, np.nan) for t in teams] for d in s.dets if d.crop_id in lp],
                                dtype=float).reshape(-1, 3)
                rows = rows[np.isfinite(rows).all(1)]
                if len(rows):
                    m = rows.mean(0)
                    dino_ev[s.id] = (m - m.mean()) * min(1.0, len(rows) / 5.0)
        base_ev = {k: opts.w_ocr * v + opts.w_cls * cls_ev.get(k, 0.0) + opts.w_dino * dino_ev.get(k, 0.0)
                   for k, v in ev.items()}
        # absent state (Options.absent_cost_dead / absent_cost; both None = the 6-permutation model)
        use_absent = opts.absent_cost_dead is not None or opts.absent_cost is not None
        if use_absent:
            dead = identity.dead_teams(asegs, ev, identity.assign(asegs, ev, opts.switch_cost))
            a_cost = [(opts.absent_cost_dead if dead[ti] else opts.absent_cost) for ti in range(3)]
            a_cost = [np.inf if c is None else float(c) for c in a_cost]
            team_dead.update({teams[ti]: bool(dead[ti]) for ti in range(3)})

        def _assign(evd):
            if not use_absent:
                return identity.assign(asegs, evd, opts.switch_cost)
            lab, ab = identity.assign_absent(asegs, evd, opts.switch_cost, a_cost, opts.absent_near_m,
                                             opts.absent_near_bonus, IDENT_FAR_M, IDENT_FAR_COST)
            team_absent.update({teams[ti]: [(float(a), float(b)) for a, b in ab.get(ti, [])] for ti in range(3)})
            return lab
        # iterate: OCR(+classifier) labels -> appearance prototypes -> combined evidence
        labels = _assign(base_ev)
        for _ in range(2):
            protos = {}
            for ti in range(3):
                vs = [app[sid] * (1 + ev[sid][ti]) for sid, lab in labels.items() if lab == ti and sid in app]
                if vs:
                    protos[ti] = np.mean(vs, axis=0)
            comb = {}
            for s in asegs:
                e = base_ev[s.id].copy()
                if opts.galleries and s.id in app:
                    gs = np.array([float(np.dot(app[s.id], opts.galleries[t]) / (np.linalg.norm(app[s.id]) + 1e-9))
                                   if t in opts.galleries else np.nan for t in teams])
                    if np.isfinite(gs).sum() >= 2:
                        gs = np.where(np.isfinite(gs), gs, np.nanmean(gs))
                        e = e + opts.w_gal * (gs - gs.mean()) * min(1.0, (s.t1 - s.t0) / 3.0)
                if s.id in app and protos:
                    sims = np.array([float(np.dot(app[s.id], protos[ti]) / (np.linalg.norm(app[s.id]) *
                                     np.linalg.norm(protos[ti]) + 1e-9)) if ti in protos else 0.0
                                     for ti in range(3)])
                    e = e + opts.w_app * (sims - sims.mean())
                comb[s.id] = e
            labels = _assign(comb)
        for sid, ti in labels.items():
            team_of_seg[sid] = teams[ti]
        md.seg_app = getattr(md, "seg_app", {})
        md.seg_app.update({sid: (app[sid], ev[sid], teams) for sid in app})
        ident_debug[al] = {str(s.id): {"slot": s.slot, "t0": round(s.t0, 2), "t1": round(s.t1, 2),
                                       "ocr": ev[s.id].round(2).tolist(),
                                       "held": held[s.id].round(2).tolist(),
                                       # DINOv2 probe vote over the alliance's 3 teams (centred mean log-prob of
                                       # its crops, > 0 = looks like that team; scaled by w_dino in the Viterbi)
                                       **({"dino": np.round(dino_ev[s.id], 2).tolist()} if s.id in dino_ev else {}),
                                       "team": team_of_seg.get(s.id)} for s in asegs}

    # ---------------- per-team field tracks
    team_track: dict[int, list] = {t: [] for t in info.red + info.blue}
    for s in segs:
        team = team_of_seg.get(s.id)
        if team is not None:
            team_track[team].extend(s.points)
    for t in team_track:
        team_track[t].sort()

    # ---------------- shots -> robots
    dets_by_view_t: dict[str, dict[float, list]] = {}
    for d in md.dets:
        dets_by_view_t.setdefault(d.view, {}).setdefault(round(d.t, 3), []).append(d)
    team_arr = {t: np.array(p) for t, p in team_track.items() if len(p) >= 2}
    alliance_of = {t: ("red" if t in info.red else "blue") for t in info.red + info.blue}

    def team_pos(team: int, t: float):
        p = _track_position(team_arr.get(team), t)
        if p is None or (not p[2] and not FILL_GAPS):
            return None
        return p

    # per view, each labelled segment's boxes in time order (BOX_HOLD_S, LINK_FILL_TIE)
    seg_boxes: dict[str, dict[int, tuple]] = {}
    if BOX_HOLD_S > 0 or LINK_FILL_TIE:
        by = {}
        for d in md.dets:
            sid = det_seg.get(id(d))
            if sid is not None:
                by.setdefault(d.view, {}).setdefault(sid, []).append(d)
        for v_, segd in by.items():
            seg_boxes[v_] = {}
            for sid, ds in segd.items():
                ds.sort(key=lambda d: d.t)
                seg_boxes[v_][sid] = (np.array([d.t for d in ds]), ds)

    def fill_drop(view: str, sid, t: float) -> float:
        e = seg_boxes.get(view, {}).get(sid)
        if e is None:
            return 0.0
        ts, ds = e
        i0, i1, i2 = np.searchsorted(ts, [t - LINK_FILL_WIN, t, t + LINK_FILL_WIN])
        pre = [d.fill for d in ds[i0:i1] if d.fill is not None]
        post = [d.fill for d in ds[i1:i2] if d.fill is not None]
        return max(0.0, float(np.median(pre) - np.median(post))) if pre and post else 0.0

    def elim_team(d, box, frame_dets, cam, t: float):
        """The team of a "?" box by elimination (ELIM_BOXES), or None."""
        cand = set(info.teams(d.alliance))
        for o in frame_dets:  # labelled on another box of this frame
            s2 = det_seg.get(id(o))
            if o is not d and s2 is not None:
                cand.discard(team_of_seg.get(s2))
        x0, y0, x1, y1 = box
        bw = max(x1 - x0, 1.0)
        for tm in list(cand):  # tracked somewhere this box is not
            p = team_pos(tm, t)
            if p is None or not p[2] or cam is None:
                continue
            u, v = cam.project(np.array([[p[0], p[1], 0.0]]))[0]
            if not (np.isfinite(u) and np.isfinite(v)) or np.hypot(u - 0.5 * (x0 + x1), v - y1) > ELIM_FAR * bw:
                cand.discard(tm)
        return cand.pop() if len(cand) == 1 else None

    def elim_arc(cam, lu: float, lv: float, t: float):
        """The shooter of an arc with no robot box or tracked robot near it (ELIM_ARCS), or None: launched inside
        an alliance zone, by the one robot of that alliance not tracked more than ELIM_ARC_M from the launch."""
        xy = cam.backproject(np.array([[lu, lv]]), z=0.55)[0]
        if not np.all(np.isfinite(xy)):
            return None
        al_z = [a for a in ("red", "blue") if fieldmod.in_alliance_zone(a, float(xy[0]), 0.0)]
        if len(al_z) != 1:
            return None
        cand = []
        for tm in info.teams(al_z[0]):
            p = team_pos(tm, t)
            if p is None or not p[2] or np.hypot(p[0] - xy[0], p[1] - xy[1]) <= ELIM_ARC_M:
                cand.append(tm)
        return cand[0] if len(cand) == 1 else None

    shot_list, inferred_shots = [], []
    n_cross = n_gated = n_zone_out = n_held = n_elim = n_elim_arc = 0
    for vname, tracks in md.tracks_all.items():
        if SPLIT_TRACKS:
            tracks = fuel.split_at_jumps(tracks)
        cam_track = md.cams.get(vname)
        tdict = dets_by_view_t.get(vname, {})
        tkeys = np.array(sorted(tdict)) if tdict else np.zeros(0)
        ox, oy = vlist[vname].offset(W, H)
        _, vy0_, _, vy1_ = vlist[vname].scaled(W, H)
        view_h = vy1_ - vy0_
        for tr in tracks:
            u0, v0 = tr.pts[0][1], tr.pts[0][2]
            cam = cam_track.at(tr.pts[0][0]) if cam_track is not None else None
            ppm = _px_per_m(cam, u0, v0)
            if not fuel.is_shot(tr, ppm):
                continue
            t0 = tr.pts[0][0]
            # launch point: back-extrapolate ~2 frames along the initial velocity
            vu, vv = tr.velocity0()
            lu, lv = u0 - vu * 2 / 60, v0 - vv * 2 / 60
            team, al, conf, how = None, None, 0.0, None
            inferred, shown = False, None  # team by elimination: used (ELIM_*) / display only
            # (1) robot box in the same view at (nearly) the same time
            if len(tkeys):
                j = int(np.argmin(np.abs(tkeys - t0)))
                frame_dets = list(tdict[tkeys[j]]) if abs(tkeys[j] - t0) <= 0.2 else []
                held = set()
                if BOX_HOLD_S > 0:  # segments seen in this view in the last BOX_HOLD_S s but not in this frame
                    have = {det_seg.get(id(d)) for d in frame_dets}
                    for sid, (ts, ds) in seg_boxes.get(vname, {}).items():
                        i = int(np.searchsorted(ts, t0, side="right")) - 1
                        if sid not in have and i >= 0 and t0 - ts[i] <= BOX_HOLD_S:
                            frame_dets.append(ds[i])
                            held.add(id(ds[i]))
                if frame_dets:
                    cands = []
                    for d in frame_dets:
                        x0, y0, x1, y1 = d.box
                        x0, x1, y0, y1 = x0 - ox, x1 - ox, y0 - oy, y1 - oy
                        bw, bh = x1 - x0, y1 - y0
                        # expanded box: shots leave from the top of the robot
                        ex0, ex1, ey0, ey1 = x0 - 0.25 * bw, x1 + 0.25 * bw, y0 - 0.9 * bh, y1
                        dx = max(ex0 - lu, 0, lu - ex1)
                        dy = max(ey0 - lv, 0, lv - ey1)
                        dist = np.hypot(dx, dy) / max(bh, 1)
                        # several boxes can contain the launch point (robots overlap in the image):
                        # prefer the one whose top edge it leaves from
                        score = dist + (TOP_TIEBREAK * abs(v0 - y0) / max(bh, 1) if SHOT_GATE else 0.0)
                        # back in time to this box's top edge (0 if the first sighting is already below it)
                        tau = max(-GATE_TRACE_S, min(0.0, (y0 - v0) / vv)) if vv < -1e-6 else 0.0
                        crossed = (x0 - SIDE_MARGIN * bw <= u0 + vu * tau <= x1 + SIDE_MARGIN * bw
                                   and (vv < -1e-6 and (y0 - v0) / vv >= -GATE_TRACE_S or v0 >= y0))
                        if LINK_X_TIE:
                            uc = u0
                            if LINK_X_CROSS and vv < -1e-6:  # rising: back in time to this box's top edge
                                uc = u0 + vu * max(-0.3, min(0.0, (y0 - v0) / vv))
                            score += LINK_X_TIE * abs(uc - 0.5 * (x0 + x1)) / max(bw, 1)
                        if id(d) in held:
                            score += BOX_HOLD_PEN
                        if LINK_FILL_TIE and dist <= 0.6:
                            score -= LINK_FILL_TIE * fill_drop(vname, det_seg.get(id(d)), t0)
                        cands.append((score, dist, d, (x0, y0, x1, y1), tau, crossed))
                    # robots of BOTH alliances at the launch point: in teleop a robot can only score from its
                    # own zone, so prefer a candidate standing in it (failure-finder cycle 24: cross-alliance
                    # picks 72 -> 91 % right vs the scouts; shares unchanged, per-launch links right more often)
                    close = [c for c in cands if c[1] <= 0.6]
                    if OVERLAP_ZONE_RULE and len({c[2].alliance for c in close}) > 1:
                        mt_l = tl.match_time(t0)
                        if mt_l is not None and mt_l >= config.AUTO_LEN:
                            inz = [c for c in close if c[2].xy is not None
                                   and fieldmod.in_alliance_zone(c[2].alliance, c[2].xy[0])]
                            if inz:
                                cands = inz
                    if LINK_FIRST_CROSS and vname in LINK_FIRST_CROSS_VIEWS:  # the first top crossed, going back
                        crossed = [c for c in cands if c[1] <= 0.6 and c[5]]
                        if crossed:
                            cands = [max(crossed, key=lambda c: c[4])]
                    best, bd, bscore, bbox = None, 1e9, 1e9, None
                    for score, dist, d, box, *_ in cands:
                        if score < bscore:
                            best, bd, bscore, bbox = d, dist, score, box
                    if best is not None and bd <= 0.6 and SHOT_GATE and not _box_shot_ok(tr, bbox, view_h):
                        n_gated += 1
                        continue  # hopper rattle, a bounce, or a throw from above the robot (human player)
                    if best is not None and bd <= 0.6:
                        sid = det_seg.get(id(best))
                        team = team_of_seg.get(sid) if sid is not None else None
                        al, conf, how = best.alliance, float(np.exp(-bd)), "box"
                        n_held += id(best) in held
                        if team is None and sid is None:
                            et = elim_team(best, bbox, frame_dets, cam, t0)
                            if et is not None and ELIM_BOXES:
                                team, inferred = et, True
                                n_elim += 1
                            else:
                                shown = et
            # (2) robot tracked in any view: project its field position into this view
            if team is None and cam is not None:
                best, bd = None, 1e9
                for tm in team_arr:
                    xy = team_pos(tm, t0)
                    if xy is None:
                        continue
                    top = cam.project(np.array([[xy[0], xy[1], 0.55]]))[0]
                    d_m = np.hypot(top[0] - lu, top[1] - lv) / max(ppm, 1e-3)
                    if not xy[2]:  # hidden: in units of its wider radius
                        d_m /= PROJ_HIDDEN_MAX_M
                    if d_m < bd:
                        best, bd = tm, d_m
                if best is not None and bd < 1.0 and SHOT_GATE and not _proj_shot_ok(tr, ppm):
                    n_gated += 1
                    continue
                if best is not None and bd < 1.0:
                    team, al, conf, how = best, alliance_of[best], float(np.exp(-bd)) * 0.8, "proj"
                    if not team_pos(best, t0)[2]:  # traced to an interpolated (hidden) position
                        conf *= FILLED_SHOT_WEIGHT
                        how = "proj_hidden"
                    n_cross += 1
            if team is None and al is None and cam is not None and _proj_shot_ok(tr, ppm):
                et = elim_arc(cam, lu, lv, t0)
                if et is not None and ELIM_ARCS:
                    team, al, conf, how, inferred = et, alliance_of[et], ELIM_ARC_WEIGHT, "elim", True
                    n_elim_arc += 1
                elif et is not None:  # shown in the inspector, not counted
                    inferred_shots.append({"t": t0, "view": vname, "team": et, "alliance": alliance_of[et],
                                           "how": "elim", "conf": ELIM_ARC_WEIGHT, "inferred": True,
                                           "pts": [(round(p[0], 3), round(p[1], 1), round(p[2], 1))
                                                   for p in tr.pts]})
            if team is None and al is None:
                continue
            if LAUNCH_ZONE_WEIGHT < 1.0 and team is not None:  # launched from outside its own zone: not a shot
                mt_l = tl.match_time(t0)
                xy = team_pos(team, t0) if (mt_l is not None and mt_l >= config.AUTO_LEN) else None
                if xy is not None and not fieldmod.in_alliance_zone(al, xy[0], LAUNCH_ZONE_MARGIN):
                    n_zone_out += 1
                    if LAUNCH_ZONE_WEIGHT <= 0:
                        continue
                    conf *= LAUNCH_ZONE_WEIGHT
            # does the arc head for the shooter's own hub? (passes / feeds usually do not)
            toward = None
            if cam is not None and al is not None:
                hc = fieldmod.hub_center(al)
                hu, hv = cam.project(np.array([[hc[0], hc[1], 1.9]]))[0]
                d0 = np.hypot(tr.pts[0][1] - hu, tr.pts[0][2] - hv)
                d1 = np.hypot(tr.pts[-1][1] - hu, tr.pts[-1][2] - hv)
                toward = bool(d1 < d0 - 0.3 * ppm)
                if toward is False:
                    conf *= AWAY_SHOT_WEIGHT
            shot_list.append({"t": t0, "view": vname, "team": team, "alliance": al, "toward": toward,
                              "u": u0, "v": v0, "conf": conf, "how": how,
                              **({"inferred": True} if inferred else {}),
                              **({"elim_team": shown} if shown is not None else {}),
                              "pts": [(round(p[0], 3), round(p[1], 1), round(p[2], 1)) for p in tr.pts]})

    # "?" boxes: the team elimination points to (ELIM_BOXES uses it for launches; the inspector shows it, marked)
    elim_of_det: dict[int, int] = {}
    for vname, tdict in dets_by_view_t.items():
        cam_track = md.cams.get(vname)
        if cam_track is None:
            continue
        ox, oy = vlist[vname].offset(W, H)
        for t, frame in tdict.items():
            for d in frame:
                if det_seg.get(id(d)) is None:
                    x0, y0, x1, y1 = d.box
                    et = elim_team(d, (x0 - ox, y0 - oy, x1 - ox, y1 - oy), frame, cam_track.at(t), t)
                    if et is not None:
                        elim_of_det[id(d)] = et

    # ---------------- zone occupancy
    # shooting signals for a version-2 credit model (autoscout/shooting.py): per team, on the TeamSignals grid
    shoot_dinoc, shoot_drain = {}, {}
    cm = attribution.load_model()
    if cm is not None and (cm.get("g_dinoc") or cm.get("g_drain")):
        from . import shooting

        sgrid = np.arange(md_t0(md), md_t1(md), shooting.GRID_STEP)
        ridge = config.ROOT / (cm.get("shooting") or {}).get("ridge", "models/shooting_ridge.npz")
        if cm.get("g_dinoc"):  # None (factor 1) until the match's crops are embedded (build_dino_probe.py embed)
            shoot_dinoc = shooting.dinoc_series(md, shooting.EMB_DIR / f"{md.key}.npz", ridge, det_seg, team_of_seg,
                                                sgrid) or {}
        if cm.get("g_drain"):
            shoot_drain = shooting.drain_series(md, det_seg, team_of_seg, sgrid)
    signals = {}
    parked: dict = {}  # Options.park_ext: team -> median 20 s movement extent (m)
    for al in ("red", "blue"):
        for team in info.teams(al):
            pts = np.array(team_track[team]) if team_track[team] else np.zeros((0, 3))
            ts = attribution.TeamSignals(team)
            sh = [s for s in shot_list if s["team"] == team]
            ts.shots = np.array([s["t"] for s in sh])
            ts.shot_w = np.array([s["conf"] for s in sh])
            if len(pts):
                grid = np.arange(md_t0(md), md_t1(md), 0.5)
                zv = np.full(len(grid), np.nan)
                vv = np.zeros(len(grid))
                for gi, g in enumerate(grid):
                    j = np.searchsorted(pts[:, 0], g)
                    near = [pts[jj] for jj in (j - 1, j) if 0 <= jj < len(pts) and abs(pts[jj][0] - g) < 1.0]
                    vv[gi] = float(any(abs(pts[jj][0] - g) <= VIS_WINDOW for jj in (j - 1, j) if 0 <= jj < len(pts)))
                    if near:
                        x = near[0][1]
                        zv[gi] = 1.0 if fieldmod.in_alliance_zone(al, x) else 0.0
                    elif FILL_GAPS:
                        p = _track_position(pts, g)
                        if p is not None:
                            zv[gi] = 1.0 if fieldmod.in_alliance_zone(al, p[0]) else 0.0
                ts.zone_t, ts.zone_v = grid, zv
                ts.vis_t, ts.vis_v = grid, vv
                if attribution.load_model() is not None:  # distance to own hub (learned credit model only)
                    hxy = fieldmod.hub_center(al)
                    dv = np.full(len(grid), np.nan)
                    for gi, g in enumerate(grid):
                        p = _track_position(pts, g)
                        if p is not None:
                            dv[gi] = float(np.hypot(p[0] - hxy[0], p[1] - hxy[1]))
                    ts.dist_t, ts.dist_v = grid, dv
            # hopper drain: the FUEL-yellow fraction inside the robot's box falling quickly
            # (same view only, so view changes do not look like drains) -> pseudo-shots
            drains = _hopper_drains([d for s in segs if team_of_seg.get(s.id) == team for d in s.dets])
            ts.n_drain = 0
            if drains and DRAIN_WEIGHT > 0:
                dt_, dw_ = zip(*drains)
                ts.shots = np.concatenate([ts.shots, np.array(dt_)])
                ts.shot_w = np.concatenate([ts.shot_w, DRAIN_WEIGHT * np.array(dw_)])
                ts.n_drain = len(drains)
            # absent state / parked robot (Options.absent_cost*, park_ext)
            if team_absent.get(team):
                ts.absent = team_absent[team]
            if (opts.absent_dead_hidden_s is not None and team_dead.get(team)
                    and sum(b - a for a, b in team_absent.get(team, [])) >= opts.absent_dead_hidden_s):
                ts.no_hidden = True
            if opts.park_ext > 0:
                parked[team] = _median_extent(pts)
                ts.prior_off = parked[team] is not None and parked[team] < opts.park_ext
            if team in shoot_dinoc:
                ts.dinoc_t, ts.dinoc_v = shoot_dinoc[team]
            if team in shoot_drain:
                ts.drain_t, ts.drain_v = shoot_drain[team]
            signals[team] = ts

    # ---------------- vision-only counter (needs a fitted model)
    vision_rate, vision_only = {}, {}
    vm = vision_count.load()
    if vm is not None and md.hub_feats:
        w, meta = vm
        feats = {k: np.array(v, dtype=np.float64).reshape(-1, 3) for k, v in md.hub_feats.items()}
        inactive = tba.inactive_periods(info) if info.breakdown else {"red": set(), "blue": set()}
        for al in ("red", "blue"):
            tb, X = vision_count.binned(feats, al, md_t0(md), md_t1(md))
            r = np.clip(vision_count.predict(w, vision_count.smooth(X, meta.get("smooth", 3))), 0, None)
            lag = meta.get("lag", 2.5)
            act = np.array([0.0 if (tl.match_time(t - lag) is None or
                                    attribution.period_of(tl.match_time(t - lag)) in inactive[al]) else 1.0
                            for t in tb])
            vision_rate[al] = (tb + lag, r)  # rate expressed at counter time
            vision_only[al] = round(float(r @ act), 1)

    # ---------------- alliance scoring timeline -> per-team split
    result_teams = {}
    sources = {}
    fuel_log: list = []
    for al in ("red", "blue"):
        hub = info.hub(al)
        if tl.counts.get(al):
            inc = attribution.increments_from_counts(_match_counts(tl, al, hub))
            sources[al] = "overlay"
        elif hub:
            pc = {k.replace("Count", ""): v for k, v in hub.items()
                  if k.endswith("Count") and k not in ("teleopCount", "totalCount")}
            rate = vision_rate.get(al)
            inc = attribution.increments_from_periods(pc, tl.video_time, rate=rate,
                                                      to_match_time=tl.match_time)
            sources[al] = "tba_periods+vision_timing" if rate is not None else "tba_periods"
        else:
            inc = []
            sources[al] = "none"
        split = attribution.split_increments(inc, [signals[t] for t in info.teams(al)],
                                             tl.match_time, log=fuel_log, opr=_attr_opr(info))
        split_video = split
        if attribution.load_model() is not None:
            # the learned model folds in an event fuel-OPR prior; keep the pure-video split beside it
            keep, attribution.ATTR_MODEL = attribution.ATTR_MODEL, None
            try:
                split_video = attribution.split_increments(inc, [signals[t] for t in info.teams(al)], tl.match_time)
            finally:
                attribution.ATTR_MODEL = keep
        towers = tba.tower_points(info, al) if info.breakdown else {}
        overlay_total = sum(n for _, n in inc)
        tba_total = hub.get("totalCount") if hub else None
        scale = (tba_total / overlay_total) if (tba_total and overlay_total and sources[al] == "overlay") else 1.0
        for team in info.teams(al):
            per = split.get(team, {})
            fuel_raw = sum(per.values())
            tw = towers.get(team, {})
            result_teams[team] = {
                "alliance": al,
                "fuel": round(fuel_raw * scale, 1),
                "fuel_by_period": {k: round(v * scale, 1) for k, v in sorted(per.items())},
                "fuel_unscaled": round(fuel_raw, 1),
                "fuel_video": round(sum(split_video.get(team, {}).values()) * scale, 1),  # video evidence only
                "shots_detected": int(len(signals[team].shots) - getattr(signals[team], "n_drain", 0)),
                "hopper_drain_events": int(getattr(signals[team], "n_drain", 0)),
                **tw,
                "points": round(fuel_raw * scale + tw.get("auto_tower_points", 0) + tw.get("endgame_tower_points", 0), 1),
            }
        result_teams[f"_{al}"] = {"source": sources[al], "timeline_total": overlay_total,
                                  "tba_total": tba_total, "scale": round(scale, 3)}

    # ---------------- auto paths
    paths = {}
    for team, pts in team_track.items():
        if not pts:
            continue
        mt = [(tl.match_time(p[0]), p[1], p[2]) for p in pts]
        paths[team] = [(round(a, 2), round(x, 3), round(y, 3)) for a, x, y in mt if a is not None]

    det_rows = []
    for d in md.dets:
        sid = det_seg.get(id(d))
        det_rows.append((round(d.t, 3), d.view, *[round(c, 1) for c in d.box], d.alliance,
                         team_of_seg.get(sid) if sid is not None else None, sid,
                         None if d.fill is None else round(d.fill, 3), elim_of_det.get(id(d))))
    md.det_rows = det_rows
    return {
        "match": info.key,
        "teams": result_teams,
        "paths": paths,
        "shots": shot_list,
        "inferred_shots": inferred_shots,  # box-less arcs elimination would give a team (ELIM_ARCS off): display only
        "identity": ident_debug,
        "timeline": {"auto_start": tl.auto_start, "teleop_start": tl.teleop_start, "layout": tl.layout,
                     "counts": tl.counts},
        "calibration": {k: {"rms": c.rms, "n_tags": c.n_tags} for k, c in md.cams.items()},
        "n_dets": len(md.dets),
        "vision_only": vision_only,
        "fuel_log": fuel_log,
        "n_segments": len(segs),
        "shots_gated": n_gated,
        "shots_out_of_zone": n_zone_out,
        "shots_held_box": n_held,
        "shots_elim_box": n_elim,
        "shots_elim_arc": n_elim_arc,
        **({"credit_model": attribution.ATTR_MODEL} if attribution.ATTR_MODEL else {}),
        # only with Options.absent_cost* / park_ext
        **({"absent": {t: [[round(a, 2), round(b, 2)] for a, b in iv] for t, iv in team_absent.items() if iv},
            "dead": [t for t, d in team_dead.items() if d]} if team_dead else {}),
        **({"parked": {t: (None if e is None else round(e, 2)) for t, e in parked.items()}} if parked else {}),
    }


AWAY_SHOT_WEIGHT = 0.35  # arcs not heading for the shooter's hub (passes) count less
DRAIN_WEIGHT = 0.0  # hopper-fill drains are noisy (robots driving through fuel); off by default
DRAIN_SCALE = 25.0  # pseudo-shots per unit of box-fill fraction drained


def _hopper_drains(dets: list) -> list[tuple[float, float]]:
    """[(t, weight)] where a robot's hopper visibly empties (per view, ~1 s windows)."""
    out = []
    by_view: dict[str, list] = {}
    for d in dets:
        if d.fill is not None:
            by_view.setdefault(d.view, []).append((d.t, d.fill))
    for view, arr in by_view.items():
        arr.sort()
        a = np.array(arr)
        if len(a) < 6:
            continue
        # median-of-3 to suppress single-frame occlusions
        f = np.array([np.median(a[max(0, i - 1):i + 2, 1]) for i in range(len(a))])
        t = a[:, 0]
        for i in range(len(a)):
            j = np.searchsorted(t, t[i] + 1.0)
            if j >= len(a) or t[j] - t[i] > 1.6:
                continue
            drop = f[i] - f[j]
            if drop > 0.03:
                out.append(((t[i] + t[j]) / 2, min(drop, 0.3) * DRAIN_SCALE / max(1, j - i)))
    return out


def md_t0(md: MatchData) -> float:
    return (md.timeline.auto_start or FIRST_AUTO_START) - 1.0


def md_t1(md: MatchData) -> float:
    ts = md.timeline.teleop_start or ((md.timeline.auto_start or FIRST_AUTO_START) + FIRST_TELEOP_GAP)
    return ts + config.TELEOP_LEN + 5.0


def process_match(key: str, opts: Options | None = None) -> dict:
    opts = opts or Options()
    info = tba.match(key)
    if not info.youtube:
        raise RuntimeError(f"{key}: no video on TBA")
    path = download(key, info.youtube[0])
    video = Video(path)
    out_dir = opts.out_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    # overlay timeline (cached)
    ov_cache = config.CACHE / f"overlay_{key}.json"
    if ov_cache.exists():
        d = json.loads(ov_cache.read_text())
        layout, rows = d["layout"], d["rows"]
    else:
        layout, rows = overlay.scan(video, hz=opts.overlay_hz)
        ov_cache.write_text(json.dumps({"layout": layout, "rows": rows}, default=str))
    tl = overlay.build_timeline(layout, rows)
    if tl.auto_start is None:
        tl.auto_start = FIRST_AUTO_START
        tl.teleop_start = FIRST_AUTO_START + FIRST_TELEOP_GAP
    mid = video.frame_at(min(video.duration - 1, (tl.auto_start or 6) + 60))
    prod = views.detect_production(mid)
    tele = tl.teleop_start or (tl.auto_start + FIRST_TELEOP_GAP)
    # end views that fail to calibrate get event-prior recovery candidates, kept only if their robots
    # agree with the main view (data/exp_calib/RESULTS_section.md)
    recover: dict = {}
    cams = calibrate(video, prod, tl.auto_start - 1, tele + config.TELEOP_LEN + 2, event=info.event_key,
                     match=key, fallbacks=recover)
    if opts.debug:
        print(f"{key}: layout={layout} prod={prod} cams={ {k: round(c.rms, 2) for k, c in cams.items()} } "
              f"auto_start={tl.auto_start} ({time.time() - tic:.0f}s)")
    md = MatchData(key, info, prod, cams, tl)
    md = scan_video(video, info, prod, cams, tl, md_t0(md), md_t1(md), opts)
    if "main" in recover:  # first, so that the end views are checked against the main camera kept
        main_cands, trusted = recover.pop("main")
        try:
            r = accept_main_cam(md, main_cands, prefer="event_prior", trusted=trusted)
            if r["flag"] or opts.debug:
                print(f"{key}: main camera {r['accepted'] or 'kept'}{' - FLAG: most window poses bad' if r['flag'] else ''}")
        except Exception as e:  # never lose a scan to the main-camera check
            md.main_calib = {"error": repr(e)}
            print(f"{key}: accept_main_cam failed: {e!r}")
    if recover:
        accept_recovered_cams(md, recover)
    if opts.debug:
        print(f"{key}: scanned ({time.time() - tic:.0f}s), dets={len(md.dets)}")
    res = analyse(md, opts)
    res["runtime_s"] = round(time.time() - tic, 1)
    _save_scan(md, out_dir)
    # per-frame hub activity features (for training / applying the vision-only counter)
    feats = {"|".join(k): np.array(v, dtype=np.float32).round(3).tolist() for k, v in md.hub_feats.items()}
    (out_dir / "hub_feats.json").write_text(json.dumps(feats))
    (out_dir / "dets.json").write_text(json.dumps(getattr(md, "det_rows", []), default=_json_default))
    if md.fuel_map is not None and md.fuel_map_n:
        np.save(out_dir / "fuel_map.npy", md.fuel_map / md.fuel_map_n)
        (out_dir / "fuel_counts.json").write_text(json.dumps(md.fuel_counts))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=_json_default))
    return res


SAVED_CROP_H = 96  # bumper crops kept in scan.pkl are at most this tall (OCR in the scan itself uses full size)


def _img(x):
    """Crop images in a saved scan are JPEG bytes; fresh ones are arrays."""
    if isinstance(x, bytes):
        return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
    return x


def _jpg(img, max_h: int | None = None, q: int = 85):
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.size:
        if max_h and img.shape[0] > max_h:  # close-up end-camera crops: digits stay ~40 px tall
            s = max_h / img.shape[0]
            img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes() if ok else None
    return img if isinstance(img, bytes) else None


def _save_scan(md: MatchData, out_dir: Path):
    """Pickle the scan for fast re-analysis. Bumper images and appearance thumbnails are kept
    JPEG-compressed (a few KB each, ~30 MB per match) so bumper OCR and appearance embeddings can be
    re-run without decoding the video again; OCR reads and CNN embeddings are cached as well."""
    import pickle

    md.crops = {k: (_jpg(v[0], SAVED_CROP_H), v[1], v[2], _jpg(v[3], q=90) if len(v) > 3 else None)
                for k, v in md.crops.items()}
    with open(out_dir / "scan.pkl", "wb") as fh:
        pickle.dump(md, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_dino(md: MatchData, match_dir: Path) -> bool:
    """Attach DINOv2 crop embeddings (scripts/build_dino.py output) to a scan, if present."""
    p = Path(match_dir) / "dino.npz"
    if not p.exists():
        return False
    d = np.load(p)
    md.dino_cache = {int(i): f.astype(np.float32) for i, f in zip(d["ids"], d["feats"])}
    return True


# 60 Hz flight tracks (data/exp_track/RESULTS_section.md): FUEL tracked on every frame instead of every other
# one separates the balls of a volley; on 54 scouted quals with process-of-elimination linking, refit fuel MAE
# 41.2 -> 39.8 (paired 95 % CI -2.67 .. -0.22). Rebuilt from config.BLOBS60 logs (autoscout/tracks60.py); a
# match without one keeps its scan's 30 Hz tracks. result.json records which: "fuel_hz".
FUEL_60HZ = True
# ... true 60p video only: tracks60.load returns None for a webcast of 30p pictures shown twice (most FIRST-production
# events), which keeps its 30 Hz tracks


def load_fuel60(md: MatchData) -> int:
    """With FUEL_60HZ, swap in the match's 60 Hz flight tracks when it has a 60 Hz blob log; sets and returns
    md.fuel_hz (60 or 30). Call after loading a scan, like load_dino; the scan's own tracks are md.tracks_30."""
    md.fuel_hz = 30
    if FUEL_60HZ:
        from . import tracks60

        tracks = tracks60.load(md.key)
        if tracks is not None:
            md.tracks_30, md.tracks_all, md.fuel_hz = md.tracks_all, tracks, 60
    return md.fuel_hz


# Cross-match DINOv2 probe (scripts/build_dino_probe.py, Options.w_dino): for every crop, log-probabilities
# over each alliance's 3 teams from a linear probe on frozen DINOv2 features (labels: segments with clear bumper
# reads). One <match>.npz per match in this folder. "causal" (build_dino_probe.py probe --causal, no hindsight):
# each match's probe is trained only on the event's earlier matches, as it would be live. "default": on all the
# event's other matches (after-the-fact analysis only).
DINO_PROBE_DIR = config.DATA / "exp_dino" / "probe" / ("causal" if NO_HINDSIGHT else "default")


def load_dino_probe(key: str) -> dict:
    """{crop id: {team: log q}} from DINO_PROBE_DIR/<key>.npz ({} if there is none)."""
    root = Path(DINO_PROBE_DIR)
    p = (root if root.is_absolute() else config.ROOT / root) / f"{key}.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    teams = [int(t) for t in d["teams"]]
    return {int(c): dict(zip(teams, row)) for c, row in zip(d["ids"], d["logq"].astype(np.float32))}


def reanalyse(key: str, opts: Options | None = None) -> dict:
    """Re-run tracking/identity/attribution from a cached scan (seconds instead of minutes)."""
    import pickle

    opts = opts or Options()
    out_dir = opts.out_dir / key
    with open(out_dir / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    load_dino(md, out_dir)
    load_fuel60(md)
    n_ocr = len(getattr(md, "ocr_cache", None) or {})
    res = analyse(md, opts)
    res["fuel_hz"] = md.fuel_hz
    if md.fuel_hz == 60:  # a scan saved below keeps its own 30 Hz tracks
        md.tracks_all = md.tracks_30
        del md.tracks_30
    old = json.loads((out_dir / "result.json").read_text()) if (out_dir / "result.json").exists() else {}
    res["runtime_s"] = old.get("runtime_s")
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=_json_default))
    (out_dir / "dets.json").write_text(json.dumps(getattr(md, "det_rows", []), default=_json_default))
    if len(getattr(md, "ocr_cache", None) or {}) > n_ocr:
        # keep the bumper reads this analysis made (new segment picks), so later re-analyses reuse them instead
        # of reading again; written aside and swapped in, as other processes read scans concurrently
        import os
        tmp = out_dir / "_reanalyse_tmp"
        tmp.mkdir(exist_ok=True)
        _save_scan(md, tmp)
        os.replace(tmp / "scan.pkl", out_dir / "scan.pkl")
        tmp.rmdir()
    return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
