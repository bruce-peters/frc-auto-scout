"""Ball-equivalent volley signal for the learned credit model (failure-finder cycle 27/28).

Per robot, every RISING flight track that leaves its box (the pipeline.analyse step (1) box link) contributes
    be = mean blob area of the track / area of one 15 cm FUEL ball at that range
(px/m from the view's camera at the track's first point, pipeline._px_per_m). Launch arcs stop separating in a
dense stream, but the moving yellow keeps growing with the number of balls, so summed be grows ~balls^0.7
where launch counts grow ~balls^0.3 (data/exp_audit, cycle 27).

What it reads from what pipeline.analyse has in memory:
    md.tracks_all : {view: [fuel.FlightTrack]} (pts = (t, u, v, area), view-local pixels)
    md.dets       : [robots.Detection] (t, view, box in full-frame 1920x1080 pixels)
    det_team      : [team | None] aligned with md.dets, i.e. team_of_seg.get(det_seg.get(id(d)))
    md.cams       : {view: field.CameraTrack} (for px/m; a view without one uses pipeline._px_per_m(None) = 80)
    md.prod       : production layout name (views.PRODUCTIONS[md.prod] gives the view offsets)

    vol = ball_equivalents(md.tracks_all, md.dets, det_team, md.cams, md.prod)   # min_speed=MIN_SPEED (2 m/s)
    -> {team: np.ndarray (n, 2) of (video t of the track's first point, be)}, sorted by t

The model feature per counter jump at video time t is the kernel-weighted sum over the robot's rows:
    s_beq = sum(attribution.kernel(t - be_t) * be)            (as in data/exp_audit/ds_vol.pkl)
and it enters the weight as  lw += g_vol * log1p(s_beq)   (spec lin/all+dist+mlprior+volexp; g_vol 0.21 at lambda 3e-5,
0.15 at export_model.py's default 3e-4: the ridge pulls g_vol toward 0).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from autoscout import pipeline as P, views

BALL_R = 0.075  # m: FUEL radius
MIN_SPEED = 2.0  # m/s at the launch point's range: drops 45 % of the slow rising tracks but only 9.5 % of be (cycle 28)


def ball_equivalents(tracks_all: dict, dets: list, det_team: list, cams: dict, prod: str,
                     min_speed: float = MIN_SPEED, W: int = 1920, H: int = 1080, with_speed: bool = False) -> dict[int, np.ndarray]:
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    by_vt: dict = defaultdict(lambda: defaultdict(list))  # view -> det time (rounded like analyse) -> [(box, team)]
    for d, team in zip(dets, det_team):
        by_vt[d.view][round(d.t, 3)].append((d.box, team))
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    out = defaultdict(list)
    for vname, tracks in tracks_all.items():
        if vname not in tk or vname not in vl:
            continue
        ox, oy = vl[vname].offset(W, H)
        cam_t = cams.get(vname)
        ts = tk[vname]
        for tr in tracks:
            if len(tr.pts) < 4:
                continue
            t0, u0, v0 = tr.pts[0][0], tr.pts[0][1], tr.pts[0][2]
            vu, vv = tr.velocity0()
            if vv >= 0:  # rising only (image v decreasing)
                continue
            j = int(np.argmin(np.abs(ts - t0)))
            if abs(ts[j] - t0) > 0.2:
                continue
            lu, lv = u0 - vu * 2 / 60, v0 - vv * 2 / 60  # launch point, as analyse
            best, bd, bs = None, 1e9, 1e9
            for box, team in by_vt[vname][ts[j]]:
                x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
                bw, bh = x1 - x0, y1 - y0
                dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw)
                dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
                dist = np.hypot(dx, dy) / max(bh, 1)
                score = dist + P.TOP_TIEBREAK * abs(v0 - y0) / max(bh, 1)
                if score < bs:
                    best, bd, bs = team, dist, score
            if best is None or bd > 0.6:
                continue
            ppm = P._px_per_m(cam_t.at(t0) if cam_t is not None else None, u0, v0)
            speed = np.hypot(vu, vv) / max(ppm, 1e-3)  # m/s at the launch point's range
            if min_speed > 0 and speed < min_speed:
                continue
            area = float(np.mean([q[3] for q in tr.pts]))
            row = (t0, area / max(np.pi * (BALL_R * ppm) ** 2, 1.0))
            out[best].append(row + (speed,) if with_speed else row)
    return {t: np.array(sorted(v)) for t, v in out.items()}


def s_beq(vol_team: np.ndarray | None, t: float) -> float:
    """Kernel-weighted ball-equivalents of one robot for a counter jump at video time t."""
    from autoscout import attribution as A
    if vol_team is None or not len(vol_team):
        return 0.0
    return float((A.kernel(t - vol_team[:, 0]) * vol_team[:, 1]).sum())
