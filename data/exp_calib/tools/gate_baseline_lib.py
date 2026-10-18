"""Null baseline for the cross-view gate (shared by gate_baseline.py and probe_view.py)."""
import numpy as np

from autoscout import pipeline as P, robots, views


def null_error(md, vname, cam, shift=30.0):
    """cross_view_error with the main-view robots taken `shift` s later (chance level)."""
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    main = md.cams["main"]

    def place(d, c, limit):
        ox, oy = vl[d.view].offset(1920, 1080)
        c = c.at(d.t)
        xy = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
        if not np.all(np.isfinite(xy)):
            return None
        dv = xy - c.center_world()[:2]
        rng = float(np.hypot(*dv))
        xy = xy + P.ROBOT_HALF_DEPTH * dv / max(rng, 1e-6)
        return xy if robots.in_field(xy) and (limit is None or rng <= limit) else None

    mains, ends = {}, {}
    for d in md.dets:
        if d.view == "main":
            p = place(d, main, None)
            if p is not None:
                mains.setdefault(round(d.t, 3), []).append((d.alliance, p))
        elif d.view == vname:
            ends.setdefault(round(d.t, 3), []).append(d)
    keys = np.array(sorted(mains))
    if not len(keys):
        return None
    out = []
    for t, ds in ends.items():
        j = int(np.argmin(np.abs(keys - (t + shift))))
        if abs(keys[j] - (t + shift)) > 0.2:
            continue
        for d in ds:
            p = place(d, cam, P.END_VIEW_RANGE_COV)
            q = [np.hypot(*(p - pm)) for al, pm in mains[keys[j]] if al == d.alliance] if p is not None else []
            if q:
                out.append(min(q))
    return float(np.median(out)) if out else None
