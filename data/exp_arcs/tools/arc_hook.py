"""Experiment copy of pipeline.analyse with a hook at the launch-to-box candidate step (pipeline.py is not edited).

build() compiles pipeline.analyse's source with one inserted call, just before the cross-alliance zone rule:

    cands = _ARC_HOOK(...)      # cands: [(score, dist, det, view-local box)]

and installs it in memory as pipeline.analyse_arc. With pipeline._ARC_HOOK = None it is pipeline.analyse.
Hook classes:
  Logger  - records every launch with >= 2 candidate boxes within the link distance (dist <= 0.6): the arc,
            the camera, and per candidate the box, score terms, alliance, team, fused and single-view position;
            returns cands unchanged (so analyse_arc == analyse).
  ArcTerm - adds W * (arc_fit cost - best cost) / SIGMA2 to the score of each close candidate.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import arcs, config, field as fieldmod, pipeline  # noqa: E402

ANCHOR = "                    close = [c for c in cands if c[1] <= 0.6]\n"
CALL = ("                    if _ARC_HOOK is not None:\n"
        "                        cands = _ARC_HOOK(md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm,\n"
        "                                          view_h, lu, lv, u0, v0, vu, vv)\n")


def build():
    src = textwrap.dedent(inspect.getsource(pipeline.analyse))
    assert src.count(ANCHOR) == 1, "anchor not found once in pipeline.analyse"
    src = src.replace(ANCHOR, CALL + ANCHOR).replace("def analyse(", "def analyse_arc(", 1)
    if not hasattr(pipeline, "_ARC_HOOK"):
        pipeline._ARC_HOOK = None
    exec(compile(src, "<analyse_arc>", "exec"), pipeline.__dict__)
    return pipeline.analyse_arc


def _seg_index(segs):
    return {s.id: s for s in segs}


def fused_xy(seg, t: float):
    """Fused (Kalman) position of a segment at the sighting nearest t: (x, y, |dt|)."""
    if seg is None or not seg.points:
        return None
    p = min(seg.points, key=lambda q: abs(q[0] - t))
    return float(p[1]), float(p[2]), float(abs(p[0] - t))


def hub_xyz(alliance: str) -> np.ndarray:
    return np.array([*fieldmod.hub_center(alliance), config.HUB_OPENING_HEIGHT])


def cand_rows(md, tr, cam, cands, t0, det_seg, segs_idx, team_of_seg, u0, v0, vu, vv):
    rows = []
    for score, dist, d, box in cands:
        if dist > 0.6:
            continue
        x0, y0, x1, y1 = box
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        uc = u0
        if pipeline.LINK_X_CROSS and vv < -1e-6:
            uc = u0 + vu * max(-0.3, min(0.0, (y0 - v0) / vv))
        sid = det_seg.get(id(d))
        fx = fused_xy(segs_idx.get(sid), t0) if sid is not None else None
        rows.append({"score": float(score), "dist": float(dist), "box": tuple(float(b) for b in box),
                     "top": float(abs(v0 - y0) / bh), "xc": float(abs(uc - 0.5 * (x0 + x1)) / bw),
                     "area": float((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])),
                     "alliance": d.alliance, "sid": sid, "team": team_of_seg.get(sid) if sid is not None else None,
                     "xy_det": d.xy, "xy_fused": fx[:2] if fx else None, "fused_dt": fx[2] if fx else None,
                     "det_t": float(d.t)})
    return rows


class Logger:
    def __init__(self):
        self.events = []
        self._segs = None

    def __call__(self, md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm, view_h, lu, lv, u0, v0, vu, vv):
        if sum(1 for c in cands if c[1] <= 0.6) < 2:
            return cands
        if self._segs is None or self._segs[0] is not segs:
            self._segs = (segs, _seg_index(segs))
        rows = cand_rows(md, tr, cam, cands, t0, det_seg, self._segs[1], team_of_seg, u0, v0, vu, vv)
        gate = {}
        for score, dist, d, box in cands:
            if dist <= 0.6:
                gate[box] = bool(pipeline._box_shot_ok(tr, box, view_h))
        for r in rows:
            r["gate_ok"] = gate[r["box"]]
        self.events.append({"key": md.key, "view": vname, "t": float(t0), "mt": tl.match_time(t0),
                            "pts": np.asarray(tr.pts, dtype=np.float64), "ppm": float(ppm), "view_h": float(view_h),
                            "cam": None if cam is None else {"K": cam.K, "dist": cam.dist, "rvec": cam.rvec,
                                                             "tvec": cam.tvec, "size": cam.size},
                            "cands": rows})
        return cands


_GATE = {"orig": None, "force": None}


def _gate_wrapper(tr, box, view_h: float = 1e9) -> bool:
    """pipeline._box_shot_ok, except for the box an ArcTerm(gate_base=True) re-ranked to: that launch already passed
    the gate on the 2-D winner, so the shot / no-shot decision stands and only the robot changes."""
    f = _GATE["force"]
    if f is not None and f[0] is tr and f[1] == box:
        return True
    return _GATE["orig"](tr, box, view_h)


def install_gate_wrapper():
    if _GATE["orig"] is None:
        _GATE["orig"] = pipeline._box_shot_ok
        pipeline._box_shot_ok = _gate_wrapper


class ArcTerm:
    """score += W * (cost_c - min cost) / SIGMA2 for each candidate within the link distance.

    pos: "fused" (segment's fused position, falling back to the box's own) or "det" (the box's own).
    prior: "soft" (arcs.SOFT_PRIOR: launch and landing points free inside priors), "hard" (one-parameter fit).
    gate_base: decide shot / no shot with the launch gate on the 2-D winner (as without the term) and let the arc
    term only choose the robot (needs install_gate_wrapper()); otherwise the gate checks the re-ranked winner.
    max_pts: fit only the first max_pts arc points (None = all). min_dur: skip arcs shorter than this (s)."""

    def __init__(self, W=0.3, SIGMA2=1.0, pos="fused", launch_z=arcs.LAUNCH_Z, max_pts=None, min_dur=0.0,
                 cap=None, prior="soft", gate_base=False):
        self.W, self.SIGMA2, self.pos, self.launch_z, self.max_pts, self.min_dur, self.cap = (
            W, SIGMA2, pos, launch_z, max_pts, min_dur, cap)
        self.prior = arcs.SOFT_PRIOR if prior == "soft" else (None if prior in (None, "none", "hard") else prior)
        self.gate_base = bool(gate_base)
        self._segs = None
        self.n_used = 0
        self.n_changed = 0

    def __call__(self, md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm, view_h, lu, lv, u0, v0, vu, vv):
        _GATE["force"] = None
        close = [i for i, c in enumerate(cands) if c[1] <= 0.6]
        if len(close) < 2 or cam is None or tr.pts[-1][0] - tr.pts[0][0] < self.min_dur:
            return cands
        # the pool the pipeline picks from next: its cross-alliance zone rule (OVERLAP_ZONE_RULE), else every box
        pool = list(range(len(cands)))
        if pipeline.OVERLAP_ZONE_RULE and len({cands[i][2].alliance for i in close}) > 1:
            mt_l = tl.match_time(t0)
            if mt_l is not None and mt_l >= config.AUTO_LEN:
                inz = [i for i in close if cands[i][2].xy is not None
                       and fieldmod.in_alliance_zone(cands[i][2].alliance, cands[i][2].xy[0])]
                if inz:
                    pool = inz
        before = min(pool, key=lambda i: cands[i][0])
        if cands[before][1] > 0.6:
            return cands  # no box link either way
        if self.gate_base and not _GATE["orig"](tr, cands[before][3], view_h):
            return cands  # dropped by the gate, as without the term
        sub = [i for i in pool if cands[i][1] <= 0.6]
        if len(sub) < 2:
            return cands
        if self._segs is None or self._segs[0] is not segs:
            self._segs = (segs, _seg_index(segs))
        xys, hubs = [], []
        for i in sub:
            d = cands[i][2]
            xy = None
            if self.pos == "fused":
                sid = det_seg.get(id(d))
                f = fused_xy(self._segs[1].get(sid), t0) if sid is not None else None
                xy = f[:2] if f else None
            if xy is None:
                xy = d.xy
            xys.append(xy)
            hubs.append(hub_xyz(d.alliance))
        fits = arcs.arc_fit(tr.pts, cam, xys, hubs, launch_z=self.launch_z, max_pts=self.max_pts, prior=self.prior)
        c = np.array([f["cost"] if f is not None else np.nan for f in fits])
        if np.isfinite(c).sum() < 2:
            return cands
        term = (c - np.nanmin(c)) / self.SIGMA2
        if self.cap is not None:
            term = np.minimum(term, self.cap)
        term = np.where(np.isfinite(term), term, np.nanmax(term))  # no position: as bad as the worst fit
        out = list(cands)
        for k, i in enumerate(sub):
            s, dist, d, box = cands[i]
            out[i] = (s + self.W * float(term[k]), dist, d, box)
        after = min(pool, key=lambda i: out[i][0])
        self.n_used += 1
        self.n_changed += int(after != before)
        if self.gate_base and after != before:
            _GATE["force"] = (tr, out[after][3])
        return out


class HiddenLogger:
    """Logs every launch the 2-D rule links to a box (winner within the link distance, with a team, passing the
    launch gate) together with (a) the boxes within the link distance and (b) the winner's teammates that have no
    box near the arc in this view (min box dist > HIDDEN_DIST or no box) but a fused position from their track:
    robots hidden behind another one, or not detected in this view. Returns cands unchanged."""

    HIDDEN_DIST = 2.0

    def __init__(self):
        self.events = []
        self._segs = None

    def __call__(self, md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm, view_h, lu, lv, u0, v0, vu, vv):
        if cam is None or not cands:
            return cands
        close = [i for i, c in enumerate(cands) if c[1] <= 0.6]
        pool = list(range(len(cands)))
        if pipeline.OVERLAP_ZONE_RULE and len({cands[i][2].alliance for i in close}) > 1:
            mt_l = tl.match_time(t0)
            if mt_l is not None and mt_l >= config.AUTO_LEN:
                inz = [i for i in close if cands[i][2].xy is not None
                       and fieldmod.in_alliance_zone(cands[i][2].alliance, cands[i][2].xy[0])]
                if inz:
                    pool = inz
        b = min(pool, key=lambda i: cands[i][0])
        if cands[b][1] > 0.6:
            return cands
        gate = _GATE["orig"] or pipeline._box_shot_ok
        if not gate(tr, cands[b][3], view_h):
            return cands
        sid = det_seg.get(id(cands[b][2]))
        team = team_of_seg.get(sid) if sid is not None else None
        if team is None:
            return cands
        if self._segs is None or self._segs[0] is not segs:
            by_team = {}
            for s in segs:
                tm = team_of_seg.get(s.id)
                if tm is not None:
                    by_team.setdefault(tm, []).append(s)
            self._segs = (segs, _seg_index(segs), by_team)
        rows = cand_rows(md, tr, cam, [cands[i] for i in pool], t0, det_seg, self._segs[1], team_of_seg, u0, v0, vu, vv)
        for r in rows:
            r["src"] = "box"
            r["winner"] = r["box"] == tuple(float(x) for x in cands[b][3])
        bdist = {}
        for score, dist, d, box in cands:
            s_ = det_seg.get(id(d))
            tm = team_of_seg.get(s_) if s_ is not None else None
            if tm is not None:
                bdist[tm] = min(bdist.get(tm, np.inf), float(dist))
        al = cands[b][2].alliance
        close_teams = {r["team"] for r in rows}
        for tm in md.info.teams(al):
            if tm in close_teams or bdist.get(tm, np.inf) <= self.HIDDEN_DIST:
                continue
            best = None
            for s in self._segs[2].get(tm, []):
                if s.t0 - 0.6 <= t0 <= s.t1 + 0.6:
                    f = fused_xy(s, t0)
                    if f is not None and f[2] <= 0.6 and (best is None or f[2] < best[2]):
                        best = f
            if best is None:
                continue
            rows.append({"src": "hidden", "team": tm, "alliance": al, "xy_fused": best[:2], "xy_det": None,
                         "fused_dt": best[2], "box_dist": bdist.get(tm, np.inf), "winner": False})
        self.events.append({"key": md.key, "view": vname, "t": float(t0), "mt": tl.match_time(t0),
                            "pts": np.asarray(tr.pts, dtype=np.float64), "ppm": float(ppm),
                            "cam": {"K": cam.K, "dist": cam.dist, "rvec": cam.rvec, "tvec": cam.tvec, "size": cam.size},
                            "winner_team": team, "cands": rows})
        return cands
