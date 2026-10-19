"""Track-every-ball lab: flight-tracker variants on the stored blob logs, and a ball-level launch-recall metric.

Blob logs: data/exp_fueldet/blobs/<key>.pkl (every 30 Hz centre frame's motion blobs + flying-fuel detections,
scripts/fuel_det_experiment.py). Scans / robot boxes / scoreboard: output/<key>/ (read only).

Recall metric (does not reward fragments)
* Sole-shooter bursts: an alliance with all three robots scouted with usable timestamps; a burst of one robot's
  scouted shots (launch = tap - 0.8 s; gaps > 1.5 s split bursts) with no teammate shot within +-EXCL s.
  Truth N = the alliance's scoreboard increase over [b0 - PAD + 0.8, b1 + PAD + 5.5] (pipeline._match_counts,
  increments_from_counts): only that robot can have scored then.
* A track counts for the shooter in a view if it passes fuel.is_shot and, traced back along its fitted parabola
  (<= 0.5 s), crosses the top of the shooter's box there (or starts inside the expanded box), with the
  launch inside [b0 - PAD, b1 + PAD].
* One physical ball once: within a view, arcs whose traced launches agree within 0.035 s and 0.25 box widths are
  one ball (fragments, split pieces); across views, launches within 0.035 s are matched one-to-one.
* recall = unique balls / N, per view (bursts where the shooter is visible there) and by coverage class (own
  end view / main view only / none).

  python data/exp_track/tools/track_lab.py recall --variants greedy ballistic [--matches ...]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "exp_attrmodel" / "tools"))
import numpy as np  # noqa: E402

import build_ds as BD  # noqa: E402
from autoscout import attribution as A, config, field as fieldmod, fuel, pipeline as P, scouting, tba, views  # noqa: E402

BLOBS = Path(os.environ.get("TRACK_BLOBS", str(config.DATA / "exp_fueldet" / "blobs")))  # 60 Hz: data/exp_track/blobs60
OUT = config.DATA / "exp_track"
SNAP = OUT / "snap"  # result.json timeline/teams + dets.json copied from output/ before finalize_v9 rewrote it


def _read(key: str, name: str):
    p = SNAP / key / name
    return json.loads((p if p.exists() else config.OUTPUT / key / name).read_text())
PAD, EXCL = 0.3, 5.5  # PAD: slack on the launch window
EP_GAP, LAG_MIN, LAG_MAX = 2.0, 0.8, 5.0  # counter episodes; launch -> count delay (attribution kernel 0.8-5.5 s)
SAME_T, SAME_U = 0.035, 0.25
TOWARD = True  # count only arcs heading for the alliance's hub (not passes)
CLEAN = (0.5, 2.0)  # bursts whose scoreboard / scouted ratio is in this range (teammates really idle)
W, H = 1920, 1080


# ---------------------------------------------------------------- tracker variants

def cluster_blobs(blobs: list, r: float) -> list:
    """Merge motion blobs whose centres are within r px (fragments of one near ball in the difference mask):
    area-weighted centre, union box, summed area."""
    if r <= 0 or len(blobs) < 2:
        return blobs
    P_ = np.array([(b.u, b.v) for b in blobs])
    parent = list(range(len(blobs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    d = np.sqrt(((P_[:, None] - P_[None]) ** 2).sum(-1))
    ii, jj = np.nonzero(np.triu(d <= r, 1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(blobs)):
        groups[find(i)].append(blobs[i])
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        a = np.array([b.area for b in g])
        u = float((a * [b.u for b in g]).sum() / a.sum()); v = float((a * [b.v for b in g]).sum() / a.sum())
        x0 = min(b.x0 for b in g); y0 = min(b.y0 for b in g)
        x1 = max(b.x0 + b.w for b in g); y1 = max(b.y0 + b.h for b in g)
        out.append(fuel.Blob(u, v, float(a.sum()), x1 - x0, y1 - y0, x0, y0))
    return out


def load_log(key: str) -> dict:
    """A blob log (pickle, or gzip-compressed pickle for the 60 Hz logs)."""
    import gzip
    p = BLOBS / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    with gzip.open(BLOBS / f"{key}.pkl.gz", "rb") as fh:
        return pickle.load(fh)


def _blobs(row, source="motion", conf=0.25, cluster=0.0):
    tc, mb, dd = row
    blobs = [] if source == "det" else [fuel.Blob(*map(float, r)) for r in mb]
    if cluster:
        blobs = cluster_blobs(blobs, cluster)
    if source in ("det", "union"):
        blobs = fuel.merge_detections(blobs, dd, conf)
    return blobs


def _extrap(pts: list, t: float, n: int = 6) -> tuple[float, float, float]:
    """Constant-acceleration extrapolation of the last n points to time t: (u, v, speed px/s)."""
    p = np.array(pts[-n:])
    tt = p[:, 0] - p[-1, 0]
    h = t - p[-1, 0]
    if len(p) >= 4:
        cu = np.polyfit(tt, p[:, 1], 2); cv = np.polyfit(tt, p[:, 2], 2)
        cu[0] = 0.0  # no horizontal acceleration
    else:
        cu = np.r_[0.0, np.polyfit(tt, p[:, 1], 1)]; cv = np.r_[0.0, np.polyfit(tt, p[:, 2], 1)]
    sp = float(np.hypot(cu[1], cv[1]))
    return float(np.polyval(cu, h)), float(np.polyval(cv, h)), sp


def _junction_res(a: list, b: list, n: int = 6) -> float:
    p = np.array(a[-n:] + b[:n])
    t = p[:, 0] - p[0, 0]
    ru = p[:, 1] - np.polyval(np.polyfit(t, p[:, 1], 1), t)
    rv = p[:, 2] - np.polyval(np.polyfit(t, p[:, 2], 2), t)
    return float(np.sqrt(np.mean(ru ** 2 + rv ** 2)))


def stitch(tracks: list, max_gap: float = 0.35, tol_px: float = 5.0, tol_k: float = 0.15, max_res: float = 4.0,
           min_len: int = 3) -> list:
    """Join tracklets that continue one ballistic path: B follows A if A's constant-acceleration extrapolation to
    B's first sighting lands within tol_px + tol_k * (A's speed * gap) of it (the next ball on the same path is a
    full spacing behind, so it is rejected) and the fit across the junction stays smooth. One-to-one, best first."""
    T = [tr for tr in tracks if len(tr.pts) >= min_len]
    starts = np.array([tr.pts[0][0] for tr in T])
    order = np.argsort(starts)
    T = [T[i] for i in order]
    starts = starts[order]
    pairs = []
    for i, a in enumerate(T):
        ta = a.pts[-1][0]
        j0, j1 = np.searchsorted(starts, ta + 1e-6), np.searchsorted(starts, ta + max_gap + 1e-6)
        for j in range(j0, j1):
            b = T[j]
            u, v, sp = _extrap(a.pts, b.pts[0][0])
            err = float(np.hypot(u - b.pts[0][1], v - b.pts[0][2]))
            if err > tol_px + tol_k * sp * (b.pts[0][0] - ta):
                continue
            res = _junction_res(a.pts, b.pts)
            if res <= max_res:
                pairs.append((err + res, i, j))
    nxt, prv = {}, {}
    for _, i, j in sorted(pairs):
        if i in nxt or j in prv:
            continue
        nxt[i], prv[j] = j, i
    out = []
    for i in range(len(T)):
        if i in prv:
            continue
        pts = list(T[i].pts)
        k = i
        while k in nxt:
            k = nxt[k]
            pts += T[k].pts
        out.append(fuel.FlightTrack(T[i].id, pts, done=True))
    return out


VARIANTS = {
    "greedy": dict(tracker="greedy"),
    "greedy+split": dict(tracker="greedy", post=["split"]),
    "ballistic": dict(tracker="ballistic"),
    "ballistic+split": dict(tracker="ballistic", post=["split"]),
    "ballistic+stitch": dict(tracker="ballistic", tk=dict(min_len=3), post=["stitch"]),
    "ballistic+split+stitch": dict(tracker="ballistic", tk=dict(min_len=3), post=["split", "stitch"]),
    "greedy+split+stitch": dict(tracker="greedy", post=["split", "stitch"]),
    # fragments of one near ball merged before association (prior-art note 2)
    "ballistic+cl10+split": dict(tracker="ballistic", cluster=10.0, post=["split"]),
    "ballistic+cl16+split": dict(tracker="ballistic", cluster=16.0, post=["split"]),
    "greedy+cl10+split": dict(tracker="greedy", cluster=10.0, post=["split"]),
    # unsplit versions for the scouting A/B (the pipeline splits stored tracks itself since SPLIT_TRACKS=True)
    "ballistic+cl10": dict(tracker="ballistic", cluster=10.0),
}


def jump_windows(md, w: float) -> np.ndarray:
    """[(t0, t1)] spans of W s before every scoreboard increase (both alliances, raw overlay reads)."""
    ts = []
    for al, counts in (getattr(md.timeline, "counts", None) or {}).items():
        last = None
        for t, c in counts:
            if last is not None and c > last:
                ts.append(t)
            last = c if last is None or c > last else last
    ts = np.array(sorted(ts))
    return np.stack([ts - w, ts], axis=1) if len(ts) else np.zeros((0, 2))


def hybrid_rows(rows: list, windows: np.ndarray) -> tuple[list, float]:
    """A 60 Hz log thinned to 30 Hz outside the windows (its even frames are the 30 Hz analysis); share kept at 60."""
    out, n60 = [], 0
    for row in rows:
        t = row[0]
        inside = bool(len(windows)) and bool(np.any((windows[:, 0] <= t) & (windows[:, 1] >= t)))
        if inside:
            out.append(row)
            n60 += 1
        elif int(round(t * 60)) % 2 == 0:
            out.append(row)
    return out, n60 / max(1, len(rows))


def run(name: str, blob_log: dict, md=None, fuel_step: int = 2) -> dict:
    """Tracks per view for a named variant. Variant names may end in '@hybW' (60 Hz logs only): 60 Hz in the W s
    before each scoreboard increase, 30 Hz elsewhere."""
    base, _, hyb = name.partition("@hyb")
    cfg = VARIANTS[base]
    windows = jump_windows(md, float(hyb)) if hyb else None
    out = {}
    for vn, rows in blob_log.items():
        if windows is not None:
            rows, share = hybrid_rows(rows, windows)
            run.share60 = share
        if cfg["tracker"] == "greedy":
            tr = fuel.FlightTracker(gate=45.0 * fuel_step ** 0.7)
        else:
            # (fuel.BallisticTracker scales its per-frame seeding distances to the frame step itself, 60 Hz too)
            tr = fuel.BallisticTracker(**cfg.get("tk", {}))
        for row in rows:
            tr.update(row[0], _blobs(row, cfg.get("source", "motion"), cfg.get("conf", 0.25), cfg.get("cluster", 0.0)))
        tracks = tr.flush()
        for step in cfg.get("post", []):
            if step == "split":
                tracks = fuel.split_at_jumps(tracks, min_len=cfg.get("tk", {}).get("min_len", 4))
            elif step == "stitch":
                tracks = stitch(tracks)
            elif callable(step):
                tracks = step(tracks)
        out[vn] = tracks
    return out


# ---------------------------------------------------------------- recall metric

def sole_bursts(key: str, r: dict, info, bym: dict) -> list[dict]:
    """Scoring episodes with one shooter: the alliance counter's increments grouped into episodes (gaps > EP_GAP
    s); an episode is a sole-shooter burst when exactly one of the three (all scouted, timed) robots has scouted
    launches (tap - 0.8 s) in [c0 - 6, c1 + 0.5] and the other two have none in [c0 - EXCL - 2, c1 + 2].
    N = the episode's balls; arcs are counted with launches in [c0 - LAG_MAX, c1 - LAG_MIN] (b0, b1)."""
    tl = BD.TL(r)
    a0 = tl.auto_start
    out = []
    if a0 is None:
        return out
    for al in ("red", "blue"):
        teams = info.teams(al)
        rows = {t: [x for x in bym.get(key, {}).get(t, []) if x.time_valid] for t in teams}
        if not all(rows.values()) or not tl.counts.get(al):
            continue
        act = {t: np.sort(np.concatenate([a0 + x.shots - 0.8 for x in rows[t]])) for t in teams}
        inc = [(t, n) for t, n in A.increments_from_counts(P._match_counts(tl, al, info.hub(al))) if n > 0]
        if not inc:
            continue
        it = np.array([x[0] for x in inc])
        inn = np.array([x[1] for x in inc])
        for ep in np.split(np.arange(len(it)), np.where(np.diff(it) > EP_GAP)[0] + 1):
            c0, c1 = float(it[ep[0]]), float(it[ep[-1]])
            n = float(inn[ep].sum())
            active = [t for t in teams if np.any((act[t] >= c0 - 6.0) & (act[t] <= c1 + 0.5))]
            if len(active) != 1:
                continue
            t = active[0]
            if any(np.any((act[o] >= c0 - EXCL - 2.0) & (act[o] <= c1 + 2.0)) for o in teams if o != t):
                continue
            sc = float(np.mean([((a0 + x.shots - 0.8 >= c0 - 6.0) & (a0 + x.shots - 0.8 <= c1 + 0.5)).sum()
                                for x in rows[t]]))
            out.append({"team": t, "al": al, "b0": c0 - LAG_MAX, "b1": c1 - LAG_MIN, "c0": c0, "c1": c1,
                        "N": n, "scouted": sc})
    return out


def _fit(tr, n: int = 10):
    p = np.array(tr.pts[:n])
    t = p[:, 0] - p[0, 0]
    bu = np.polyfit(t, p[:, 1], 1) if len(p) >= 2 else np.array([0.0, p[0, 1]])
    bv = np.polyfit(t, p[:, 2], 2) if len(p) >= 3 else np.array([0.0, *np.polyfit(t, p[:, 2], 1)])
    return bu, bv


def launch_link(tr, box, max_back: float = 0.5):
    """(launch video time, u where it crossed the box top) if the arc rose out of this box, else None."""
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    u0, v0 = tr.pts[0][1], tr.pts[0][2]
    if x0 - 0.25 * bw <= u0 <= x1 + 0.25 * bw and y0 - 0.9 * bh <= v0 <= y1:
        return tr.pts[0][0], u0
    if v0 > y0:  # first seen below the box top but outside the box
        return None
    bu, bv = _fit(tr)
    c2, c1, c0 = bv
    roots = np.roots([c2, c1, c0 - y0]) if abs(c2) > 1e-9 else (np.array([-(c0 - y0) / c1]) if abs(c1) > 1e-9 else [])
    best = None
    for rt in np.atleast_1d(roots):
        if abs(np.imag(rt)) > 1e-9:
            continue
        tau = float(np.real(rt))
        if -max_back <= tau <= 0 and (2 * c2 * tau + c1) < 0:  # rising when it crossed the box top
            if best is None or tau > best:
                best = tau
    if best is None:
        return None
    uc = float(np.polyval(bu, best))
    if x0 - 0.25 * bw <= uc <= x1 + 0.25 * bw:
        return tr.pts[0][0] + best, uc
    return None


def _unique(launches: list[tuple[float, float]], bw: float) -> list[tuple[float, float]]:
    out = []
    for t, u in sorted(launches):
        if out and t - out[-1][0] <= SAME_T and abs(u - out[-1][1]) <= SAME_U * bw:
            continue
        out.append((t, u))
    return out


def _match_union(per_view: dict[str, list[float]]) -> int:
    """Balls seen in any view: launches of different views within SAME_T s matched one-to-one."""
    pool = sorted((t, v) for v, ts in per_view.items() for t in ts)
    used = [False] * len(pool)
    n = 0
    for i, (t, v) in enumerate(pool):
        if used[i]:
            continue
        used[i] = True
        n += 1
        seen = {v}
        for j in range(i + 1, len(pool)):
            if pool[j][0] - t > SAME_T:
                break
            if not used[j] and pool[j][1] not in seen:
                used[j] = True
                seen.add(pool[j][1])
    return n


def burst_counts(key: str, tracks_all: dict, md, r: dict, info, bursts: list[dict]) -> list[dict]:
    dets = _read(key, "dets.json")
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    boxes = defaultdict(list)  # (team, view) -> [(t, x0, y0, x1, y1)] view-local
    for d in dets:
        if d[7] is None or d[1] not in vl:
            continue
        ox, oy = vl[d[1]].offset(W, H)
        boxes[(d[7], d[1])].append((d[0], d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy))
    boxes = {k: np.array(sorted(v)) for k, v in boxes.items()}
    shot_cache, toward = {}, {}
    for vn, tracks in tracks_all.items():
        cam_track = md.cams.get(vn)
        lst = []
        for tr in tracks:
            if len(tr.pts) < 4:
                continue
            cam = cam_track.at(tr.pts[0][0]) if cam_track is not None else None
            ppm = P._px_per_m(cam, tr.pts[0][1], tr.pts[0][2])
            if fuel.is_shot(tr, ppm):
                lst.append(tr)
                # heads for a hub (the pipeline's "toward" test): passes / feeds do not
                for al_ in ("red", "blue"):
                    ok = True
                    if cam is not None:
                        hu, hv = cam.project(np.array([[*fieldmod.hub_center(al_), 1.9]]))[0]
                        d0 = np.hypot(tr.pts[0][1] - hu, tr.pts[0][2] - hv)
                        d1 = np.hypot(tr.pts[-1][1] - hu, tr.pts[-1][2] - hv)
                        ok = bool(d1 < d0 - 0.3 * ppm)
                    toward[(vn, len(lst) - 1, al_)] = ok
        shot_cache[vn] = (lst, np.array([tr.pts[0][0] for tr in lst]))
    teams_in = {vn: sorted({t for (t, v) in boxes if v == vn}) for vn in tracks_all}

    def box_at(team, vn, t):
        bx = boxes.get((team, vn))
        if bx is None or not len(bx):
            return None
        j = np.searchsorted(bx[:, 0], t)
        cand = [k for k in (j - 1, j) if 0 <= k < len(bx) and abs(bx[k, 0] - t) <= 0.3]
        return bx[min(cand, key=lambda k: abs(bx[k, 0] - t)), 1:] if cand else None

    owner_cache = {}

    def owner(vn, i):
        """Exclusive link like the pipeline's: the robot whose box top the arc crossed nearest its centre."""
        key_ = (vn, i)
        if key_ not in owner_cache:
            tr = shot_cache[vn][0][i]
            best = None
            for t in teams_in[vn]:
                box = box_at(t, vn, tr.pts[0][0])
                if box is None:
                    continue
                lk = launch_link(tr, box)
                if lk is None:
                    continue
                cost = abs(lk[1] - 0.5 * (box[0] + box[2])) / max(box[2] - box[0], 1.0)
                if best is None or cost < best[0]:
                    best = (cost, t, lk, box[2] - box[0])
            owner_cache[key_] = best
        return owner_cache[key_]

    def robot(team, al, b0, b1):
        """Unique balls credited (exclusively) to this robot's boxes in each view during the burst window."""
        vis, cnt, per_view = {}, {}, {}
        for vn in tracks_all:
            bx = boxes.get((team, vn))
            if bx is None or not len(bx):
                vis[vn] = 0.0
                continue
            m = (bx[:, 0] >= b0 - 0.5) & (bx[:, 0] <= b1 + 0.5)
            vis[vn] = float(min(1.0, m.sum() / max(1.0, (b1 - b0 + 1.0) * 10)))
            lst, t0s = shot_cache[vn]
            sel = np.nonzero((t0s >= b0 - PAD - 0.5) & (t0s <= b1 + PAD + 0.5))[0]
            launches, bws = [], []
            for i in sel:
                ow = owner(vn, i)
                if ow is None or ow[1] != team:
                    continue
                if TOWARD and not toward.get((vn, i, al), True):
                    continue
                lk = ow[2]
                if not (b0 - PAD <= lk[0] <= b1 + PAD):
                    continue
                launches.append(lk)
                bws.append(ow[3])
            uniq = _unique(launches, float(np.median(bws)) if bws else 50.0)
            per_view[vn] = [t for t, _ in uniq]
            cnt[vn] = {"arcs": len(launches), "unique": len(uniq)}
        mo = {vn: x for vn, x in per_view.items() if vn in ("main", f"{al}_end")}
        return {"visible": vis, "views": cnt, "union": _match_union(per_view), "union_mo": _match_union(mo)}

    out = []
    for bst in bursts:
        b0, b1, team, al = bst["b0"], bst["b1"], bst["team"], bst["al"]
        rec = {**bst, **robot(team, al, b0, b1)}
        # control: the two teammates were not shooting (sole-shooter burst) - anything credited to them is false
        rec["idle"] = [{"team": t, **robot(t, al, b0, b1)} for t in info.teams(al) if t != team]
        out.append(rec)
    return out


def coverage(rec: dict) -> str:
    own = f"{rec['al']}_end"
    vis = rec["visible"]
    if vis.get(own, 0) >= 0.5:
        return "own end"
    if vis.get("main", 0) >= 0.5:
        return "main only"
    return "none"


def summarize(recs: list[dict]) -> dict:
    """Ball-weighted recall (shooter) and false launches (idle teammates) per view kind and coverage class."""
    acc = defaultdict(lambda: np.zeros(4))  # N, unique, arcs, idle unique (same views, idle robot visible)
    for r in recs:
        if r["N"] < 3 or not (CLEAN[0] <= r["N"] / max(r["scouted"], 1e-9) <= CLEAN[1]):
            continue
        for vn, c in r["views"].items():
            if r["visible"].get(vn, 0) < 0.5:
                continue
            kind = "main" if vn == "main" else ("own end" if vn == f"{r['al']}_end" else "other end")
            acc[("view", kind)] += (r["N"], c["unique"], c["arcs"], 0)
        for x in r.get("idle", []):
            for vn, c in x["views"].items():
                if x["visible"].get(vn, 0) < 0.5:
                    continue
                kind = "main" if vn == "main" else ("own end" if vn == f"{r['al']}_end" else "other end")
                acc[("idleview", kind)] += (r["N"], c["unique"], 0, 0)
        cls = coverage(r)
        idle_mo = sum(x["union_mo"] for x in r.get("idle", []))
        for c in (cls, "all"):
            acc[("cov", c)] += (r["N"], r["union_mo"], r["union"], idle_mo)
        size = "3-9" if r["N"] < 10 else ("10-19" if r["N"] < 20 else "20+")
        acc[("size", size)] += (r["N"], r["union_mo"], r["union"], idle_mo)
    out = {}
    for (a, b), x in acc.items():
        out[f"{a}|{b}"] = {"N": float(x[0]), "recall": float(x[1] / x[0]) if x[0] else float("nan"),
                           "arcs_per_ball": float(x[2] / x[0]) if x[0] else float("nan"),
                           "idle_per_ball": float(x[3] / x[0]) if x[0] else float("nan")}
    return out


def cmd_recall(keys: list[str], variants: list[str], tag: str):
    bym = scouting.by_match(scouting.load())
    for key in keys:
        todo = [v for v in variants if not (OUT / "recall" / tag / v / f"{key}.json").exists()]
        if not todo:
            continue
        tic = time.time()
        r = _read(key, "result.json")
        info = tba.match(key)
        bursts = sole_bursts(key, r, info, bym)
        if not bursts:
            for v in todo:
                d = OUT / "recall" / tag / v
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{key}.json").write_text("[]")
            continue
        log = load_log(key)
        with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        for v in todo:
            tracks = run(v, log["blob_log"], md, log["fuel_step"])
            recs = burst_counts(key, tracks, md, r, info, bursts)
            d = OUT / "recall" / tag / v
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{key}.json").write_text(json.dumps(recs))
        del md, log
        print(f"{key}: {len(bursts)} sole-shooter bursts, {', '.join(todo)} ({time.time() - tic:.0f}s)", flush=True)


def cmd_report(variants: list[str], tag: str, keys: list[str] | None = None):
    print(f"ball-level launch recall on sole-shooter bursts: unique balls / scoreboard balls (results {tag})")
    print("  view X: shooter visible in that view; 'false' = launches credited to the idle teammates, per scored ball")
    print("  cov: balls seen in the main or the own end view (one-to-one across the two), by which camera sees the shooter")
    cols = [("view", "own end"), ("view", "main"), ("cov", "own end"), ("cov", "main only"), ("cov", "none"),
            ("cov", "all"), ("size", "3-9"), ("size", "10-19"), ("size", "20+")]
    print(f"{'variant':24s} " + " ".join(f"{a + ' ' + b:>16s}" for a, b in cols))
    rep = {}
    for v in variants:
        recs = []
        for f in sorted((OUT / "recall" / tag / v).glob("*.json")):
            if keys is None or f.stem in keys:
                recs += json.loads(f.read_text())
        s = summarize(recs)
        rep[v] = s
        cells = []
        for a, b in cols:
            x = s.get(f"{a}|{b}")
            if not x:
                cells.append("-")
            elif a == "view":
                fi = s.get(f"idleview|{b}")
                cells.append(f"{x['recall']:.2f} f{fi['recall'] if fi else float('nan'):.2f}")
            else:
                cells.append(f"{x['recall']:.2f} f{x['idle_per_ball']:.2f}")
        n = [r for r in recs if r["N"] >= 3 and CLEAN[0] <= r["N"] / max(r["scouted"], 1e-9) <= CLEAN[1]]
        print(f"{v:24s} " + " ".join(f"{c:>16s}" for c in cells) + f"   ({len(n)} bursts, {sum(r['N'] for r in n):.0f} balls)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recall", "report"])
    ap.add_argument("--variants", nargs="*", default=["greedy", "ballistic"])
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    bym = scouting.by_match(scouting.load())
    keys = a.matches or sorted(k for k in {p.name.split(".")[0] for p in BLOBS.glob("2026cc_qm*.pkl*")}
                               if k in bym and k != "2026cc_qm14")
    if a.cmd == "recall":
        si, sn = map(int, a.shard.split("/"))
        cmd_recall(keys[si::sn], a.variants, a.tag)
    else:
        cmd_report(a.variants, a.tag, a.matches)


if __name__ == "__main__":
    main()
