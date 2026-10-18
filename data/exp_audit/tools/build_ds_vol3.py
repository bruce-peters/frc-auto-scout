"""Build the per-scoreboard-jump attribution dataset from stored results (result.json + dets.json).

For every alliance whose scoreboard counter was read from the overlay: each counter jump (video t, balls)
and, for each of the alliance's three robots, the inputs split_increments sees (kernel-weighted launch
evidence, visibility, zone) further split by link kind (box / proj / proj_hidden) and camera view (own end /
other end / main), per-view visibility from dets.json, distance to its hub, plus match-level priors (fuel
OPR, the team's typical video share in its other matches at the event) and the scouting targets (match
shares from scouted shot counts, per-jump shares from kernel-weighted scouted shot times).

TeamSignals are rebuilt from result.json exactly like pipeline.analyse builds them; the old rule
re-run on them is checked against the stored per-team fuel.

  .venv/Scripts/python data/exp_attrmodel/tools/build_ds.py SRC_DIR OUT.pkl
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import attribution as A, config, field as F, opr as O, pipeline as P, scouting, tba  # noqa: E402

TAP = 0.8  # scout tap lands ~0.8 s after the launch
VIEWS = ("own", "oth", "main")


class TL:
    """overlay.Timeline stand-in from result['timeline']."""

    def __init__(self, r):
        t = r["timeline"]
        self.auto_start, self.teleop_start = t["auto_start"], t["teleop_start"]
        self.counts = {k: [tuple(x) for x in v] for k, v in (t.get("counts") or {}).items()}

    def match_time(self, vt):
        if self.auto_start is None:
            return None
        if self.teleop_start is not None and vt >= self.teleop_start:
            return config.AUTO_LEN + (vt - self.teleop_start)
        return vt - self.auto_start

    def video_time(self, mt):
        if mt >= config.AUTO_LEN and self.teleop_start is not None:
            return self.teleop_start + (mt - config.AUTO_LEN)
        return self.auto_start + mt

    def t0(self):
        return (self.auto_start or P.FIRST_AUTO_START) - 1.0

    def t1(self):
        ts = self.teleop_start or ((self.auto_start or P.FIRST_AUTO_START) + P.FIRST_TELEOP_GAP)
        return ts + config.TELEOP_LEN + 5.0


def path_video_times(path, tl: TL) -> np.ndarray:
    """result['paths'][team] rows (match t, x, y) -> (video t, x, y), undoing the auto/teleop pause."""
    mts = [p[0] for p in path]
    jump = next((i for i in range(1, len(mts)) if mts[i] < mts[i - 1] - 0.5), None)
    out = []
    for i, (mt, x, y) in enumerate(path):
        if mt >= config.AUTO_LEN and jump is not None and i < jump:
            vt = tl.auto_start + mt
        else:
            vt = tl.video_time(mt)
        out.append((vt, x, y))
    a = np.array(out) if out else np.zeros((0, 3))
    return a[np.argsort(a[:, 0], kind="stable")] if len(a) else a


PARK_EXT = P.Options().park_ext
DEAD_HIDDEN_S = P.Options().absent_dead_hidden_s
# failure-finder cycle 28 (copy of build_ds.py at 09:44 + the volume feature s_beq)
MIN_SPEED = float(os.environ.get("VOL_MIN_SPEED", "0"))  # m/s: keep only rising tracks at least this fast
VOL_CACHE = Path(os.environ.get("VOL_CACHE", ROOT / "data" / "exp_audit" / "vol_cache"))  # c28_cache.py output


def volley_tracks(key, dets, info):
    """{team: (t0, be)} from the cycle-28 cache (vol_signal.ball_equivalents on the scan with these dets'
    labels, columns t0, be, speed), filtered to speed >= VOL_MIN_SPEED."""
    p = VOL_CACHE / f"{key}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for t in z.files:
        a = z[t]
        a = a[a[:, 2] >= MIN_SPEED] if MIN_SPEED > 0 else a
        out[int(t)] = (a[:, 0], a[:, 1])
    return out


def _vid(rt):
    """Pure-video per-team fuel (the hand-tuned rule): "fuel" is the learned model's estimate once that is on."""
    return rt.get("fuel_video", rt["fuel"])


def _vid_unscaled(rt):
    if "fuel_video" not in rt:
        return rt["fuel_unscaled"]
    return rt["fuel_video"] * rt["fuel_unscaled"] / rt["fuel"] if rt["fuel"] else rt["fuel_video"]


def team_signals(r, info, al, team, tl, dets_by_team, vol=None):
    """attribution.TeamSignals as pipeline.analyse builds it, plus extra per-shot / per-view arrays."""
    ts = A.TeamSignals(team)
    sh = [s for s in r["shots"] if s["team"] == team]
    ts.shots = np.array([s["t"] for s in sh], float)
    ts.shot_w = np.array([s["conf"] for s in sh], float)
    ts.shot_how = np.array([s["how"] or "" for s in sh], dtype=object)
    own = f"{al}_end"
    ts.shot_view = np.array(["main" if s["view"] == "main" else ("own" if s["view"] == own else "oth") for s in sh],
                            dtype=object)
    ts.shot_toward = np.array([{True: 1, False: -1}.get(s["toward"], 0) for s in sh], int)
    pts = path_video_times(r["paths"].get(str(team), []), tl)
    ts.pts = pts
    # parked robot (pipeline Options.park_ext): no PRIOR credit
    ext = P._median_extent(pts) if PARK_EXT > 0 and len(pts) else None
    ts.prior_off = ext is not None and ext < PARK_EXT
    # absent identity state (pipeline Options.absent_cost*): stretches with no segment labelled this team
    # (video time) get no PRIOR/HIDDEN credit; a dead team absent long enough gets no HIDDEN at all
    ts.absent = [(float(a), float(b)) for a, b in (r.get("absent") or {}).get(str(team), [])]
    if (DEAD_HIDDEN_S is not None and team in (r.get("dead") or [])
            and sum(b - a for a, b in ts.absent) >= DEAD_HIDDEN_S):
        ts.no_hidden = True
    grid = np.arange(tl.t0(), tl.t1(), 0.5)
    ts.grid = grid
    if len(pts):
        zv = np.full(len(grid), np.nan)
        vv = np.zeros(len(grid))
        for gi, g in enumerate(grid):
            j = np.searchsorted(pts[:, 0], g)
            near = [pts[jj] for jj in (j - 1, j) if 0 <= jj < len(pts) and abs(pts[jj][0] - g) < 1.0]
            vv[gi] = float(any(abs(pts[jj][0] - g) <= P.VIS_WINDOW for jj in (j - 1, j) if 0 <= jj < len(pts)))
            if near:
                zv[gi] = 1.0 if F.in_alliance_zone(al, near[0][1]) else 0.0
            elif P.FILL_GAPS:
                p = P._track_position(pts, g)
                if p is not None:
                    zv[gi] = 1.0 if F.in_alliance_zone(al, p[0]) else 0.0
        ts.zone_t, ts.zone_v = grid, zv
        ts.vis_t, ts.vis_v = grid, vv
    # per-view sightings on the same grid (a box of this team in that view within VIS_WINDOW)
    ts.view_seen = {}
    for vk in VIEWS:
        tv = dets_by_team.get((team, vk), np.zeros(0))
        if len(tv):
            j = np.clip(np.searchsorted(tv, grid), 1, len(tv) - 1) if len(tv) > 1 else np.zeros(len(grid), int)
            d = np.minimum(np.abs(tv[j] - grid), np.abs(tv[np.maximum(j - 1, 0)] - grid))
            ts.view_seen[vk] = (d <= P.VIS_WINDOW).astype(float)
        else:
            ts.view_seen[vk] = np.zeros(len(grid))
    # distance to own hub and observed/known position on the grid
    hub = F.hub_center(al)
    dist = np.full(len(grid), np.nan)
    if len(pts):
        for gi, g in enumerate(grid):
            p = P._track_position(pts, g)
            if p is not None:
                dist[gi] = float(np.hypot(p[0] - hub[0], p[1] - hub[1]))
    ts.dist = dist
    ts.vol_t, ts.vol_w = (vol or {}).get(team, (np.zeros(0), np.zeros(0)))
    return ts


def win_mean(grid, v, t0, t1, default=np.nan):
    m = (grid >= t0) & (grid <= t1)
    if not m.any():
        return default
    x = v[m]
    x = x[~np.isnan(x)] if np.issubdtype(np.asarray(x).dtype, np.floating) else x
    return float(x.mean()) if len(x) else default


FEATS = ["s_all", "s_box", "s_proj", "s_projh", "s_own", "s_oth", "s_main",
         "c_all", "c_toward", "c_away", "vis", "vis_own", "vis_oth", "vis_main", "zone", "dist", "known",
         "cv_own", "cv_other", "cv_none", "s_mid", "s_match", "pres", "hid_on", "s_beq"]
MID = 10.0  # s: the mid-range launch window extends the kernel window this far either side


def robot_feats(ts: A.TeamSignals, t: float, gate: bool) -> np.ndarray:
    lo, hi = t - A.LAG_HI, t - A.LAG_LO
    if len(ts.shots):
        k = A.kernel(t - ts.shots)
        kw = k * ts.shot_w
        s_all = float(kw.sum())
        s_box = float(kw[ts.shot_how == "box"].sum())
        s_proj = float(kw[ts.shot_how == "proj"].sum())
        s_projh = float(kw[ts.shot_how == "proj_hidden"].sum())
        s_own = float(kw[ts.shot_view == "own"].sum())
        s_oth = float(kw[ts.shot_view == "oth"].sum())
        s_main = float(kw[ts.shot_view == "main"].sum())
        c_all = float(k.sum())
        c_toward = float(k[ts.shot_toward == 1].sum())
        c_away = float(k[ts.shot_toward == -1].sum())
        s_mid = float(ts.shot_w[(ts.shots >= lo - MID) & (ts.shots <= hi + MID)].sum())
        s_match = float(ts.shot_w.sum())
    else:
        s_all = s_box = s_proj = s_projh = s_own = s_oth = s_main = c_all = c_toward = c_away = 0.0
        s_mid = s_match = 0.0
    vis = ts.vis_frac(lo, hi)
    vo = [win_mean(ts.grid, ts.view_seen[vk], lo, hi, 0.0) for vk in VIEWS]
    zone = ts.zone_frac(lo, hi) if gate else 1.0
    dist = win_mean(ts.grid, ts.dist, lo, hi)
    known = win_mean(ts.grid, (~np.isnan(ts.dist)).astype(float), lo, hi, 0.0)
    # exclusive coverage: boxed in its own end view / only in the main or other end view / no box
    own = ts.view_seen["own"]
    other = np.maximum(ts.view_seen["main"], ts.view_seen["oth"]) * (1 - own)
    cv_own = win_mean(ts.grid, own, lo, hi, 0.0)
    cv_other = win_mean(ts.grid, other, lo, hi, 0.0)
    pres = 1.0 - ts.absent_frac(lo, hi) if ts.absent else 1.0  # not in an absent stretch
    s_beq = float((A.kernel(t - ts.vol_t) * ts.vol_w).sum()) if len(ts.vol_t) else 0.0
    return np.array([s_all, s_box, s_proj, s_projh, s_own, s_oth, s_main, c_all, c_toward, c_away,
                     vis, *vo, zone, dist, known, cv_own, cv_other, 1.0 - cv_own - cv_other, s_mid, s_match,
                     pres, 0.0 if ts.no_hidden else 1.0, s_beq], float)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    keys = sorted(p.parent.name for p in src.glob("2026*/result.json"))
    bm = scouting.by_match(scouting.load())
    oprs = {}
    results = {}
    for k in keys:
        results[k] = json.loads((src / k / "result.json").read_text())
    # each team's video share in every stored match (for the typical-share prior, leave-one-match-out)
    vshares = defaultdict(list)  # (event, team) -> [(key, share)]
    infos = {}
    for k, r in results.items():
        info = infos[k] = tba.match(k)
        for al in ("red", "blue"):
            ts_ = info.teams(al)
            try:
                f = np.array([_vid(r["teams"][str(t)]) for t in ts_], float)
            except KeyError:
                continue
            if f.sum() > 0:
                for t, x in zip(ts_, f / f.sum()):
                    vshares[(info.event_key, t)].append((k, float(x)))
    recs = []
    n_bad = 0
    for k in keys:
        r, info = results[k], infos[k]
        ev = info.event_key
        if ev not in oprs:
            oprs[ev] = O.fuel_opr(ev)
        tl = TL(r)
        dets = json.loads((src / k / "dets.json").read_text())
        vol = volley_tracks(k, dets, info)
        print(k, "volley tracks", sum(len(v[0]) for v in vol.values()), flush=True)
        dbt = defaultdict(list)
        for d in dets:
            if d[7] is None:
                continue
            al_t = "red" if d[7] in info.red else "blue"
            vk = "main" if d[1] == "main" else ("own" if d[1] == f"{al_t}_end" else "oth")
            dbt[(d[7], vk)].append(d[0])
        dbt = {kk: np.array(sorted(v)) for kk, v in dbt.items()}
        for al in ("red", "blue"):
            teams = info.teams(al)
            hub = info.hub(al) or {}
            src_al = (r["teams"].get(f"_{al}") or {}).get("source")
            if not tl.counts.get(al) or src_al != "overlay":
                recs.append({"key": k, "event": ev, "al": al, "teams": teams, "overlay": False,
                             "video_share": None})
                continue
            inc = A.increments_from_counts(P._match_counts(tl, al, hub))
            sigs = [team_signals(r, info, al, t, tl, dbt, vol) for t in teams]
            # old rule, to check the reconstruction
            split = A.split_increments(inc, sigs, tl.match_time)
            old = np.array([sum(split[t].values()) for t in teams])
            stored = np.array([_vid_unscaled(r["teams"][str(t)]) for t in teams])
            if np.abs(old - stored).max() > 3.0:
                n_bad += 1
                print(f"  {k} {al}: rebuilt old split {old.round(1)} vs stored {stored}", flush=True)
            T, N, PER, X = [], [], [], []
            for t, n in inc:
                if n <= 0:
                    continue
                mt = tl.match_time(t - 2.0)
                per = A.period_of(mt) if mt is not None else "unknown"
                gate = A.AUTO_ZONE_GATE or per != "auto"
                X.append(np.stack([robot_feats(s, t, gate) for s in sigs]))
                T.append(t)
                N.append(n)
                PER.append(per)
            X = np.array(X) if X else np.zeros((0, 3, len(FEATS)))
            # match-level priors
            o = np.array([oprs[ev].get(t, np.nan) for t in teams])
            typ = np.array([np.mean([x for kk, x in vshares[(ev, t)] if kk != k] or [np.nan]) for t in teams])
            ntyp = np.array([len([1 for kk, _ in vshares[(ev, t)] if kk != k]) for t in teams])
            f = np.array([_vid(r["teams"][str(t)]) for t in teams], float)
            rec = {"key": k, "event": ev, "al": al, "teams": teams, "overlay": True,
                   "t": np.array(T), "n": np.array(N, float), "period": np.array(PER, dtype=object), "X": X,
                   "opr": o, "typ": typ, "ntyp": ntyp, "tba_total": hub.get("totalCount"),
                   "video_share": f / f.sum() if f.sum() > 0 else None, "old_unscaled": old,
                   "prior_off": np.array([s.prior_off for s in sigs], float),
                   "auto_start": tl.auto_start}
            # scouting targets
            sc = bm.get(k, {})
            if all(sc.get(t) for t in teams):
                rec["scout_n"] = np.array([np.mean([x.n for x in sc[t]]) for t in teams])
                timed = [[x for x in sc[t] if x.time_valid] for t in teams]
                if all(timed) and len(T) and tl.auto_start is not None:
                    Q = np.zeros((len(T), 3))
                    for ri, rows in enumerate(timed):
                        for x in rows:
                            launch = tl.auto_start + x.shots - TAP
                            Q[:, ri] += A.kernel(np.array(T)[:, None] - launch[None, :]).sum(1) / len(rows)
                    rec["scout_inc"] = Q
                    rec["scout_n_timed"] = np.array([np.mean([x.n for x in rows]) for rows in timed])
            recs.append(rec)
    with open(out, "wb") as fh:
        pickle.dump({"feats": FEATS, "recs": recs, "src": str(src)}, fh)
    ok = [x for x in recs if x["overlay"]]
    print(f"{len(keys)} matches, {len(recs)} alliances ({len(ok)} with overlay counter); "
          f"{sum('scout_n' in x for x in ok)} fully scouted, {sum('scout_inc' in x for x in ok)} with timed scouting; "
          f"{n_bad} alliances where the rebuilt old split differs from the stored one by > 3 balls -> {out}")


if __name__ == "__main__":
    main()
