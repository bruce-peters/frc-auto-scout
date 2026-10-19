"""Pass 3: A/B of the two-view relink on the scouted Chezy quals (output/ is not written; pipeline.py is not edited).

build() compiles pipeline.analyse with one inserted call just before the "no robot found" exit of the launch loop:

    team, al, conf, how = _MV_HOOK(md, vname, tr, t0, team, al, conf, how, team_arr, alliance_of)

and installs it in memory as pipeline.analyse_mv (with _MV_HOOK = None it is pipeline.analyse). The hook sees
every launch the 2-D rules credit (box or projection link, after the launch gate), so whether a launch counts is
decided as today and only the robot can change. Relink: if the launch track has an accepted two-view fit
(pass-1 pairs, common.best_candidate under RULE), the robot is the one triangulate.pick_robot picks among the
match's robots (fused team tracks at the launch time, team_arr, any view, boxes not needed); otherwise unchanged.

Arms (one scan at a time, all arms in the same process): base (hook off) and each --arms entry.
Scored with scripts/score_scouting.score, paired bootstrap over matches (ab_arcs.paired).

  .venv/Scripts/python -u data/exp_multiview/tools/ab_mv.py --arms "mv:rule=norm2" [--score-only]
"""
from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data" / "exp_arcs" / "tools"))
import common as C  # noqa: E402
from autoscout import config, field as fieldmod, pipeline, scouting, triangulate as T  # noqa: E402

ANCHOR = "            if team is None and al is None:\n                continue\n"
CALL = ("            if _MV_HOOK is not None:\n"
        "                team, al, conf, how = _MV_HOOK(md, vname, tr, t0, team, al, conf, how, team_arr, alliance_of)\n")

RULES = {
    "norm2": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 2.0},
    "norm1.5r": {"sigma": {"main": 8.0, "end": 4.0}, "max_norm": 1.5, "ratio": 1.5},
    "rms10": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 10},
    "rms6": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 6},
    "rms15": {"sigma": {"main": 1.0, "end": 1.0}, "max_norm": 15},
    "any": {},
}


def build():
    src = textwrap.dedent(inspect.getsource(pipeline.analyse))
    assert src.count(ANCHOR) == 1, "anchor not found once in pipeline.analyse"
    src = src.replace(ANCHOR, CALL + ANCHOR).replace("def analyse(", "def analyse_mv(", 1)
    if not hasattr(pipeline, "_MV_HOOK"):
        pipeline._MV_HOOK = None
    exec(compile(src, "<analyse_mv>", "exec"), pipeline.__dict__)
    return pipeline.analyse_mv


class Relink:
    """The hook. pairs: pass-1 pairs of this match; rule: common.best_candidate rule; same_alliance: only robots of
    the 2-D link's alliance may be picked; gate/margin: triangulate.pick_robot."""

    def __init__(self, pairs, rule, same_alliance=False, gate=T.GATE_CHI2, margin=T.MARGIN_CHI2,
                 sigma_robot=T.SIGMA_ROBOT, only_how=None):
        self.ix = C.anchor_index(pairs)
        self.rule, self.same_alliance = rule, same_alliance
        self.gate, self.margin, self.sigma_robot = gate, margin, sigma_robot
        self.only_how = only_how
        self.n_seen = self.n_fit = self.n_picked = self.n_changed = 0
        self.changes = []

    def __call__(self, md, vname, tr, t0, team, al, conf, how, team_arr, alliance_of):
        if team is None and al is None:
            return team, al, conf, how
        if self.only_how and how not in self.only_how:
            return team, al, conf, how
        self.n_seen += 1
        a = self.ix.get(C.anchor_key(vname, tr.pts))
        c = C.best_candidate(a, self.rule) if a is not None else None
        if c is None or c["launch"] is None:
            return team, al, conf, how
        self.n_fit += 1
        L = c["launch"]
        rob = {}
        for tm, arr in team_arr.items():
            if self.same_alliance and al is not None and alliance_of[tm] != al:
                continue
            p = pipeline._track_position(arr, L["t"])
            if p is not None:
                rob[tm] = (p[0], p[1])
        if not rob:
            return team, al, conf, how
        r = T.pick_robot(L, rob, sigma_robot=self.sigma_robot, gate=self.gate, margin=self.margin)
        if r["team"] is None:
            return team, al, conf, how
        self.n_picked += 1
        nt = r["team"]
        if nt != team:
            self.n_changed += 1
            self.changes.append((round(float(t0), 3), vname, team, nt, how))
        return nt, alliance_of[nt], conf, (how or "") + "+mv"


class SoftTerm:
    """arc_hook-style hook (pipeline.analyse_arc, at the launch-to-box candidate step): for a launch with >= 2
    boxes within the link distance and an accepted two-view fit, score_i += lam * (d2_i - min d2), d2_i = the
    Mahalanobis^2 of box i's robot (its segment's fused position at the launch time, else the box's own) from the
    launch point. The launch gate is decided on today's 2-D winner and the term only picks the robot
    (arc_hook.install_gate_wrapper), as recommended for the arc term in data/exp_arcs."""

    def __init__(self, pairs, rule, lam=0.01, sigma_robot=T.SIGMA_ROBOT):
        import arc_hook
        self.ah = arc_hook
        self.ix = C.anchor_index(pairs)
        self.rule, self.lam, self.sigma_robot = rule, lam, sigma_robot
        self._segs = None
        self.n_used = self.n_changed = 0
        self.changes = []

    def __call__(self, md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm, view_h, lu, lv, u0, v0, vu, vv):
        ah = self.ah
        ah._GATE["force"] = None
        close = [i for i, c in enumerate(cands) if c[1] <= 0.6]
        if len(close) < 2:
            return cands
        pool = list(range(len(cands)))
        if pipeline.OVERLAP_ZONE_RULE and len({cands[i][2].alliance for i in close}) > 1:
            mt_l = tl.match_time(t0)
            if mt_l is not None and mt_l >= config.AUTO_LEN:
                inz = [i for i in close if cands[i][2].xy is not None
                       and fieldmod.in_alliance_zone(cands[i][2].alliance, cands[i][2].xy[0])]
                if inz:
                    pool = inz
        before = min(pool, key=lambda i: cands[i][0])
        if cands[before][1] > 0.6 or not ah._GATE["orig"](tr, cands[before][3], view_h):
            return cands
        sub = [i for i in pool if cands[i][1] <= 0.6]
        if len(sub) < 2:
            return cands
        a = self.ix.get(C.anchor_key(vname, tr.pts))
        c = C.best_candidate(a, self.rule) if a is not None else None
        if c is None or c["launch"] is None:
            return cands
        L = c["launch"]
        if self._segs is None or self._segs[0] is not segs:
            self._segs = (segs, {s.id: s for s in segs})
        Si = np.linalg.inv(L["cov"] + self.sigma_robot ** 2 * np.eye(2))
        d2 = []
        for i in sub:
            d = cands[i][2]
            sid = det_seg.get(id(d))
            f = ah.fused_xy(self._segs[1].get(sid), L["t"]) if sid is not None else None
            xy = f[:2] if f else d.xy
            if xy is None:
                d2.append(np.nan)
                continue
            r = np.asarray(xy[:2], float) - L["xy"]
            d2.append(float(r @ Si @ r))
        d2 = np.array(d2)
        if not np.isfinite(d2).any():
            return cands
        d2 = np.where(np.isfinite(d2), d2, np.nanmax(d2))
        out = list(cands)
        for k, i in enumerate(sub):
            sc, dist, d, box = cands[i]
            out[i] = (sc + self.lam * float(d2[k] - d2.min()), dist, d, box)
        after = min(pool, key=lambda i: out[i][0])
        self.n_used += 1
        if after != before:
            self.n_changed += 1
            ah._GATE["force"] = (tr, out[after][3])
            self.changes.append((round(float(t0), 3), vname))
        return out


def parse(spec):
    out = {}
    for item in spec.split(";"):
        name, _, kv = item.partition(":")
        kw = {}
        for p in filter(None, kv.split(",")):
            k, _, v = p.partition("=")
            kw[k] = v
        out[name] = kw
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="mv:rule=norm2")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--out", default="ab")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    arms = parse(a.arms)
    OUT = C.EXP / a.out
    keys = a.matches or sorted(k for k in scouting.by_match(scouting.load())
                               if (config.OUTPUT / k / "scan.pkl").exists() and (C.EXP / "pairs" / f"{k}.pkl").exists())
    if not a.score_only:
        import hashlib
        from autoscout import attribution
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
        am = attribution.ATTR_MODEL
        amp = Path(am) if am and Path(am).is_absolute() else ROOT / str(am)
        print("pipeline.py", sha(ROOT / "autoscout" / "pipeline.py"), "triangulate.py", sha(ROOT / "autoscout" / "triangulate.py"),
              "credit model", am, sha(amp) if amp.exists() else None, flush=True)
        build()
        import arc_hook  # noqa: E402
        arc_hook.build()
        arc_hook.install_gate_wrapper()
        print(len(keys), "scouted matches", flush=True)
        stats = {n: [0, 0, 0, 0] for n in arms}
        changes = {n: {} for n in arms}
        tic = time.time()
        for k in keys:
            with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
                raw = fh.read()
            md = pickle.loads(raw)
            del raw
            pipeline.load_dino(md, config.OUTPUT / k)
            pairs_by = {}
            for kw in arms.values():
                pd = kw.get("pairs", "pairs")
                if pd not in pairs_by:
                    C.PAIRS_DIR = pd
                    pairs_by[pd] = C.load_pairs(k)
            C.PAIRS_DIR = "pairs"
            pipeline._MV_HOOK = None
            res = pipeline.analyse_mv(md, pipeline.Options())
            od = OUT / "base" / k
            od.mkdir(parents=True, exist_ok=True)
            (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
            for name, kw in arms.items():
                if kw.get("soft"):
                    pairs = pairs_by[kw.get("pairs", "pairs")]
                    hook = SoftTerm(pairs, RULES[kw.get("rule", "norm2")], lam=float(kw["soft"]))
                    pipeline._ARC_HOOK = hook
                    res = pipeline.analyse_arc(md, pipeline.Options())
                    pipeline._ARC_HOOK = None
                    arc_hook._GATE["force"] = None
                    hook.n_seen, hook.n_fit, hook.n_picked = hook.n_used, hook.n_used, hook.n_used
                else:
                    pairs = pairs_by[kw.get("pairs", "pairs")]
                    hook = Relink(pairs, RULES[kw.get("rule", "norm2")], same_alliance=kw.get("same_alliance") == "1",
                                  gate=float(kw.get("gate", T.GATE_CHI2)), margin=float(kw.get("margin", T.MARGIN_CHI2)),
                                  sigma_robot=float(kw.get("sigma_robot", T.SIGMA_ROBOT)),
                                  only_how=set(kw["only_how"].split("|")) if kw.get("only_how") else None)
                    pipeline._MV_HOOK = hook
                    res = pipeline.analyse_mv(md, pipeline.Options())
                    pipeline._MV_HOOK = None
                od = OUT / name / k
                od.mkdir(parents=True, exist_ok=True)
                (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
                s = stats[name]
                s[0] += hook.n_seen
                s[1] += hook.n_fit
                s[2] += hook.n_picked
                s[3] += hook.n_changed
                changes[name][k] = hook.changes
            print(k, f"{time.time() - tic:.0f}s", {n: tuple(s) for n, s in stats.items()}, flush=True)
            del md, res
        json.dump({"stats(seen, with fit, picked, changed)": stats, "changes": changes}, open(OUT / "hook_stats.json", "w"))

    from score_scouting import score  # noqa: E402
    import ab_arcs  # noqa: E402

    base = score(OUT / "base", keep_rows=True)
    b = base
    print(f"base: video only MAE {b['vonly']['mae']:.2f} corr {b['vonly']['corr']:.3f} top {100 * b['top_scorer']['vonly']:.0f} % | "
          f"best estimate MAE {b['video']['mae']:.2f} corr {b['video']['corr']:.3f} top {100 * b['top_scorer']['video']:.0f} % "
          f"| burst {100 * b['burst_overlap']['video']:.1f} % | {b['matches']} matches", flush=True)
    out = {"base": {k: v for k, v in base.items() if k != "rows"}}
    for name in arms:
        s = score(OUT / name, keep_rows=True)
        d, n, better, worse = ab_arcs.paired(base, s)
        out[name] = {"score": {k: v for k, v in s.items() if k != "rows"}, "paired": d, "n": n, "better": better, "worse": worse}
        print(f"{name}: video only MAE {s['vonly']['mae']:.2f} corr {s['vonly']['corr']:.3f} top {100 * s['top_scorer']['vonly']:.0f} % | "
              f"best estimate MAE {s['video']['mae']:.2f} corr {s['video']['corr']:.3f} top {100 * s['top_scorer']['video']:.0f} % "
              f"| burst {100 * s['burst_overlap']['video']:.1f} %", flush=True)
        for kk, (m, lo, hi) in d.items():
            print(f"    {kk:11s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"    video-only MAE better in {better} matches, worse in {worse} (of {n})", flush=True)
    json.dump(out, open(OUT / "ab_summary.json", "w"), indent=1, default=float)
