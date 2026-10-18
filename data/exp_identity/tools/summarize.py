"""Metrics for absent_eval.py runs.

  summarize.py RUN [RUN2 ...] [--variants a b] [--per]

Per variant:
* scouting (5026, cc quals, alliances with all three robots scouted): within-alliance share error
  (total variation, mean over alliances), pooled share correlation, fuel MAE vs the scouted reference
  (TBA total split by scouted shots, score_scouting.py's definition); on all scouted alliances, on the
  non-shooter alliances (some robot scouted share < 0.06 with every scout <= 15 shots, cycle 16b), and on
  the "new" non-shooter alliances (not in data/backup_output_v1; cycle 16's 0.275 group)
* fuel OPR: pooled within-alliance share correlation (alliances with >= 40 fuel), all / train / test
* held-out identity (when the run has it)
Paired bootstrap CI (over alliances) of the TV-error change vs base.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import opr, scouting, tba  # noqa: E402

RUNS = ROOT / "data" / "exp_identity" / "runs"
TRAIN = {"2026casnv", "2026joh"}
ORIG = {p.parent.name for p in (ROOT / "data" / "backup_output_v1").glob("*/result.json")}


def load_runs(names):
    recs = {}
    for n in names:
        for line in (RUNS / f"{n}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["key"] in recs:
                    recs[r["key"]]["v"].update(r["v"])
                else:
                    recs[r["key"]] = r
    return recs


_BYM = None


def scout_alliances():
    global _BYM
    if _BYM is None:
        _BYM = scouting.by_match(scouting.load())
    out = []
    for key, teams in _BYM.items():
        if not key.startswith("2026cc_qm"):
            continue
        try:
            info = tba.match(key)
        except Exception:
            continue
        for al in ("red", "blue"):
            tm = info.teams(al)
            rs = [teams.get(t, []) for t in tm]
            if not all(rs):
                continue
            s = np.array([np.mean([x.n for x in rr]) for rr in rs])
            if s.sum() <= 0:
                continue
            sh = s / s.sum()
            ns = any(sh[i] < 0.06 and all(x.n <= 15 for x in rs[i]) for i in range(3))
            out.append({"key": key, "al": al, "teams": tm, "scout": s, "nonshooter": ns, "new": key not in ORIG,
                        "total": (info.hub(al) or {}).get("totalCount")})
    return out


def scout_metrics(recs, v, als):
    tv, a, b, err_ref, ref_all, vid_all, keys = [], [], [], [], [], [], []
    for x in als:
        r = recs.get(x["key"])
        if r is None or v not in r["v"]:
            continue
        f = np.array([r["v"][v]["fuel"][str(t)] for t in x["teams"]], float)
        if f.sum() <= 0:
            continue
        f_, s_ = f / f.sum(), x["scout"] / x["scout"].sum()
        tv.append(0.5 * np.abs(f_ - s_).sum())
        a += f_.tolist(); b += s_.tolist()
        if x["total"]:
            ref = x["total"] * s_
            ref_all += ref.tolist(); vid_all += f.tolist()
        keys.append((x["key"], x["al"]))
    if not tv:
        return None
    ref_all, vid_all = np.array(ref_all), np.array(vid_all)
    return {"n": len(tv), "tv": float(np.mean(tv)), "corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 else np.nan,
            "mae": float(np.mean(np.abs(ref_all - vid_all))) if len(ref_all) else np.nan,
            "fcorr": float(np.corrcoef(ref_all, vid_all)[0, 1]) if len(ref_all) > 5 else np.nan,
            "_tv": dict(zip(keys, tv))}


_OPR = {}


def opr_metrics(recs, v):
    res = {}
    for name, sel in (("all", None), ("train", True), ("test", False)):
        sa, so, n = [], [], 0
        for key, r in recs.items():
            if v not in r["v"]:
                continue
            if sel is not None and (r["event"] in TRAIN) != sel:
                continue
            if r["event"] not in _OPR:
                _OPR[r["event"]] = opr.fuel_opr(r["event"])
            o = _OPR[r["event"]]
            for al in ("red", "blue"):
                tm = r[al]
                f = np.array([r["v"][v]["fuel"][str(t)] for t in tm], float)
                op = np.array([o.get(t, np.nan) for t in tm])
                if f.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
                    continue
                sa += (f / f.sum()).tolist(); so += (op / op.sum()).tolist(); n += 1
        res[name] = (float(np.corrcoef(sa, so)[0, 1]) if len(sa) > 5 else np.nan, n)
    return res


def boot(d, n=5000, seed=0):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--per", action="store_true", help="per-alliance TV table for non-shooter alliances")
    ap.add_argument("--json", default=None)
    ap.add_argument("--ref", default="base", help="reference variant for the paired comparison")
    a = ap.parse_args()
    recs = load_runs(a.runs)
    vs = a.variants or sorted({v for r in recs.values() for v in r["v"]}, key=lambda s: (s != a.ref, s))
    als = scout_alliances()
    groups = {"all": als, "nonshoot": [x for x in als if x["nonshooter"]],
              "ns_new": [x for x in als if x["nonshooter"] and x["new"]], "rest": [x for x in als if not x["nonshooter"]]}
    out = {}
    print(f"{len(recs)} matches")
    hdr = f"{'variant':16s} | " + " | ".join(f"{g}: n tv corr mae" for g in groups) + " | OPR all/train/test | holdout"
    print(hdr)
    for v in vs:
        row = {}
        cells = []
        for g, xs in groups.items():
            m = scout_metrics(recs, v, xs)
            row[g] = m
            cells.append("-" if m is None else f"{m['n']:2d} {m['tv']:.3f} {m['corr']:.3f} {m['mae']:5.1f}")
        om = opr_metrics(recs, v)
        row["opr"] = om
        ag = sum(r["v"][v].get("agree", 0) for r in recs.values() if v in r["v"])
        ms = sum(r["v"][v].get("mass", 0) for r in recs.values() if v in r["v"])
        row["holdout"] = ag / ms if ms else None
        print(f"{v:16s} | " + " | ".join(cells) + f" | {om['all'][0]:.3f}/{om['train'][0]:.3f}/{om['test'][0]:.3f} "
              f"(n={om['all'][1]})" + (f" | {100 * ag / ms:.2f}%" if ms else ""))
        out[v] = row
    if a.ref in out:
        print(f"paired TV change vs {a.ref} (mean, 95% CI over alliances):")
        for v in vs:
            if v == a.ref:
                continue
            for g in ("all", "nonshoot", "rest"):
                b0, b1 = out[a.ref][g], out[v][g]
                if not b0 or not b1:
                    continue
                ks = [k for k in b1["_tv"] if k in b0["_tv"]]
                d = [b1["_tv"][k] - b0["_tv"][k] for k in ks]
                lo, hi = boot(d)
                print(f"   {v:16s} {g:9s} {np.mean(d):+.4f} [{lo:+.4f}, {hi:+.4f}]  better {sum(x < -1e-3 for x in d)} worse {sum(x > 1e-3 for x in d)} of {len(d)}")
    if a.per:
        for x in groups["nonshoot"] + [x for x in als if not x["nonshooter"]]:
            r = recs.get(x["key"])
            if r is None:
                continue
            ss = x["scout"] / x["scout"].sum()
            line = f"{x['key']:13s} {x['al']:4s} {'NS' if x['nonshooter'] else '  '} {x['teams']} scout {np.round(ss, 2).tolist()}"
            for v in vs:
                if v in r["v"]:
                    f = np.array([r["v"][v]["fuel"][str(t)] for t in x["teams"]], float)
                    line += f" | {v} {np.round(f / max(f.sum(), 1e-9), 2).tolist()}"
            print(line)
    if a.json:
        Path(a.json).write_text(json.dumps({v: {g: ({k: vv for k, vv in m.items() if k != "_tv"} if isinstance(m, dict) else m)
                                                for g, m in row.items()} for v, row in out.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
