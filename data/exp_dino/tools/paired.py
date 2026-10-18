"""Paired comparison of per-team fuel vs 5026's scouting reference between two result dirs (bootstrap over matches).

Reference as in scripts/score_scouting.py: the alliance's TBA fuel total split by scouted shots.
  paired.py DIR_A DIR_B [--field fuel|fuel_video]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import scouting, tba  # noqa: E402


def rows(d: Path, field: str) -> dict:
    out = {}
    bm = scouting.by_match(scouting.load())
    for key, teams in bm.items():
        rp = d / key / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        info = tba.match(key)
        for al in ("red", "blue"):
            ts = info.teams(al)
            total = (info.hub(al) or {}).get("totalCount")
            if not total or not all(teams.get(t) for t in ts):
                continue
            n = np.array([np.mean([r.n for r in teams[t]]) for t in ts])
            if n.sum() <= 0:
                continue
            ref = total * n / n.sum()
            for i, t in enumerate(ts):
                v = res["teams"][str(t)]
                out[(key, al, t)] = (ref[i], v.get(field, v["fuel"]))
    return out


def metrics(r: dict, ks: list) -> dict:
    ref = np.array([r[k][0] for k in ks])
    vid = np.array([r[k][1] for k in ks])
    top, groups = 0, {}
    for i, k in enumerate(ks):
        groups.setdefault(k[:2], []).append(i)
    for idx in groups.values():
        top += int(np.argmax(vid[idx]) == np.argmax(ref[idx]))
    return {"mae": float(np.abs(vid - ref).mean()), "corr": float(np.corrcoef(vid, ref)[0, 1]),
            "top": top / len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--field", default="fuel")
    x = ap.parse_args()
    a, b = rows(Path(x.a), x.field), rows(Path(x.b), x.field)
    ks = sorted(set(a) & set(b))
    ea = np.array([abs(a[k][1] - a[k][0]) for k in ks])
    eb = np.array([abs(b[k][1] - b[k][0]) for k in ks])
    matches = sorted({k[0] for k in ks})
    idx = {m: [i for i, k in enumerate(ks) if k[0] == m] for m in matches}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        s = rng.choice(matches, len(matches))
        ii = np.concatenate([idx[m] for m in s])
        diffs.append(eb[ii].mean() - ea[ii].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    ma, mb = metrics(a, ks), metrics(b, ks)
    print(f"[{x.field}] n={len(ks)} team-matches / {len(matches)} matches: MAE {ma['mae']:.1f} -> {mb['mae']:.1f} "
          f"(diff {mb['mae'] - ma['mae']:+.2f}, 95% CI {lo:+.2f} .. {hi:+.2f}); corr {ma['corr']:.3f} -> {mb['corr']:.3f}; "
          f"top {100 * ma['top']:.0f}% -> {100 * mb['top']:.0f}%; team-matches better {np.mean(eb < ea - 1e-9):.0%}, "
          f"worse {np.mean(eb > ea + 1e-9):.0%}")


if __name__ == "__main__":
    main()
