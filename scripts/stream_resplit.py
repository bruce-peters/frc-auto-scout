"""Re-split stream_experiment results under other credit rules: {old rule, learned model} x {no de-dup, de-dup}.

For every scouted alliance (all three robots scouted, overlay counter present) of the given result variants
(data/exp_stream/<STREAM_RESULTS>/<variant>/<key>.json, written by scripts/stream_experiment.py run),
TeamSignals are rebuilt exactly like the main session's de-dup test does (data/exp_attrmodel/tools/
build_ds.team_signals; per-view sightings from the run's own _views file), then:
  * de-dup W s: a team's launches from *different* views within W s become one (max weight kept);
  * rule: the hand-tuned split (attribution.ATTR_MODEL = None) or the learned credit model
    (data/exp_attrmodel/attr_model.json, with the event fuel OPR as its match-level prior).
Scored vs 5026's scouting (match shares of mean scouted shots): TV = 0.5 * sum |video share - scout share|
per alliance, share corr, top-scorer agreement, and the 5026 gap (mean video - scout share of 5026).

  STREAM_RESULTS=results_v3full python scripts/stream_resplit.py --variants motion motion+bt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "exp_attrmodel" / "tools"))
import numpy as np  # noqa: E402

import build_ds as B  # noqa: E402
from autoscout import attribution as A, config, opr as O, pipeline as P, scouting, tba  # noqa: E402

RESULTS = config.DATA / "exp_stream" / os.environ.get("STREAM_RESULTS", "results")
MODEL = "data/exp_attrmodel/attr_model.json"


def dedup(ts, w: float):
    """The main session's rule (dedup_test.py): merge consecutive launches from different views within w s."""
    if w <= 0 or len(ts.shots) < 2:
        return ts.shots, ts.shot_w
    o = np.argsort(ts.shots)
    t, wt, v = ts.shots[o], ts.shot_w[o], ts.shot_view[o]
    kt, kw, kv = [], [], []
    for ti, wi, vi in zip(t, wt, v):
        if kt and ti - kt[-1] <= w and vi != kv[-1]:
            kw[-1] = max(kw[-1], wi)
            continue
        kt.append(ti); kw.append(wi); kv.append(vi)
    return np.array(kt), np.array(kw)


def alliances(variant: str, bym: dict) -> list:
    out = []
    for f in sorted((RESULTS / variant).glob("*.json")):
        key = f.stem
        if key not in bym:
            continue
        r = json.loads(f.read_text())
        vp = RESULTS / "_views" / f"{key}.json"
        views = json.loads(vp.read_text()) if vp.exists() else {}
        info = tba.match(key)
        tl = B.TL(r)
        for al in ("red", "blue"):
            teams = info.teams(al)
            if not tl.counts.get(al) or not all(bym[key].get(t) for t in teams):
                continue
            n = np.array([np.mean([x.n for x in bym[key][t]]) for t in teams])
            if n.sum() <= 0:
                continue
            dbt = defaultdict(list)
            for t in teams:
                for vname, times in views.get(str(t), {}).items():
                    vk = "main" if vname == "main" else ("own" if vname == f"{al}_end" else "oth")
                    dbt[(t, vk)] += times
            dbt = {k: np.array(sorted(v)) for k, v in dbt.items()}
            sigs = [B.team_signals(r, info, al, t, tl, dbt) for t in teams]
            stored = np.array([r["teams"][str(t)]["fuel_unscaled"] for t in teams], float)
            # the same post-match filter of the scoreboard reads as pipeline.analyse
            inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al)))
            out.append(dict(key=key, al=al, teams=teams, ref=n / n.sum(), sigs=sigs, tl=tl, stored=stored,
                            inc=inc, event=info.event_key))
    return out


def score(recs: list, rule: str, w: float, oprs: dict) -> dict:
    A.ATTR_MODEL = MODEL if rule == "model" else None
    tv, a, b, top, gap, parity = [], [], [], [], [], []
    for x in recs:
        for s in x["sigs"]:
            if not hasattr(s, "_orig"):
                s._orig = (s.shots, s.shot_w)
            s.shots, s.shot_w = s._orig
            s.shots, s.shot_w = dedup(s, w)
        sp = A.split_increments(x["inc"], x["sigs"], x["tl"].match_time,
                                opr=oprs.get(x["event"]) if rule == "model" else None)
        f = np.array([sum(sp[t].values()) for t in x["teams"]])
        sh = f / max(f.sum(), 1e-9)
        if rule == "old" and w == 0:
            parity.append(float(np.max(np.abs(f - x["stored"]))))
        tv.append(0.5 * np.abs(sh - x["ref"]).sum())
        a += sh.tolist(); b += x["ref"].tolist()
        top.append(int(np.argmax(sh) == np.argmax(x["ref"])))
        if 5026 in x["teams"]:
            i = x["teams"].index(5026)
            gap.append(sh[i] - x["ref"][i])
    A.ATTR_MODEL = None
    out = {"n": len(recs), "tv": float(np.mean(tv)), "corr": float(np.corrcoef(a, b)[0, 1]),
           "top": float(np.mean(top)), "gap5026": float(np.mean(gap)) if gap else float("nan"), "n5026": len(gap),
           "_tv": tv, "_keys": [x["key"] for x in recs]}
    if parity:
        out["parity_max_abs_fuel"] = float(np.max(parity))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["motion", "motion+bt"])
    ap.add_argument("--dedup", type=float, nargs="*", default=[0.0, 0.1])
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="match keys to leave out")
    a = ap.parse_args()
    bym = {k: v for k, v in scouting.by_match(scouting.load()).items() if k not in set(a.exclude)}
    data = {v: alliances(v, bym) for v in a.variants}
    common = set.intersection(*[{(x["key"], x["al"]) for x in d} for d in data.values()])
    data = {v: [x for x in d if (x["key"], x["al"]) in common] for v, d in data.items()}
    oprs = {ev: O.fuel_opr(ev) for ev in {x["event"] for d in data.values() for x in d}}
    print(f"{len(common)} scouted alliances ({len({k for k, _ in common})} matches), results {RESULTS.name}\n")
    print(f"{'tracker':12s} {'rule':6s} {'dedup':>5s} | {'scout TV':>8s} {'corr':>6s} {'top':>5s} {'5026 gap':>8s} (n)")
    rows = {}
    for v, recs in data.items():
        for rule in ("old", "model"):
            for w in a.dedup:
                m = score(recs, rule, w, oprs)
                rows[(v, rule, w)] = m
                par = f"   parity vs stored fuel: max |diff| {m['parity_max_abs_fuel']:.1f}" if "parity_max_abs_fuel" in m else ""
                print(f"{v:12s} {rule:6s} {w:5.2f} | {m['tv']:8.3f} {m['corr']:6.3f} {100 * m['top']:4.0f}% "
                      f"{m['gap5026']:+8.3f} ({m['n5026']}){par}")
    # paired bootstrap over matches of the TV difference vs the first configuration
    base = (a.variants[0], "old", a.dedup[0])
    rng = np.random.default_rng(0)
    bk = rows[base]["_keys"]
    keys = sorted(set(bk))
    print(f"\npaired bootstrap over matches, TV difference vs {base} (90 % CI; P(lower TV)):")
    for k, m in rows.items():
        if k == base:
            continue
        d = np.array(m["_tv"]) - np.array(rows[base]["_tv"])
        by = defaultdict(list)
        for kk, dd in zip(bk, d):
            by[kk].append(dd)
        bs = [np.mean([x for kk in rng.choice(keys, len(keys)) for x in by[kk]]) for _ in range(2000)]
        print(f"  {k[0]:12s} {k[1]:6s} {k[2]:4.2f}: dTV {np.mean(d):+.4f} CI [{np.percentile(bs, 5):+.4f}, "
              f"{np.percentile(bs, 95):+.4f}] P={np.mean(np.array(bs) < 0):.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
                                           for k, m in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
