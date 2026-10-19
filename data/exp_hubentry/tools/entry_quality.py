"""Entry detection quality: flight tracks that drop into a hub opening (autoscout.hubentry.is_entry) vs the
scoreboard counter, beside the hub-top blob activity (hub_feats).

Per alliance-match: entry tracks per counted ball per view; how many of them are launch arcs already linked to a
robot (result.json shots, same view and first point); and impulse responses (lag_deconv) from each entry signal to
the counter: a sharp, early response = good timing.

  .venv/Scripts/python data/exp_hubentry/tools/entry_quality.py [--du 1.0 --dv0 -0.6 --dv1 1.2]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lag_deconv import design, solve, stats  # noqa: E402
from lag_study import BIN, OUT, DST, counter_series, feat_series, hist, tl_window  # noqa: E402

from autoscout import hubentry as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--du", type=float, default=H.OPEN_DU)
ap.add_argument("--dv0", type=float, default=H.OPEN_DV[0])
ap.add_argument("--dv1", type=float, default=H.OPEN_DV[1])
ap.add_argument("--tag", default="")
a = ap.parse_args()

CACHE = DST / "cache"
rows, pooled = [], {}
for cp in sorted(CACHE.glob("*.pkl")):
    key = cp.stem
    r = json.loads((OUT / key / "result.json").read_text())
    tl = r["timeline"]
    lay = tl.get("layout") or "?"
    t0, t1 = tl_window(tl)
    edges = np.arange(t0, t1 + BIN, BIN)
    c = pickle.loads(cp.read_bytes())
    hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
          for k, v in json.loads((OUT / key / "hub_feats.json").read_text()).items()}
    shot_keys = {(s["view"], round(s["pts"][0][0], 2)) for s in r["shots"]}
    for al in ("red", "blue"):
        y = counter_series(tl["counts"].get(al) or [], edges)
        if y.sum() < 20:
            continue
        row = {"key": key, "al": al, "layout": lay, "balls": float(y.sum())}
        sigs = {}
        allent = np.zeros(len(edges) - 1)
        for v, T in c["views"].items():
            vk = "main" if v == "main" else ("near" if v.startswith(al) else "far")
            e = H.is_entry(T, al, a.du, (a.dv0, a.dv1))
            m = e & (T[:, H.C["t1"]] >= t0) & (T[:, H.C["t1"]] <= t1)
            ts = T[m, H.C["t1"]]
            row[f"n_{vk}"] = int(m.sum())
            row[f"shot_{vk}"] = int(sum((v, round(float(t), 2)) in shot_keys for t in T[m, H.C["t0"]]))
            row[f"len_{vk}"] = float(np.median(T[m, H.C["n"]])) if m.any() else None
            s = hist(ts, edges)
            sigs[f"ent_{vk}"] = s
            if vk != "far":
                allent += s / max(s.sum(), 1) * y.sum() if s.sum() else 0
        sigs["ent_mn"] = allent
        tb = np.zeros(len(edges) - 1)
        for v in ("main", f"{al}_end"):
            s = feat_series(hf.get((v, al, "top")), edges)
            if s is not None and s.sum() > 0:
                tb += s / s.sum() * y.sum()
        sigs["top_mn"] = tb
        sigs["ent+top"] = 0.5 * allent + 0.5 * tb
        for nm, s in sigs.items():
            if s.sum() <= 0:
                continue
            s = s * y.sum() / s.sum()
            X = design(s, y)
            row[nm] = stats(solve([X], [y]))
            pooled.setdefault((lay, nm), ([], []))
            pooled[(lay, nm)][0].append(X)
            pooled[(lay, nm)][1].append(y)
        rows.append(row)
    print(key, *[f"{x['al']} balls {x['balls']:.0f} ent main {x.get('n_main')} near {x.get('n_near')} far {x.get('n_far')}"
                 f" (shots {x.get('shot_main')}/{x.get('shot_near')})" for x in rows[-2:]], flush=True)

summ = {}
for lay in ("chezy", "first"):
    rr = [x for x in rows if x["layout"] == lay]
    b = np.array([x["balls"] for x in rr])
    s = {"alliances": len(rr), "balls": float(b.sum())}
    for vk in ("main", "near", "far"):
        n = np.array([x.get(f"n_{vk}", 0) for x in rr], float)
        sh = np.array([x.get(f"shot_{vk}", 0) for x in rr], float)
        s[f"entries_per_ball_{vk}"] = float(n.sum() / b.sum())
        s[f"entries_per_ball_{vk}_corr"] = float(np.corrcoef(n, b)[0, 1]) if n.std() > 0 else None
        s[f"entry_tracks_that_are_linked_launches_{vk}"] = float(sh.sum() / max(n.sum(), 1))
    summ[lay] = s
    print(lay, json.dumps(s))
for (lay, nm), (Xs, ys) in sorted(pooled.items()):
    st = stats(solve(Xs, ys))
    pk = [x[nm]["peak"] for x in rows if x["layout"] == lay and nm in x and x[nm]]
    sd = [x[nm]["sd"] for x in rows if x["layout"] == lay and nm in x and x[nm]]
    st.update({"per_alliance_peak_median": float(np.median(pk)), "per_alliance_peak_sd": float(np.std(pk)),
               "per_alliance_width_median": float(np.median(sd))})
    summ[f"{lay}|{nm}"] = st
    print(f"{lay:6s} {nm:8s} pooled peak {st['peak']:.2f} mean {st['mean']:.2f} sd {st['sd']:.2f} q {np.round(st['q10_25_50_75_90'], 2)} "
          f"gain {st['gain']:.2f} | per alliance peak {st['per_alliance_peak_median']:.2f} sd {st['per_alliance_peak_sd']:.2f} "
          f"width {st['per_alliance_width_median']:.2f}")
(DST / f"entry_quality{a.tag}.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
