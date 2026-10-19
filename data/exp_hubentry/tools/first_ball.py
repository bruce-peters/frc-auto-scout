"""Lag study, step 4: delay of the FIRST ball of a counter burst (the hub's queue is empty after a pause, so this
is the bare entry -> counter delay d0), and the counter's service rate inside bursts (a ceiling = metered hub).

For every burst preceded by >= QUIET s without counts: the last entry track (main / own end view) or hub-top blob
onset before the first jump. Also the 1-s count rate distribution inside bursts.

  .venv/Scripts/python data/exp_hubentry/tools/first_ball.py -> data/exp_hubentry/first_ball.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lag_study import OUT, DST, tl_window  # noqa: E402

from autoscout import hubentry as H  # noqa: E402

QUIET = 3.0
res = {"chezy": {"d0_track": [], "d0_blob": [], "rate1s": [], "rate_max_per_burst": []},
       "first": {"d0_track": [], "d0_blob": [], "rate1s": [], "rate_max_per_burst": []}}
for cp in sorted((DST / "cache").glob("*.pkl")):
    key = cp.stem
    r = json.loads((OUT / key / "result.json").read_text())
    tl = r["timeline"]
    lay = tl.get("layout") or "?"
    c = pickle.loads(cp.read_bytes())
    hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
          for k, v in json.loads((OUT / key / "hub_feats.json").read_text()).items()}
    t0, t1 = tl_window(tl)
    for al in ("red", "blue"):
        inc = []
        prev = 0
        for t, n in tl["counts"].get(al) or []:
            if n > prev and t0 <= t <= t1:
                inc.append((t, n - prev))
            prev = max(prev, n)
        if sum(n for _, n in inc) < 20:
            continue
        ent = []
        for v in ("main", f"{al}_end"):
            T = c["views"].get(v)
            if T is None:
                continue
            e = H.is_entry(T, al)
            ent.append(T[e, H.C["t1"]])
        ent = np.sort(np.concatenate(ent)) if ent else np.zeros(0)
        blob = hf.get(("main", al, "top"))
        bursts = H._bursts(inc, gap=1.0)
        last_t = -1e9
        for b in bursts:
            ta = b[0][0]
            nb = sum(n for _, n in b)
            ts = np.array([t for t, _ in b])
            if len(b) >= 3:
                # balls per second over the burst's busiest 1 s
                best = max(sum(n for t, n in b if s <= t < s + 1.0) for s, _ in b)
                res[lay]["rate_max_per_burst"].append((nb, best))
                for s, _ in b:
                    res[lay]["rate1s"].append(sum(n for t, n in b if s <= t < s + 1.0))
            if ta - last_t >= QUIET:
                # first-ball delay: the earliest entry track in [ta - 3.5, ta - 0.3] that starts the burst's entries
                m = (ent >= ta - 3.5) & (ent <= ta - 0.2)
                if m.any():
                    res[lay]["d0_track"].append(float(ta - ent[m].min()))
                if blob is not None and len(blob):
                    w = (blob[:, 0] >= ta - 3.5) & (blob[:, 0] <= ta - 0.2)
                    on = blob[w][blob[w, 1] >= 2]
                    if len(on):
                        res[lay]["d0_blob"].append(float(ta - on[0, 0]))
            last_t = b[-1][0]
summ = {}
for lay, d in res.items():
    s = {}
    for k in ("d0_track", "d0_blob"):
        a = np.array(d[k])
        s[k] = {"n": len(a), "q10_25_50_75_90": np.percentile(a, [10, 25, 50, 75, 90]).round(2).tolist()}
    r1 = np.array(d["rate1s"])
    s["count_rate_1s_q50_90_99_max"] = np.percentile(r1, [50, 90, 99, 100]).round(1).tolist()
    rm = np.array(d["rate_max_per_burst"])
    big = rm[rm[:, 0] >= 30]
    s["busiest_1s_in_bursts_ge30_balls_q10_50_90"] = np.percentile(big[:, 1], [10, 50, 90]).round(1).tolist() if len(big) else None
    summ[lay] = s
    print(lay, json.dumps(s))
(DST / "first_ball.json").write_text(json.dumps(summ, indent=1))
