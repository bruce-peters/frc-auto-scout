"""Re-timed entry -> counter delay per ball (hubentry.retime_queue, main + own end view evidence), by position in its
counter burst (balls counted before it in the same burst) and by layout. Shows the hub's queue: the deeper into a
hopper dump, the later the count.

  .venv/Scripts/python data/exp_hubentry/tools/delay_profile.py -> data/exp_hubentry/delay_profile.json
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

BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 999)]
acc = {}
for cp in sorted((DST / "cache").glob("*.pkl")):
    key = cp.stem
    r = json.loads((OUT / key / "result.json").read_text())
    tl = r["timeline"]
    lay = tl.get("layout") or "?"
    t0, t1 = tl_window(tl)
    c = pickle.loads(cp.read_bytes())
    hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
          for k, v in json.loads((OUT / key / "hub_feats.json").read_text()).items()}
    for al in ("red", "blue"):
        inc, prev = [], 0
        for t, n in tl["counts"].get(al) or []:
            if n > prev and t0 <= t <= t1:
                inc.append((t, float(n - prev)))
            prev = max(prev, n)
        if sum(n for _, n in inc) < 20:
            continue
        rt, rate = H.entry_series(c["views"], hf, al, t0, t1)
        ents = H.retime_queue(inc, rt, rate)
        # position in burst
        bursts = H._bursts(inc, 1.0)
        start = {}
        for b in bursts:
            cum = 0.0
            for t, n in b:
                start[t] = cum
                cum += n
        seen = {}
        for e in ents:
            if e.conf <= 0:
                continue
            pos = start.get(e.count_t, 0.0) + seen.get(e.count_t, 0.0)
            seen[e.count_t] = seen.get(e.count_t, 0.0) + e.n
            acc.setdefault(lay, []).append((pos, e.count_t - H.READ_GAP / 2 - e.entry_t, e.n))
out = {}
for lay, rows in acc.items():
    R = np.array(rows)
    o = {}
    for lo, hi in BINS:
        m = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if m.sum() < 10:
            continue
        d, w = R[m, 1], R[m, 2]
        idx = np.argsort(d)
        cw = np.cumsum(w[idx]) / w.sum()
        q = [float(np.interp(x, cw, d[idx])) for x in (0.1, 0.5, 0.9)]
        o[f"balls {lo}-{hi} into burst"] = {"balls": float(w.sum()), "delay_q10_50_90": [round(x, 2) for x in q]}
    d, w = R[:, 1], R[:, 2]
    idx = np.argsort(d)
    cw = np.cumsum(w[idx]) / w.sum()
    o["all"] = {"balls": float(w.sum()), "delay_q10_25_50_75_90": [round(float(np.interp(x, cw, d[idx])), 2)
                                                                  for x in (0.1, 0.25, 0.5, 0.75, 0.9)]}
    out[lay] = o
    print(lay, json.dumps(o, indent=1))
(DST / "delay_profile.json").write_text(json.dumps(out, indent=1))
