"""Figure: one hopper dump (2026cc_qm24 blue, 33-46 s): cumulative balls entering the hub (entry tracks, main and
blue end view, scaled to the counted total) vs the scoreboard counter, and the re-timed entry times.

  .venv/Scripts/python data/exp_hubentry/tools/fig_queue.py -> data/exp_hubentry/img/queue_qm24_blue.png
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lag_study import BIN, DST, OUT, counter_series, hist, tl_window  # noqa: E402

from autoscout import hubentry as H  # noqa: E402

key, al, a, b = "2026cc_qm24", "blue", 33.0, 46.0
r = json.loads((OUT / key / "result.json").read_text())
tl = r["timeline"]
c = pickle.loads((DST / "cache" / f"{key}.pkl").read_bytes())
hf = {tuple(k.split("|")): np.array(v).reshape(-1, 3) for k, v in json.loads((OUT / key / "hub_feats.json").read_text()).items()}
edges = np.arange(a, b + BIN, BIN)
tc = edges[:-1] + BIN / 2
y = counter_series(tl["counts"][al], edges)
t0, t1 = tl_window(tl)
inc, prev = [], 0
for t, n in tl["counts"][al]:
    if n > prev and t0 <= t <= t1:
        inc.append((t, float(n - prev)))
    prev = max(prev, n)
rt, rate = H.entry_series(c["views"], hf, al, t0, t1)
ents = [e for e in H.retime_queue(inc, rt, rate) if a <= e.count_t <= b]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.step(tc, np.cumsum(y), where="post", color="k", lw=2, label="scoreboard counter")
for v, col, nm in (("main", "C0", "main view"), (f"{al}_end", "C1", "blue end view")):
    T = c["views"][v]
    e = H.is_entry(T, al)
    ts = T[e, H.C["t1"]]
    h = hist(ts[(ts >= a) & (ts <= b)], edges)
    ax.plot(tc, np.cumsum(h) * y.sum() / max(h.sum(), 1), col, lw=1.5, label=f"balls dropping into the opening, {nm} (scaled)")
et = np.array([e.entry_t for e in ents])
en = np.array([e.n for e in ents])
o = np.argsort(et)
ax.step(et[o], np.cumsum(en[o]), where="post", color="C2", ls="--", lw=1.5, label="counted balls re-timed to entry (FIFO)")
sh = [s["t"] for s in r["shots"] if s["alliance"] == al and s["team"] and a <= s["t"] <= b]
ax.plot(sh, np.full(len(sh), -3), "|", color="0.4", ms=8, label="detected launches")
ax.set_xlabel("video time (s)")
ax.set_ylabel("balls")
ax.set_title("2026cc_qm24 blue: 80 % of a 65-ball dump drops in within 2.5 s; the counter needs 5.7 s")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(DST / "img" / "queue_qm24_blue.png", dpi=110)
