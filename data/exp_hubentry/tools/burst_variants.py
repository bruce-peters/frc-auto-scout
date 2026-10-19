"""How much does the burst metric depend on its own timing window?

score_scouting's burst metric takes, for each scoreboard jump at counter time T, the scouts' taps around T - 2.0 s
(+-2.5 s triangle) as the truth. In launch terms that window is 0.3-5.3 s before the jump, peaking at 2.8 s: the
same shape as today's credit kernel (0.8 / 2.8 / 5.5), so the metric's truth carries the same smear as the
baseline. This recomputes the metric (balls given to the robots scouts saw shooting) for several truth windows,
and at the episode level (all jumps of a counter burst pooled, gaps > GAP s; truth = taps in [first - 4.5,
last + 0.5] s on the scouts' clock), where the timing inside a burst no longer matters. Paired bootstrap vs base.

  .venv/Scripts/python data/exp_hubentry/tools/burst_variants.py --arms base,entry_q,...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import config, scouting, tba  # noqa: E402
from score_scouting import to_mt_fn  # noqa: E402

AB = config.DATA / "exp_hubentry" / "ab"
WINDOWS = {"official (-2.0, 2.5)": (-2.0, 2.5), "narrow (-2.0, 1.5)": (-2.0, 1.5), "narrow (-2.4, 1.5)": (-2.4, 1.5),
           "wide (-2.4, 3.5)": (-2.4, 3.5)}
EPISODE_GAPS = (2.0, 4.0)


def tri(dt, lag, half):
    return np.clip(1 - np.abs(dt - lag) / half, 0, None)


def per_match(out_root):
    bm = scouting.by_match(scouting.load())
    pm = {}
    for key, teams in sorted(bm.items()):
        rp = out_root / key / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        info = tba.match(key)
        to_mt = to_mt_fn(res["timeline"])
        a0 = res["timeline"].get("auto_start") or 6.13
        d = pm.setdefault(key, {})
        for al in ("red", "blue"):
            ts = info.teams(al)
            hub = info.hub(al) or {}
            if not hub.get("totalCount") or not all(teams.get(t) for t in ts):
                continue
            log = [(mt, tm, b) for mt, tm, b in res.get("fuel_log") or [] if tm in ts]
            timed = {t: [r for r in teams[t] if r.time_valid] for t in ts}
            if not all(timed.values()):
                continue
            jumps, prev = [], 0
            for vt, c in res["timeline"]["counts"].get(al, []):
                if c > prev:
                    mt_log = to_mt(vt - 2.0)
                    got = np.array([sum(b for m, tm, b in log if tm == t and abs(m - mt_log) < 0.06) for t in ts])
                    jumps.append((vt, c - prev, got))
                prev = max(prev, c)
            for name, (lag, half) in WINDOWS.items():
                for vt, k, got in jumps:
                    st_c = vt - a0
                    w = np.array([np.mean([tri(r.shots - st_c, lag, half).sum() for r in timed[t]]) for t in ts])
                    if w.sum() <= 0 or got.sum() <= 0:
                        continue
                    x = d.setdefault(name, [0.0, 0.0])
                    x[0] += k
                    x[1] += k * float(np.minimum(got / got.sum(), w / w.sum()).sum())
            for gap in EPISODE_GAPS:
                eps, cur = [], []
                for j in jumps:
                    if cur and j[0] - cur[-1][0] > gap:
                        eps.append(cur)
                        cur = []
                    cur.append(j)
                if cur:
                    eps.append(cur)
                for e in eps:
                    lo, hi = e[0][0] - a0 - 4.5, e[-1][0] - a0 + 0.5
                    w = np.array([np.mean([((r.shots >= lo) & (r.shots <= hi)).sum() for r in timed[t]]) for t in ts], float)
                    got = sum(j[2] for j in e)
                    k = sum(j[1] for j in e)
                    if w.sum() <= 0 or got.sum() <= 0:
                        continue
                    x = d.setdefault(f"episode gap {gap:g} s", [0.0, 0.0])
                    x[0] += k
                    x[1] += k * float(np.minimum(got / got.sum(), w / w.sum()).sum())
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,entry_q")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    arms = a.arms.split(",")
    P = {arm: per_match(AB / arm) for arm in arms}
    keys = sorted(set.intersection(*[set(p) for p in P.values()]))
    names = list(WINDOWS) + [f"episode gap {g:g} s" for g in EPISODE_GAPS]
    rng = np.random.default_rng(0)
    boots = [rng.choice(keys, len(keys)) for _ in range(a.boot)]

    def val(p, ks, nm):
        b = sum(p[k].get(nm, [0, 0])[0] for k in ks)
        o = sum(p[k].get(nm, [0, 0])[1] for k in ks)
        return o / max(b, 1e-9)

    out = {}
    for nm in names:
        b0 = val(P["base"], keys, nm)
        row = {"base": b0}
        line = f"{nm:22s} base {100 * b0:.1f}"
        for arm in arms[1:]:
            v = val(P[arm], keys, nm)
            d = np.array([val(P[arm], ks, nm) - val(P["base"], ks, nm) for ks in boots])
            ci = np.percentile(d, [2.5, 97.5])
            row[arm] = {"value": v, "diff": v - b0, "ci95": ci.tolist()}
            line += f" | {arm} {100 * v:.1f} ({100 * (v - b0):+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}])"
        out[nm] = row
        print(line, flush=True)
    (config.DATA / "exp_hubentry" / "burst_variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
