"""Cycle 15: 'shadow' tracks - a team's track running within 1 m of a teammate's for a long time
(a duplicate track of one robot that took the other team's identity). Per team-match: fraction of the
time both are observed (fixes within 0.3 s) that they are < 1.0 m apart. Link to errors vs scouting."""
import json
import numpy as np
from common import AUDIT, P, TL, load, match_keys, path_video_times, scout_counts, tba
sc = scout_counts()
rows = []
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    grid = np.arange(tl.t0(), tl.t1(), 0.5)
    for al in ("red", "blue"):
        tm = info.teams(al)
        P_ = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in tm}
        obs = {}
        for t in tm:
            p = P_[t]
            if not len(p):
                obs[t] = np.full((len(grid), 2), np.nan); continue
            j = np.clip(np.searchsorted(p[:, 0], grid), 1, len(p) - 1)
            a, b = p[j - 1], p[j]
            pick = np.where((np.abs(a[:, 0] - grid) < np.abs(b[:, 0] - grid))[:, None], a, b)
            ok = np.abs(pick[:, 0] - grid) <= 0.3
            o = pick[:, 1:].copy(); o[~ok] = np.nan
            obs[t] = o
        f = np.array([r["teams"][str(t)]["fuel"] for t in tm])
        s = np.array([sc[k][t] for t in tm]) if k in sc and all(t in sc[k] for t in tm) else None
        for i, t in enumerate(tm):
            best = 0.0; mate = None
            for j, u in enumerate(tm):
                if u == t:
                    continue
                both = ~np.isnan(obs[t][:, 0]) & ~np.isnan(obs[u][:, 0])
                if both.sum() < 20:
                    continue
                close = np.hypot(*(obs[t][both] - obs[u][both]).T) < 1.0
                if close.mean() > best:
                    best, mate = float(close.mean()), u
            rows.append({"key": k, "team": t, "mate": mate, "shadow": best, "fuel_share": float(f[i] / f.sum()) if f.sum() else None,
                         "res_scout": float(f[i] / f.sum() - s[i] / s.sum()) if s is not None and s.sum() > 0 and f.sum() > 0 else None})
sh = np.array([x["shadow"] for x in rows])
print(f"{len(rows)} team-matches; share of jointly-observed time within 1 m of a teammate: median {np.median(sh):.2f}, "
      f"q90 {np.quantile(sh, .9):.2f}; >= 0.3: {np.mean(sh >= 0.3) * 100:.1f} %, >= 0.5: {np.mean(sh >= 0.5) * 100:.1f} %")
sc_rows = [x for x in rows if x["res_scout"] is not None]
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.01)):
    g = [x for x in sc_rows if lo <= x["shadow"] < hi]
    if g:
        print(f"   shadow {lo:.1f}-{hi:.1f}: {len(g):3d} scouted team-matches, mean |share err| {np.mean([abs(x['res_scout']) for x in g]):.3f}, "
              f"mean signed err {np.mean([x['res_scout'] for x in g]):+.3f}")
for x in sorted(rows, key=lambda x: -x["shadow"])[:10]:
    print("  ", x)
json.dump(rows, open(AUDIT / "c15_shadow.json", "w"), indent=0)
