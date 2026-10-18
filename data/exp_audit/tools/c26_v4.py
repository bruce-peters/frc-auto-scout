"""Cycle 26: re-check after finalize_v4 (learned credit model, park_ext, counter window, OCR persistence).

A) columns: baseline (snapshot_baseline2.json, 02:35, old video rule) / fuel_video now / fuel (model) now:
   scout share corr + TV error, scout fuel MAE (TBA total x scouted share), compression slope, OPR corr
   (circular for the model: it has a fuel-OPR prior), 5026 gap, original vs new cc.
B) failure classes (fuel_video): counter vs TBA, missing views, non-shooters' credit, identity contradictions,
   never-read teams, shadow pairs, SEG_MAX flips, same-alliance ambiguous launch share.
C) error budget (fuel_video): per tag, alliances, mean team-match fuel error vs scouting, excess error mass.
"""
import json
import os
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, P, TL, load, match_keys, opr, path_video_times, scout_counts, scout_fuel_mae, scout_metrics, share_metrics, tba, time_valid
from autoscout import scouting, views

snap = json.load(open(AUDIT / "snapshot_baseline2.json"))["matches"]
keys = [k for k in match_keys("v2") if k in snap]
sc = scout_counts()
bym = scouting.by_match(scouting.load())
ORIG = set(match_keys("v1"))
R = {k: load(k, "v2") for k in keys}
INFO = {k: tba.match(k) for k in keys}
cols = {"baseline 02:35": {k: {int(t): f for t, f in snap[k]["fuel"].items()} for k in keys},
        "fuel_video now": {k: {t: R[k]["teams"][str(t)]["fuel_video"] for t in INFO[k].red + INFO[k].blue} for k in keys},
        "fuel (model) now": {k: {t: R[k]["teams"][str(t)]["fuel"] for t in INFO[k].red + INFO[k].blue} for k in keys}}


def per_alliance(pm):
    out = {}
    for k, f in pm.items():
        if k not in sc:
            continue
        for al in ("red", "blue"):
            tm = INFO[k].teams(al)
            tot = INFO[k].hub(al).get("totalCount")
            if not tot or not all(t in sc[k] for t in tm):
                continue
            s = np.array([sc[k][t] for t in tm]); v = np.array([f[t] for t in tm])
            if s.sum() <= 0 or v.sum() <= 0:
                continue
            out[(k, al)] = {"tv": float(np.abs(v / v.sum() - s / s.sum()).sum() / 2),
                            "fuel_err": float(np.mean(np.abs(v - tot * s / s.sum()))), "vs": (v / v.sum()).tolist(), "ss": (s / s.sum()).tolist(),
                            "teams": tm}
    return out


print("A) columns")
PA = {}
for name, pm in cols.items():
    pa = per_alliance(pm); PA[name] = pa
    vs = np.concatenate([x["vs"] for x in pa.values()]); ss = np.concatenate([x["ss"] for x in pa.values()])
    g5 = [x["vs"][x["teams"].index(5026)] - x["ss"][x["teams"].index(5026)] for x in pa.values() if 5026 in x["teams"]]
    eo = np.mean([x["tv"] for (k, al), x in pa.items() if k in ORIG]); en = np.mean([x["tv"] for (k, al), x in pa.items() if k not in ORIG])
    m = share_metrics(pm); s = scout_metrics(pm); f = scout_fuel_mae(pm)
    print(f"   {name:17s} scout corr {s['scout_corr']:.3f} TV {np.mean([x['tv'] for x in pa.values()]):.3f} fuel MAE {f['scout_fuel_mae']:.1f} "
          f"(n={f['n']}) slope {np.polyfit(ss, vs, 1)[0]:.2f} | OPR corr {m['corr']:.3f} | 5026 gap {np.mean(g5):+.3f} (n={len(g5)}) | "
          f"TV orig cc {eo:.3f} new cc {en:.3f}")

print("\nB) failure classes (fuel_video now unless noted)")
off = [(k, al) for k in keys for al in ("red", "blue") if R[k]["teams"][f"_{al}"].get("tba_total") and R[k]["teams"][f"_{al}"].get("timeline_total")
       and abs(R[k]["teams"][f"_{al}"]["timeline_total"] / R[k]["teams"][f"_{al}"]["tba_total"] - 1) > 0.10]
print(f"   counter > 10 % off TBA: {off}")
print(f"   missing views: {[(k, sorted(R[k]['calibration'])) for k in keys if len(R[k]['calibration']) < 3]}")
# non-shooters
ns = defaultdict(list)
for name in cols:
    for (k, al), x in PA[name].items():
        for i, t in enumerate(x["teams"]):
            rows = bym[k].get(t, [])
            if x["ss"][i] < 0.06 and rows and all(r_.n <= 15 for r_ in rows):
                ns[name].append(x["vs"][i])
print("   non-shooters (scouted share < 0.06): mean credited share " + ", ".join(f"{n} {np.mean(v):.3f} (n={len(v)})" for n, v in ns.items()))
# identity contradictions / never-read teams
st = sd = cn = un = 0
for k in keys:
    info = INFO[k]
    for al in ("red", "blue"):
        tm = info.teams(al); ide = R[k]["identity"][al]
        for e in ide.values():
            o = e["ocr"]
            if sum(o) >= 1.5 and max(o) / sum(o) >= 0.7:
                st += 1; cn += int(e["team"] in tm and tm[int(np.argmax(o))] != e["team"])
        own = {t: sum(e["ocr"][i] for e in ide.values() if e["team"] == t) for i, t in enumerate(tm)}
        un += sum(1 for t in tm if own[t] < 0.5 and all(own[u] >= 3 for u in tm if u != t))
print(f"   identity: strongly-read segments contradicted {cn}/{st} ({100 * cn / st:.1f} %; was 11.6 % before SEG_MAX); never-read team-matches {un}")
# shadow pairs
sh = tot_tm = 0
for k in keys:
    tl = TL(R[k]); grid = np.arange(tl.t0(), tl.t1(), 0.5)
    for al in ("red", "blue"):
        tm = INFO[k].teams(al); obs = {}
        for t in tm:
            p = path_video_times(R[k]["paths"].get(str(t), []), tl)
            o = np.full((len(grid), 2), np.nan)
            if len(p):
                j = np.clip(np.searchsorted(p[:, 0], grid), 1, len(p) - 1)
                a, b = p[j - 1], p[j]
                pick = np.where((np.abs(a[:, 0] - grid) < np.abs(b[:, 0] - grid))[:, None], a, b)
                ok = np.abs(pick[:, 0] - grid) <= 0.3
                o = pick[:, 1:].copy(); o[~ok] = np.nan
            obs[t] = o
        for t in tm:
            tot_tm += 1
            best = 0.0
            for u in tm:
                if u == t:
                    continue
                both = ~np.isnan(obs[t][:, 0]) & ~np.isnan(obs[u][:, 0])
                if both.sum() >= 20:
                    best = max(best, float((np.hypot(*(obs[t][both] - obs[u][both]).T) < 1.0).mean()))
            sh += int(best >= 0.3)
print(f"   shadow team-matches (>= 30 % within 1 m of a teammate): {sh}/{tot_tm} ({100 * sh / tot_tm:.1f} %; was 3.7 %)")

print("\nC) error budget, fuel_video, scouted cc alliances: tag -> alliances, mean team-match fuel error, excess error mass")
pa = PA["fuel_video now"]
tags = defaultdict(set)
for (k, al), x in pa.items():
    tm = x["teams"]
    if any(x["ss"][i] < 0.06 for i in range(3)):
        tags["has non-shooter"].add((k, al))
    if 5026 in tm:
        tags["5026 alliance"].add((k, al))
    if k not in ORIG:
        tags["new cc match"].add((k, al))
    ide = R[k]["identity"][al]
    for i, t in enumerate(tm):
        segs = [e for e in ide.values() if e["team"] == t]
        dur = sum(e["t1"] - e["t0"] for e in segs) or 1e-9
        if sum(e["t1"] - e["t0"] for e in segs if sum(e["ocr"]) >= 1.5 and max(e["ocr"]) / sum(e["ocr"]) >= 0.7 and int(np.argmax(e["ocr"])) != i) / dur >= 0.15:
            tags["identity contradiction"].add((k, al))
        if sum(e["t1"] - e["t0"] for e in segs if sum(e["ocr"]) == 0) / dur >= 0.8:
            tags["robot mostly unread (>= 80 % of track time)"].add((k, al))
    # volley-heavy: >= 60 % of the alliance's scouted balls in bursts of 20+
    tl = TL(R[k]); n20 = ntot = 0
    for t in tm:
        for row in [r_ for r_ in bym[k].get(t, []) if time_valid(r_)]:
            st_ = np.sort(row.shots)
            for b in np.split(st_, np.where(np.diff(st_) > 1.5)[0] + 1) if len(st_) else []:
                ntot += len(b); n20 += len(b) * (len(b) >= 20)
    if ntot and n20 / ntot >= 0.6:
        tags["volley-heavy (>= 60 % of balls in 20+ bursts)"].add((k, al))
allerr = np.array([x["fuel_err"] for x in pa.values()]); mean_all = allerr.mean(); mass = allerr.sum()
rows = []
for t, s_ in tags.items():
    e_in = np.array([pa[a]["fuel_err"] for a in s_]); e_out = np.array([pa[a]["fuel_err"] for a in pa if a not in s_])
    excess = (e_in.mean() - e_out.mean()) * len(e_in) if len(e_out) else 0.0
    rows.append((excess / mass, t, len(e_in), e_in.mean(), e_out.mean()))
print(f"   all {len(pa)} alliances: mean team-match fuel error {mean_all:.1f}")
for ex, t, n, a, b in sorted(rows, reverse=True):
    print(f"   {t:46s} n {n:3d}  error {a:5.1f} vs {b:5.1f} without  -> excess {100 * ex:5.1f} % of all error")
json.dump({k: {f"{a[0]}|{a[1]}": v for a, v in PA[k].items()} for k in PA}, open(AUDIT / "c26_v4.json", "w"), default=float)
