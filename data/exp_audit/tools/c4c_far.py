"""Far-contradicted segments: how much credit moves, and examples with the concurrent segment holding the read team."""
import copy, json, sys
import numpy as np
from common import AUDIT, P, TL, load, path_video_times, resplit, tba
src = sys.argv[1] if len(sys.argv) > 1 else "v2"
D = json.load(open(AUDIT / f"c4_identity_{src}.json"))
moved = total = 0.0
rows = []
for k, bad in D["bad"].items():
    r = load(k, src); info = tba.match(k); tl = TL(r)
    paths = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in info.red + info.blue}
    r2 = copy.deepcopy(r)
    for sid, kind, a, b, assigned, should in bad:
        if should is None or assigned is None:
            continue
        d = []
        for t in np.arange(a, b + 1e-6, 0.5):
            x = P._track_position(paths[should], t) if len(paths[should]) else None
            y = P._track_position(paths[assigned], t) if len(paths[assigned]) else None
            if x is not None and y is not None and x[2] and y[2]:
                d.append(np.hypot(x[0] - y[0], x[1] - y[1]))
        if not d or np.median(d) < 1.0:
            continue
        al = "red" if assigned in info.red else "blue"
        ide = r["identity"][al]
        conc = [(int(s2), e["team"], e["ocr"], round(e["t0"], 1), round(e["t1"], 1)) for s2, e in ide.items()
                if e["team"] == should and e["t0"] < b and e["t1"] > a]
        nl = 0
        for s, s2 in zip(r["shots"], r2["shots"]):
            if s["team"] == assigned and s["toward"] and a - 0.3 <= s["t"] <= b + 0.3:
                s2["team"] = should; nl += 1
        rows.append({"key": k, "sid": sid, "kind": kind, "t0": round(a, 1), "t1": round(b, 1), "assigned": assigned,
                     "reads": should, "own_ocr": ide[str(sid)]["ocr"], "teams": info.teams(al), "dist": round(float(np.median(d)), 2),
                     "concurrent_read_team_segs": conc, "launches": nl})
    f0, f1 = resplit(r, info), resplit(r2, info)
    moved += sum(abs(f1[t] - f0[t]) for t in f0) / 2; total += sum(f0.values())
print(f"far-contradicted segments: {len(rows)}; credit moved by re-crediting their launches: {moved:.0f} of {total:.0f} balls "
      f"({100 * moved / total:.2f} %)")
rows.sort(key=lambda x: -x["launches"])
for x in rows[:12]:
    print(" ", x)
json.dump(rows, open(AUDIT / f"c4c_far_{src}.json", "w"), indent=1, default=float)
