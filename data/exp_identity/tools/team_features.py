"""Per team-match features that might tell a dead / absent robot from a real one whose bumper was never
read (scouted cc quals). Runs analyse (base labels, via proto with nothing enabled) per match.

Features per team: own reads (all segments), foreign reads on its labelled segments, labelled seconds,
seconds visible, launches, 'near-mate' fraction (labelled time with a teammate-labelled concurrent
segment within NEAR m), median 20 s movement extent, credit source L/P/H.
Writes data/exp_identity/team_features.json.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402

import proto  # noqa: E402
from absent_eval import wait_mem  # noqa: E402
from autoscout import pipeline, scouting  # noqa: E402

NEAR = 2.0
bym = scouting.by_match(scouting.load())
keys = sorted(p.parent.name for p in (ROOT / "output").glob("2026cc_qm*/scan.pkl") if p.parent.name in bym)
proto.install()
import absent_eval as AE  # noqa: E402
from autoscout import ocr  # noqa: E402
ocr.read_numbers = AE._stub_read  # uncached reads: residual injection from result.json (absent_eval.prepare)
AE._ORIG_TE = ocr.team_evidence
ocr.team_evidence = AE._team_evidence
OUTP = ROOT / "data" / "exp_identity" / "team_features.json"
rows = json.loads(OUTP.read_text()) if OUTP.exists() else []
done = {r["key"] for r in rows}
for key in keys:
    if key in done:
        continue
    wait_mem()
    tic = time.time()
    md = pickle.load(open(ROOT / "output" / key / "scan.pkl", "rb"))
    md.ocr_cache = dict(getattr(md, "ocr_cache", None) or {})
    sp = ROOT / "data" / "exp_identity" / "ocr_sidecar" / f"{key}.pkl"
    if sp.exists():
        md.ocr_cache.update(pickle.load(open(sp, "rb")))
    inj = AE.prepare(md, key)
    proto.CFG.update({"abs_cost_dead": np.inf, "zero": "none"})
    proto.CFG["on"] = True
    proto.reset(md)
    res = pipeline.analyse(md, pipeline.Options())
    for al in ("red", "blue"):
        st = proto.CTX["al"].get(al)
        if st is None:
            continue
        tm = md.info.teams(al)
        segs, lab = st["segs"], st["labels"]
        ide = res["identity"][al]
        raw = {int(k): np.array(v["ocr"]) for k, v in ide.items()}
        grid = np.arange(min(s.t0 for s in segs), max(s.t1 for s in segs), 0.2)
        # interpolated positions per segment on the grid
        pos = {}
        for s in segs:
            p = np.array(s.points)
            m = (grid >= s.t0) & (grid <= s.t1)
            if m.sum() and len(p):
                pos[s.id] = (m, np.interp(grid[m], p[:, 0], p[:, 1]), np.interp(grid[m], p[:, 0], p[:, 2]))
        for ti, t in enumerate(tm):
            mine = [s for s in segs if lab.get(s.id) == ti]
            others = [s for s in segs if lab.get(s.id) is not None and lab.get(s.id) != ti]
            own_all = float(sum(raw.get(s.id, np.zeros(3))[ti] for s in segs))
            foreign = float(sum(raw.get(s.id, np.zeros(3))[u] for s in mine for u in range(3) if u != ti))
            lab_t = np.zeros(len(grid), bool)
            near_t = np.zeros(len(grid), bool)
            for s in mine:
                if s.id not in pos:
                    continue
                m, x, y = pos[s.id]
                lab_t |= m
                idx = np.where(m)[0]
                for o in others:
                    if o.id not in pos:
                        continue
                    mo, xo, yo = pos[o.id]
                    both = m & mo
                    if not both.any():
                        continue
                    ia = np.searchsorted(idx, np.where(both)[0])
                    io = np.searchsorted(np.where(mo)[0], np.where(both)[0])
                    d = np.hypot(x[ia] - xo[io], y[ia] - yo[io])
                    near_t[np.where(both)[0][d < NEAR]] = True
            pts = np.array(sorted(p for s in mine for p in s.points)) if mine else np.zeros((0, 3))
            ext = []
            for a in np.arange(grid[0], grid[-1] - 20, 10):
                mm = (pts[:, 0] >= a) & (pts[:, 0] <= a + 20) if len(pts) else np.zeros(0, bool)
                if mm.sum() >= 3:
                    q = pts[mm, 1:]
                    ext.append(float(np.hypot(*(q.max(0) - q.min(0)))))
            src = proto.CTX.get("src", {}).get(t, np.zeros(3))
            sc = bym[key].get(t, [])
            rows.append({"key": key, "al": al, "team": t, "own": round(own_all, 2), "foreign": round(foreign, 2),
                         "mates_own": [round(float(sum(raw.get(s.id, np.zeros(3))[u] for s in segs)), 1) for u in range(3) if u != ti],
                         "lab_s": round(float(lab_t.sum() * 0.2), 1), "near_frac": round(float(near_t.sum() / max(lab_t.sum(), 1)), 3),
                         "launches": int(sum(1 for q in res["shots"] if q["team"] == t)),
                         "ext_med": round(float(np.median(ext)), 2) if ext else None,
                         "ext_q25": round(float(np.quantile(ext, 0.25)), 2) if ext else None,
                         "src": np.round(src / max(src.sum(), 1e-9), 2).tolist(),
                         "fuel": res["teams"][t]["fuel"],
                         "scout": [x.n for x in sc], "comments": [x.comments for x in sc], "inject": inj})
    OUTP.write_text(json.dumps(rows, indent=0, default=float), encoding="utf-8")
    print(f"{key} {time.time() - tic:.0f}s", flush=True)
    md = None
