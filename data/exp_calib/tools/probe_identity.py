"""Held-out identity split by where a segment's detections come from (main only / with end-view
boxes), before vs after, so a drop can be told apart from a change of composition. Read-only.
Usage: probe_identity.py STATE MATCH [MATCH ...]     (STATE: before | after | a scan dir)
"""
import os
import pickle
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import ocr, pipeline as P  # noqa: E402

ocr.read_numbers = lambda img: []
state = sys.argv[1]
base = ROOT / "data" / "exp_calib" / state if (ROOT / "data" / "exp_calib" / state).exists() else Path(state)
for key in sys.argv[2:]:
    tot = {}
    for k in (0, 1):
        with open(base / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        r = P.analyse(md, P.Options(ocr_holdout=k))
        views_of = {}
        for row in md.det_rows:  # (t, view, x0, y0, x1, y1, alliance, team, sid, fill)
            if row[8] is not None:
                views_of.setdefault(row[8], set()).add(row[1])
        for al in ("red", "blue"):
            teams = md.info.teams(al)
            for sid, s in r["identity"][al].items():
                if s["team"] is None or sum(s["held"]) == 0:
                    continue
                vs = views_of.get(int(sid), set())
                kind = "main_only" if vs <= {"main"} else ("end_only" if "main" not in vs else "main+end")
                own = f"{al}_end" in vs
                for grp in (f"{al}:{kind}", f"{al}:{'with_own_end' if own else 'no_own_end'}", f"{al}:all"):
                    a, m = tot.get(grp, (0.0, 0.0))
                    tot[grp] = (a + s["held"][teams.index(s["team"])], m + sum(s["held"]))
        del md
    print(key, state)
    for g in sorted(tot):
        a, m = tot[g]
        print(f"   {g:22s} held-out agreement {100 * a / m:5.1f} %  (mass {m:6.1f})")
