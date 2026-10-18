"""Segments a team loses between two configs (non-holdout runs): their bumper reads, probe vote, duration.
  switched.py 2026cc_qm25 9032"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import eval_ident as E  # noqa: E402
from autoscout import pipeline  # noqa: E402

key, team = sys.argv[1], int(sys.argv[2])
with open(ROOT / "output" / key / "scan.pkl", "rb") as fh:
    md = pickle.load(fh)
with open(E.OCRC / f"{key}.pkl", "rb") as fh:
    md.ocr_cache.update(pickle.load(fh))
pipeline.DINO_PROBE_DIR = E.EXP / "probe" / "box_inner_ocr"
lp = pipeline.load_dino_probe(key)
A = dict(absent_cost_dead=2, absent_cost=2, absent_near_bonus=2)
base = E.identity_only(md, pipeline.Options())
res = pipeline.analyse(md, pipeline.Options(w_dino=2, **A))
seg_crops = {}
for d, r in zip(md.dets, md.det_rows):
    if d.crop_id is not None and r[8] is not None:
        seg_crops.setdefault(int(r[8]), []).append(d.crop_id)
al = "red" if team in md.info.red else "blue"
teams = md.info.teams(al)
lost = moved = 0.0
for sid, s in base[al].items():
    n = res["identity"][al][sid]
    if s["team"] != team or n["team"] == team:
        continue
    rows = np.array([[lp[c].get(t, np.nan) for t in teams] for c in seg_crops.get(int(sid), []) if c in lp]).reshape(-1, 3)
    q = np.exp(rows.mean(0)) if len(rows) else None
    lost += s["t1"] - s["t0"]
    conc = [o["team"] for oid, o in res["identity"][al].items() if oid != sid and o["t0"] < s["t1"] and o["t1"] > s["t0"]]
    print(f"seg {sid:>4} slot {s['slot']} {s['t0']:6.1f}-{s['t1']:6.1f} ocr {s['ocr']} -> {n['team']}  probe q "
          f"{None if q is None else np.round(q / q.sum(), 2).tolist()} (n {len(rows)})  concurrent labels {conc}")
print(teams, f"lost {lost:.0f} s")
