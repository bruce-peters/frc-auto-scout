"""Before/after the end-camera recovery: pipeline.analyse(md, Options()) in memory on the copies in
data/exp_calib/{before,after}/<match>/scan.pkl.

Per match and alliance: team fuel, share error vs fuel OPR (total variation, as data/exp_audit c3f),
credited launches (toward the own hub, linked to a team) and how many came from the own end view;
held-out identity (scripts/eval_analysis.py --holdout logic: identities from half the bumper reads,
scored on the other half); for scouted matches, scripts/score_scouting.py's comparison with 5026's
human scouting (results written to data/exp_calib/results/<state>/<match>/result.json and scored there).

OCR: --ocr none (default) makes no new bumper reads (cached reads only, identical for both states, no
GPU, as the failure-finder tools did); --ocr cpu reads uncached crops with EasyOCR on the CPU.
Usage: measure.py [--ocr none|cpu] MATCH [MATCH ...]
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

from autoscout import ocr, opr, pipeline as P, tba  # noqa: E402

EXP = ROOT / "data" / "exp_calib"
ap = argparse.ArgumentParser()
ap.add_argument("matches", nargs="+")
ap.add_argument("--ocr", default="none", choices=["none", "cpu"])
ap.add_argument("--out", default=str(EXP / "measure.json"))
a = ap.parse_args()
N_NEW = [0]
if a.ocr == "none":
    def _no_read(img):
        N_NEW[0] += 1
        return []
    ocr.read_numbers = _no_read
else:
    import hashlib
    _read = ocr.read_numbers
    _MEMO: dict = {}  # image hash -> reads (each analysis starts from its pickle's cache, as eval_analysis does)

    def _count_read(img):
        h = hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()
        if h not in _MEMO:
            N_NEW[0] += 1
            _MEMO[h] = _read(img)
        return _MEMO[h]
    ocr.read_numbers = _count_read

OWN_END = {"blue": "blue_end", "red": "red_end"}
_OPR: dict = {}


def load(state, key):
    with open(EXP / state / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    P.load_dino(md, EXP / state / key)
    return md


def run(state, key, holdout=None):
    md = load(state, key)
    n0 = N_NEW[0]
    tic = time.time()
    res = P.analyse(md, P.Options(ocr_holdout=holdout) if holdout is not None else P.Options())
    cal = {k: getattr(c, "recovered", None) or "tags" for k, c in md.cams.items()}
    info = md.info
    del md
    gc.collect()
    return res, info, cal, N_NEW[0] - n0, time.time() - tic


out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
for key in a.matches:
    row = {}
    for state in ("before", "after"):
        res, info, cal, nnew, dt = run(state, key)
        ev = info.event_key
        if ev not in _OPR:
            _OPR[ev] = opr.fuel_opr(ev)
        op = _OPR[ev]
        st = {"calibrated": cal, "new_ocr_reads": nnew, "seconds": round(dt, 1), "alliances": {}}
        for al in ("red", "blue"):
            teams = info.teams(al)
            f = np.array([res["teams"][t]["fuel"] for t in teams], float)
            o = np.array([op.get(t, np.nan) for t in teams], float)
            err = float(np.abs(f / f.sum() - o / o.sum()).sum() / 2) if f.sum() > 0 and np.all(np.isfinite(o)) else None
            sh = [s for s in res["shots"] if s["alliance"] == al and s["toward"]]
            st["alliances"][al] = {
                "teams": teams, "fuel": f.round(1).tolist(), "opr": o.round(2).tolist(), "share_err_opr": err,
                "credited": sum(1 for s in sh if s["team"] is not None),
                "credited_own_end": sum(1 for s in sh if s["team"] is not None and s["view"] == OWN_END[al]),
                "credited_by_team": [sum(1 for s in sh if s["team"] == t) for t in teams],
                "unlinked": sum(1 for s in sh if s["team"] is None)}
        # held-out identity (eval_analysis.py --holdout)
        agree = {"red": 0.0, "blue": 0.0}
        mass = {"red": 0.0, "blue": 0.0}
        for k in (0, 1):
            r, info, _, nn, _ = run(state, key, holdout=k)
            st["new_ocr_reads"] += nn
            for al in ("red", "blue"):
                teams = info.teams(al)
                for s in r["identity"][al].values():
                    if s["team"] is None:
                        continue
                    agree[al] += s["held"][teams.index(s["team"])]
                    mass[al] += sum(s["held"])
        st["holdout"] = {al: {"agree": agree[al], "mass": mass[al]} for al in agree}
        # result.json for score_scouting.py
        d = EXP / "results" / state / key
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, default=P._json_default), encoding="utf-8")
        row[state] = st
        al_s = " | ".join(f"{al}: err {x['share_err_opr'] if x['share_err_opr'] is None else round(x['share_err_opr'], 3)} "
                          f"credited {x['credited']} (own end {x['credited_own_end']}) id "
                          f"{100 * agree[al] / max(mass[al], 1e-9):.1f}%" for al, x in st["alliances"].items())
        print(f"{key} {state:6s} {cal}  {al_s}  new OCR reads {st['new_ocr_reads']}  ({dt:.0f}s/analyse)", flush=True)
    out[key] = row
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
