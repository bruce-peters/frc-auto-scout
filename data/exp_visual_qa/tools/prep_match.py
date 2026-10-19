"""Run the pipeline in memory (today's defaults, output/ untouched) for some matches and cache what the frame
renderer needs: result.json, dets.json (md.det_rows), tracks.pkl (every flight track per view, split at jumps the
way analyse() does) and meta.json (production, team-of-segment). One scan in memory at a time.

  .venv/Scripts/python data/exp_visual_qa/tools/prep_match.py 2026cc_qm17 2026cc_qm55 ...
"""
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, fuel, pipeline as P  # noqa: E402

DST = ROOT / "data/exp_visual_qa/cache"
for key in sys.argv[1:]:
    tic = time.time()
    od = DST / key
    if (od / "tracks.pkl").exists():
        print(key, "cached")
        continue
    with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    P.load_dino(md, config.OUTPUT / key)
    res = P.analyse(md, P.Options())
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(res, default=P._json_default))
    (od / "dets.json").write_text(json.dumps(md.det_rows, default=P._json_default))
    tr = {}
    for vname, tracks in md.tracks_all.items():
        tr[vname] = [[(round(p[0], 4), round(p[1], 1), round(p[2], 1)) for p in t.pts]
                     for t in (fuel.split_at_jumps(tracks) if P.SPLIT_TRACKS else tracks)]
    with open(od / "tracks.pkl", "wb") as fh:
        pickle.dump(tr, fh)
    (od / "meta.json").write_text(json.dumps({"prod": md.prod, "teams_red": md.info.red, "teams_blue": md.info.blue,
                                              "n_tracks": {k: len(v) for k, v in tr.items()}}))
    del md
    print(key, f"{time.time() - tic:.0f}s", flush=True)
