"""Debug one alliance: per-segment raw reads, final evidence and labels under base and a proto variant.

  debug_alliance.py MATCH ALLIANCE [cfg-json]
"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402

import proto  # noqa: E402
from autoscout import pipeline  # noqa: E402

key, al = sys.argv[1], sys.argv[2]
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"abs_cost_dead": 2.0}
md = pickle.load(open(ROOT / "output" / key / "scan.pkl", "rb"))
proto.install()
EV = {}
orig_hook = proto.assign_hook


def spy(segs, evidence, switch_cost=0.6):
    if segs and segs[0].alliance == al:
        EV.setdefault("calls", []).append({k: np.asarray(v, float).copy() for k, v in evidence.items()})
    return orig_hook(segs, evidence, switch_cost)


import autoscout.identity as I  # noqa: E402
I.assign = spy
proto.reset(md)
proto.CFG["on"] = False
r0 = pipeline.analyse(md, pipeline.Options())
base_calls = EV.pop("calls")
proto.CFG.update(cfg)
proto.CFG["on"] = True
proto.reset(md)
r1 = pipeline.analyse(md, pipeline.Options())
st = proto.CTX["al"][al]
teams = md.info.teams(al)
print(key, al, teams, "dead", st["dead"], "own", np.round(st["own"], 1))
print("absent", {teams[t]: [(round(a, 1), round(b, 1)) for a, b in iv] for t, iv in st["absent"].items()})
i0, i1 = r0["identity"][al], r1["identity"][al]
fin = EV["calls"][-1]
for sid, e in sorted(i1.items(), key=lambda x: x[1]["t0"]):
    s0 = i0[sid]
    ev = fin.get(int(sid))
    mark = " <<" if s0["team"] != e["team"] else ""
    print(f"{sid:>4s} sl{e['slot']} {e['t0']:6.1f}-{e['t1']:6.1f} reads {e['ocr']} base {s0['team']} new {e['team']} "
          f"ev {None if ev is None else np.round(ev, 2).tolist()}{mark}")
for t in teams:
    print(t, r0["teams"][t]["fuel"], "->", r1["teams"][t]["fuel"])
