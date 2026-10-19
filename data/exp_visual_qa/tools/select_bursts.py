"""Pick the bursts to inspect from find_bursts.py output on the in-memory cache: ~20 with the biggest
scouted-vs-credited disagreement (at most 3 per match, about 10 matches) and 10 random controls.

  .venv/Scripts/python data/exp_visual_qa/tools/select_bursts.py [--src bursts_cache.json] [--out sel.json]
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data/exp_visual_qa/bursts_cache.json"))
ap.add_argument("--out", default=str(ROOT / "data/exp_visual_qa/sel.json"))
ap.add_argument("--n_dis", type=int, default=20)
ap.add_argument("--n_ctl", type=int, default=10)
ap.add_argument("--max_matches", type=int, default=10)
ap.add_argument("--per_match", type=int, default=3)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

B = [b for b in json.loads(Path(a.src).read_text()) if b["tv"] is not None]


def tag(b, kind):
    return f"{b['match'].replace('2026cc_', '')}_{b['alliance'][0]}_{b['t0']:.0f}_{kind}"


cand = []
for b in B:
    i = b["teams"].index(b["scout_top"])
    if (b["K"] >= 8 and b["k_scouted"] >= 0.8 * b["K"] and b["sshare"][i] >= 0.6
            and b["cshare"][i] <= b["sshare"][i] - 0.4):
        cand.append(b)
cand.sort(key=lambda b: -b["moved"])
print(len(cand), "disagreement candidates")
# make sure one 5026 alliance is in, if it has a candidate
pick, per = [], {}
c5026 = [b for b in cand if 5026 in b["teams"]]
order = c5026[:1] + [b for b in cand if b not in c5026[:1]]
for b in order:
    m = b["match"]
    if per.get(m, 0) >= a.per_match or (m not in per and len(per) >= a.max_matches):
        continue
    pick.append(b)
    per[m] = per.get(m, 0) + 1
    if len(pick) >= a.n_dis:
        break
rng = random.Random(a.seed)
pool = [b for b in B if b["K"] >= 5 and b not in pick and b["k_scouted"] >= 0.8 * b["K"]]
ctl = rng.sample(pool, a.n_ctl)
sel = [{"tag": tag(b, "D"), "kind": "dis", **b} for b in pick] + [{"tag": tag(b, "C"), "kind": "ctl", **b} for b in ctl]
sel.sort(key=lambda s: (s["match"], s["t0"]))
Path(a.out).write_text(json.dumps(sel, indent=1))
for s in sel:
    print(s["tag"], s["K"], s["teams"], "scout", s["sshare"], "credit", s["cshare"], "moved", s["moved"], "launch",
          s["n_launch"], "vis", s["vis"])
print(len(per), "matches:", sorted(per))
