"""Markdown rows for the A/B (data/exp_multiview/ab/ab_summary.json) and the 254 burst check per arm
(data/exp_audit/tools/team_volleys.py 254 --out <arm folder>)."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AB = ROOT / "data" / "exp_multiview" / "ab"
s = json.load(open(AB / "ab_summary.json"))
hs = json.load(open(AB / "hook_stats.json"))["stats(seen, with fit, picked, changed)"]


def row(name, sc):
    return (f"| {name} | {sc['vonly']['mae']:.2f} | {sc['vonly']['corr']:.3f} | {100 * sc['top_scorer']['vonly']:.0f} % | "
            f"{sc['video']['mae']:.2f} | {100 * sc['top_scorer']['video']:.0f} % | {100 * sc['burst_overlap']['video']:.1f} % |")


print("| arm | video-only fuel MAE | video-only corr | top scorer (video only) | best-estimate MAE | top scorer (best) | burst |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
print(row("today", s["base"]))
for name, v in s.items():
    if name == "base":
        continue
    print(row(name, v["score"]))
    d = v["paired"]
    f = lambda k, m=1.0, nd=2: f"{m * d[k][0]:+.{nd}f} [{m * d[k][1]:+.{nd}f}, {m * d[k][2]:+.{nd}f}]"
    print(f"| paired diff | {f('vonly_mae')} | {f('vonly_corr', nd=3)} | {f('vonly_top', 100, 1)} pp | {f('video_mae')} | "
          f"{f('video_top', 100, 1)} pp | |")
    print(f"   {name}: better {v['better']} / worse {v['worse']} of {v['n']}; hook (seen, with fit, picked, changed) {hs.get(name)}")
for arm in ["base"] + [n for n in s if n != "base"]:
    out = subprocess.run([sys.executable, str(ROOT / "data" / "exp_audit" / "tools" / "team_volleys.py"), "254", "--out", str(AB / arm)],
                         capture_output=True, text=True).stdout
    m = re.search(r"team 254: (\d+) jumps, (\d+) balls.*scouts give it (\d+), credit gives it (\d+)", out)
    m2 = re.search(r"credit is > 0.3 lower: (\d+), (\d+) balls lost", out)
    print(f"254 burst check {arm}: {m.groups() if m else out[:200]}  lost jumps/balls {m2.groups() if m2 else None}")
