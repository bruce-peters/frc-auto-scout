"""Team summary across all processed matches: per-match fuel/points table and all auto paths.

Red-alliance paths are rotated 180 deg (point symmetry of the field) so every auto is drawn
as if the team started on the blue side, which makes repeated auto routines line up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, opr, report, tba  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=5026)
    a = ap.parse_args()
    team = a.team
    rows = []
    events = sorted({f.parent.name.split("_")[0] for f in config.OUTPUT.glob("*/result.json")})
    ncol = 2
    nrow = (len(events) + 1) // 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(16, 4.4 * nrow))
    axs = np.atleast_1d(axs).ravel()
    ax_of = {ev: axs[i] for i, ev in enumerate(events)}
    for ev, ax_ in ax_of.items():
        report.draw_field(ax_)
        ax_.set_title(ev)
    for ax_ in axs[len(events):]:
        ax_.axis("off")
    cmap = plt.get_cmap("tab20")
    k = 0
    oprs = {}
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        t = r["teams"].get(str(team))
        if not t:
            continue
        info = tba.match(r["match"])
        ev = info.event_key
        if ev not in oprs:
            oprs[ev] = opr.fuel_opr(ev)
        meta = r["teams"][f"_{t['alliance']}"]
        rows.append({"match": r["match"], "alliance": t["alliance"], "fuel": t["fuel"],
                     "fuel_video": t.get("fuel_video", t["fuel"]),
                     "auto": t["fuel_by_period"].get("auto", 0.0), "points": t["points"],
                     "alliance_fuel": meta.get("tba_total"), "opr": oprs[ev].get(team, float("nan")),
                     "tower": f"{t.get('auto_tower', '')}/{t.get('endgame_tower', '')}"})
        pts = np.array([p for p in r["paths"].get(str(team), []) if p[0] is not None and -0.5 <= p[0] <= 20])
        if len(pts) >= 3:
            x, y = pts[:, 1], pts[:, 2]
            if t["alliance"] == "red":
                x, y = config.FIELD_LENGTH - x, config.FIELD_WIDTH - y
            ax = ax_of[r["match"].split("_")[0]]
            c = cmap(len(ax.lines) // 2 % 20)
            ax.plot(x, y, "-", color=c, lw=1.6, alpha=0.85, label=r["match"].split("_", 1)[1])
            ax.plot(x[0], y[0], "o", color=c, ms=6)
            k += 1
    for ev, ax in ax_of.items():
        if ax.lines:
            ax.legend(fontsize=6, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.0))
    fig.suptitle(f"Team {team}: autonomous paths (0-20 s), drawn as if on the blue alliance; o = start")
    fig.tight_layout()
    out = config.OUTPUT / f"team_{team}_autos.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines = [f"# Team {team} — video scouting summary", "",
             f"![auto paths](team_{team}_autos.png)", "",
             "fuel scored = from that match's video alone (no OPR or team history); team fuel OPR for reference.", "",
             "| match | alliance | fuel scored | auto fuel | tower (auto/end) | points | alliance fuel (TBA) | team fuel OPR |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for x in rows:
        lines.append(f"| {x['match']} | {x['alliance']} | {x['fuel']:.0f} | {x['auto']:.0f} | {x['tower']} | "
                     f"{x['points']:.0f} | {x['alliance_fuel']} | {x['opr']:.0f} |")
    if rows:
        by_ev = {}
        for x in rows:
            by_ev.setdefault(x["match"].split("_")[0], []).append(x)
        lines += ["", "| event | matches | avg fuel | fuel OPR |", "| --- | ---: | ---: | ---: |"]
        for ev, xs in by_ev.items():
            lines.append(f"| {ev} | {len(xs)} | {np.mean([x['fuel'] for x in xs]):.0f} | {xs[0]['opr']:.0f} |")
    (config.OUTPUT / f"team_{team}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
