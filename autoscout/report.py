"""Match report: per-team table, auto-path plot, HTML page."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config, field as fieldmod  # noqa: E402

TEAM_COLORS = {"red": ["#d62728", "#ff7f0e", "#8c564b"], "blue": ["#1f77b4", "#17becf", "#9467bd"]}


def draw_field(ax):
    L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
    ax.add_patch(plt.Rectangle((0, 0), L, W, fill=True, fc="#e8e8e8", ec="k", lw=1.5))
    ax.plot([L / 2, L / 2], [0, W], color="w", lw=2)
    z = fieldmod.ALLIANCE_ZONE_DEPTH
    ax.plot([z, z], [0, W], color="#1f77b4", lw=1, ls="--")
    ax.plot([L - z, L - z], [0, W], color="#d62728", lw=1, ls="--")
    for al, c in (("blue", "#1f77b4"), ("red", "#d62728")):
        cx, cy = fieldmod.hub_center(al)
        h = fieldmod.HUB_HALF
        ax.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h, fc=c, alpha=0.5))
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, W + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_paths(res: dict, path: Path, t_max: float = 20.0, title: str = "Auto paths (0-20 s)"):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    draw_field(ax)
    for al in ("red", "blue"):
        teams = [int(t) for t, v in res["teams"].items() if not str(t).startswith("_") and v["alliance"] == al]
        for i, team in enumerate(sorted(teams)):
            pts = res["paths"].get(str(team)) or res["paths"].get(team) or []
            pts = np.array([p for p in pts if p[0] is not None and -1 <= p[0] <= t_max])
            if len(pts) == 0:
                continue
            c = TEAM_COLORS[al][i % 3]
            ax.plot(pts[:, 1], pts[:, 2], "-", color=c, lw=2, label=f"{team}")
            ax.plot(pts[0, 1], pts[0, 2], "o", color=c, ms=8)
            ax.plot(pts[-1, 1], pts[-1, 2], "s", color=c, ms=7)
    ax.legend(loc="upper center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{res['match']} — {title}  (o start, ■ end)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_fuel_map(fmap: np.ndarray, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    draw_field(ax)
    L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
    ax.imshow(fmap.T, origin="lower", extent=(0, fmap.shape[0] * 0.25, 0, fmap.shape[1] * 0.25),
              cmap="YlOrBr", alpha=0.85, vmin=0, vmax=max(1e-6, np.percentile(fmap, 99)), zorder=2)
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, W + 0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_fuel_timeline(res: dict, path: Path):
    """Cumulative attributed fuel per team over match time (one panel per alliance)."""
    log = res.get("fuel_log") or []
    if not log:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
    for ax, al in zip(axes, ("red", "blue")):
        teams = sorted(int(t) for t, v in res["teams"].items() if not str(t).startswith("_") and v["alliance"] == al)
        for i, team in enumerate(teams):
            ev = sorted((t, n) for t, tm, n in log if tm == team)
            if not ev:
                continue
            t = np.array([e[0] for e in ev])
            c = np.cumsum([e[1] for e in ev])
            ax.step(t, c, where="post", color=TEAM_COLORS[al][i % 3], lw=2, label=str(team))
        for x in (20, 30, 55, 80, 105, 130):
            ax.axvline(x, color="#bbb", lw=0.8, ls=":")
        ax.set_title(f"{al} alliance: cumulative fuel by team")
        ax.set_xlabel("match time (s)  [auto 0-20 | transition | shifts 1-4 | endgame 130-160]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def write_html(res: dict, out_dir: Path):
    plot_paths(res, out_dir / "auto_paths.png")
    tl_html = ""
    if plot_fuel_timeline(res, out_dir / "fuel_timeline.png"):
        tl_html = "<h3>When each team scored</h3><img src='fuel_timeline.png' width='1000'>"
    fmap_html = ""
    if (out_dir / "fuel_map.npy").exists():
        plot_fuel_map(np.load(out_dir / "fuel_map.npy"), out_dir / "fuel_map.png",
                      f"{res['match']} — where FUEL sat on the field (main camera, time-averaged)")
        fmap_html = "<h3>Fuel on the field</h3><img src='fuel_map.png' width='900'>"
    plot_paths(res, out_dir / "full_paths.png", t_max=200, title="Full-match paths")
    rows = []
    for al in ("red", "blue"):
        meta = res["teams"].get(f"_{al}", {})
        for t, v in res["teams"].items():
            if str(t).startswith("_") or v["alliance"] != al:
                continue
            per = v.get("fuel_by_period", {})
            rows.append(
                f"<tr class='{al}'><td>{t}</td><td>{al}</td><td><b>{v['fuel']:.0f}</b></td>"
                f"<td>{per.get('auto', 0):.0f}</td><td>{sum(x for k, x in per.items() if k != 'auto'):.0f}</td>"
                f"<td>{v.get('auto_tower', '')}/{v.get('endgame_tower', '')}</td>"
                f"<td>{v.get('points', 0):.0f}</td><td>{v.get('shots_detected', 0)}</td></tr>")
        rows.append(f"<tr class='meta'><td colspan=8>{al}: source={meta.get('source')} "
                    f"timeline total={meta.get('timeline_total')} TBA total={meta.get('tba_total')}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(res['match'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px}}table{{border-collapse:collapse}}
td,th{{padding:4px 10px;border-bottom:1px solid #ddd}}tr.red td:first-child{{color:#c00}}
tr.blue td:first-child{{color:#06c}}tr.meta td{{color:#666;font-size:12px}}</style></head><body>
<h2>{html.escape(res['match'])}</h2>
<table><tr><th>Team</th><th>Alliance</th><th>Fuel scored</th><th>Auto fuel</th><th>Teleop fuel</th>
<th>Tower (auto/end)</th><th>Points</th><th>Shots seen</th></tr>{''.join(rows)}</table>
{tl_html}
<h3>Auto paths</h3><img src='auto_paths.png' width='900'>
<h3>Full match</h3><img src='full_paths.png' width='900'>
{fmap_html}
</body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")


def load(match_key: str) -> dict:
    return json.loads((config.OUTPUT / match_key / "result.json").read_text())


def write_index(team: int | None = 5026):
    """output/index.html: one row per processed match, highlighting `team`."""
    rows = []
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        key = r["match"]
        cells = []
        for al in ("red", "blue"):
            meta = r["teams"].get(f"_{al}", {})
            ts = [(t, v) for t, v in r["teams"].items() if not str(t).startswith("_") and v["alliance"] == al]
            txt = ", ".join((f"<b>{t}</b>" if str(t) == str(team) else str(t)) + f" {v['fuel']:.0f}" for t, v in ts)
            cells.append(f"<td class='{al}'>{txt}</td><td>{meta.get('tba_total')}</td>")
        rows.append(f"<tr><td><a href='{key}/report.html'>{key}</a></td>{''.join(cells)}</tr>")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>auto-scout</title><style>"
            "body{font-family:system-ui,sans-serif;margin:20px}td,th{padding:3px 10px;border-bottom:1px solid #ddd}"
            "td.red{color:#a00}td.blue{color:#05a}</style></head><body><h2>auto-scout match reports</h2>"
            "<table><tr><th>match</th><th>red: team fuel</th><th>TBA</th><th>blue: team fuel</th><th>TBA</th></tr>"
            + "".join(rows) + "</table></body></html>")
    (config.OUTPUT / "index.html").write_text(page, encoding="utf-8")
