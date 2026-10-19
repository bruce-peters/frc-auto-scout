"""One-page status figure (2026-10-18 evening): accuracy over the day, ball-tracking recall, headline numbers and
what is adopted / ruled out / running. Numbers are copied from the write-ups in data/exp_*/RESULTS_section.md.

  .venv/Scripts/python data/exp_audit/tools/status_figure.py OUT.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, PANEL, INK, INK2, INK3 = "#0a0a0a", "#141414", "#f2f2f2", "#b4b4b4", "#7a7a7a"
ACC, GREY, RUN = "#ffcb1f", "#4a4a4a", "#ff9a3c"
plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": PANEL, "figure.facecolor": BG, "axes.facecolor": PANEL})

fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1.25, 1], hspace=0.42, wspace=0.28,
                      left=0.05, right=0.97, top=0.82, bottom=0.07)
fig.text(0.05, 0.945, "auto-scout: where it stands", fontsize=26, weight="bold")
fig.text(0.05, 0.905, "FRC 2026 REBUILT broadcast scouting. Scored against FRC 5026's human scouting of 54 Chezy Champs "
                      "quals, held out, with no hindsight (nothing from an event's later matches).", fontsize=12, color=INK2)

# 1) per-robot error over the day
ax = fig.add_subplot(gs[0, :2])
steps = ["this morning", "learned credit\nmodel", "DINO robot\nrecognition", "no hindsight\n(live-safe)",
         "+ season-to-date\nprior", "+ split ball tracks\n(volleys)"]
mae = [51.9, 44.5, 43.9, 44.4, 41.9, 38.8]
cols = [GREY, GREY, GREY, GREY, GREY, ACC]
bars = ax.bar(range(len(steps)), mae, color=cols, width=0.62)
for b, v in zip(bars, mae):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=13, weight="bold",
            color=ACC if v == mae[-1] else INK)
ax.axhline(45.9, color=INK3, ls="--", lw=1)
ax.text(len(steps) - 0.45, 46.6, "two human scouts disagree by ~46", ha="right", fontsize=10, color=INK3)
ax.set_xticks(range(len(steps)), steps, fontsize=10)
ax.set_ylim(30, 56)
ax.set_ylabel("balls off per robot per match")
ax.set_title("Per-robot fuel error vs 5026's scouts (lower is better)", loc="left", fontsize=14, color=INK, pad=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# 2) headline numbers
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_title("Headline numbers", loc="left", fontsize=14, color=INK, pad=10)
cards = [("3 %", "alliance fuel totals vs\nofficial TBA counts (bar: 15 %)", "met"),
         ("96.6 %", "which robot is which\n(held-out bumper check)", "met"),
         ("18 %", "median per-robot fuel error\n(proposed target ~15 %)", "close"),
         ("76 %", "alliance top scorer\npicked correctly", "")]
for i, (big, small, tag) in enumerate(cards):
    y = 0.86 - i * 0.25
    ax.text(0.0, y, big, fontsize=30, weight="bold", color=ACC if tag == "met" else INK, transform=ax.transAxes)
    ax.text(0.47, y + 0.04, small, fontsize=10.5, color=INK2, transform=ax.transAxes, va="center")

# 3) ball tracking recall in 20+ ball volleys
ax = fig.add_subplot(gs[1, :2])
arms = ["old tracker\n30 fps", "split tracks\n30 fps (live now)", "ballistic + split\n30 fps", "split tracks\n60 fps*",
        "ballistic + split\n60 fps*"]
rec = [17, 77, 84, 91, 99]
cols = [GREY, ACC, GREY, RUN, RUN]
bars = ax.bar(range(len(arms)), rec, color=cols, width=0.62)
for b, v in zip(bars, rec):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} %", ha="center", fontsize=13, weight="bold")
ax.set_xticks(range(len(arms)), arms, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("scored balls turned into the right robot's arc")
ax.set_title("Ball tracking in 20+ ball volleys (higher is better)", loc="left", fontsize=14, color=INK, pad=10)
ax.text(len(arms) - 0.5, 108, "* 8-match test; 60 fps comparison vs scouts lands ~00:45", ha="right", fontsize=10,
        color=RUN)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# 4) status board
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_title("What's in, out, and running", loc="left", fontsize=14, color=INK, pad=10)
rows = [("IN", ACC, ["learned credit model + season prior", "DINO identity (live-safe)", "split tracks + traced gate",
                     "side-by-side tie-break", "viewer: live credit, all tracks, hub queue"]),
        ("OUT", INK3, ["ballistic tracker at 30 fps", "3-D arc fit, multi-camera triangulation",
                       "shooting-signal model, hub-entry credit", "Statbotics EPA prior, best-camera dedup"]),
        ("RUNNING", RUN, ["60 fps re-scan + scouting test (~00:45)", "neutral-zone launch filter test",
                          "stacked-robots linking rule (queued)"])]
y = 0.97
for tag, col, items in rows:
    ax.text(0.0, y, tag, fontsize=11, weight="bold", color=col, transform=ax.transAxes)
    y -= 0.075
    for it in items:
        ax.text(0.04, y, it, fontsize=10.5, color=INK2, transform=ax.transAxes)
        y -= 0.065
    y -= 0.03

fig.savefig(sys.argv[1], dpi=110, facecolor=BG)
print(sys.argv[1])
