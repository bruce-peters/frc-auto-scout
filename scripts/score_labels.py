"""Score the per-team fuel split against hand shooter labels (data/labels/).

For every labelled scoring burst, the pipeline's allocation of that burst's balls (from
result.json `fuel_log`) is compared with the label (balls split evenly among the robots the
labeller saw launching; "other" = no robot). Reported:

* attribution accuracy: share of labelled balls the pipeline gave to a robot that really shot
* per-team error over the labelled bursts (team-match totals), vs two baselines that need no
  video: an even 1/3 split and a fuel-OPR-share split
* agreement between labellers who labelled the same bursts

Unsure / skipped bursts are left out. Writes data/exp_labels/RESULTS_section.md (appended to
RESULTS.md by scripts/evaluate.py) and data/exp_labels/score.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoscout import config, labels, opr, tba  # noqa: E402

OUT = config.DATA / "exp_labels"


def to_mt_fn(tl: dict):
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")
    return lambda vt: (20 + vt - ts) if ts and vt >= ts else vt - a0


def pipeline_alloc(res: dict, teams: list[int], t0: float, t1: float) -> dict[int, float]:
    """Balls the pipeline credited to each team for counter increments in [t0, t1] (video s)."""
    f = to_mt_fn(res["timeline"])
    lo, hi = f(t0 - 2.0) - 0.06, f(t1 - 2.0) + 0.06  # fuel_log is stamped at counter time - 2 s
    out = {t: 0.0 for t in teams}
    for mt, team, n in res.get("fuel_log") or []:
        if team in out and lo <= mt <= hi:
            out[team] += n
    return out


def pct(x: float) -> str:
    return "–" if not np.isfinite(x) else f"{100 * x:.0f} %"


def main():
    files = labels.all_label_files()
    if not files:
        print("no labels yet (data/labels/)")
        return
    oprs: dict[str, dict] = {}
    rows = []  # one per (labeller, burst)
    by_burst = defaultdict(dict)  # (match, alliance, round(t0,1)) -> {labeller: frozenset}
    for p in files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        key, who = doc["match"], doc["labeller"]
        rp = config.OUTPUT / key / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        info = tba.match(key)
        if info.event_key not in oprs:
            oprs[info.event_key] = opr.fuel_opr(info.event_key)
        for bid, lab in doc["labels"].items():
            if lab["status"] not in ("labeled", "other"):
                continue
            al, teams = lab["alliance"], info.teams(lab["alliance"])
            shooters = [t for t in lab["shooters"] if t in teams] if lab["status"] == "labeled" else []
            if lab["status"] == "labeled" and not shooters:
                continue
            balls = float(lab["balls"])
            truth = {t: (balls / len(shooters) if t in shooters else 0.0) for t in teams}
            alloc = pipeline_alloc(res, teams, lab["t0"], lab["t1"])
            got = sum(alloc.values())
            if got > 0:  # same units as the label (the overlay counter): rescale tiny rounding drift
                alloc = {t: v * balls / got for t, v in alloc.items()}
            o = np.array([oprs[info.event_key].get(t, np.nan) for t in teams])
            o_share = o / o.sum() if np.all(np.isfinite(o)) and o.sum() > 0 else np.full(3, 1 / 3)
            rows.append({"match": key, "labeller": who, "alliance": al, "burst": bid, "balls": balls,
                         "teams": teams, "truth": truth, "video": alloc,
                         "uniform": {t: balls / 3 for t in teams},
                         "opr": {t: balls * s for t, s in zip(teams, o_share)},
                         "correct_video": sum(alloc[t] for t in shooters),
                         "correct_uniform": balls * len(shooters) / 3,
                         "correct_opr": sum(balls * s for t, s in zip(teams, o_share) if t in shooters)})
            by_burst[(key, al, round(lab["t0"], 1))][who] = frozenset(shooters) if shooters else frozenset({"other"})
    if not rows:
        print("no usable labels (only unsure/skip so far)")
        return
    tot = sum(r["balls"] for r in rows)
    acc = {m: sum(r[f"correct_{m}"] for r in rows) / tot for m in ("video", "uniform", "opr")}
    # team-match totals over the labelled bursts (one labeller per match: the one with most labels)
    best_lab = {}
    for r in rows:
        best_lab.setdefault(r["match"], defaultdict(int))[r["labeller"]] += 1
    team_tot = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["labeller"] != max(best_lab[r["match"]], key=best_lab[r["match"]].get):
            continue
        for t in r["teams"]:
            for m in ("truth", "video", "uniform", "opr"):
                team_tot[(r["match"], t)][m] += r[m][t]
    tt = list(team_tot.values())
    err = {m: float(np.mean([abs(x[m] - x["truth"]) for x in tt])) for m in ("video", "uniform", "opr")}
    big = [x for x in tt if x["truth"] >= 10]
    rel = {m: float(np.median([abs(x[m] - x["truth"]) / x["truth"] for x in big])) if big else float("nan")
           for m in ("video", "uniform", "opr")}
    corr = float(np.corrcoef([x["video"] for x in tt], [x["truth"] for x in tt])[0, 1]) if len(tt) > 2 else float("nan")
    pairs = [(a, b) for v in by_burst.values() if len(v) >= 2 for a, b in [tuple(v.values())[:2]]]
    agree = float(np.mean([a == b for a, b in pairs])) if pairs else float("nan")
    n_matches = len({r["match"] for r in rows})
    lines = ["## Per-team attribution vs hand shooter labels", "",
             f"{len(rows)} labelled scoring bursts ({tot:.0f} balls) in {n_matches} matches, "
             f"labelled in the match inspector (`python -m autoscout.inspector`, Label shooters).", "",
             "| | video (auto-scout) | even 1/3 split | fuel-OPR split |", "| --- | ---: | ---: | ---: |",
             f"| balls credited to a robot that really shot | **{100 * acc['video']:.0f} %** | {100 * acc['uniform']:.0f} % | {100 * acc['opr']:.0f} % |",
             f"| team-match fuel error, mean (labelled bursts only) | **{err['video']:.1f}** | {err['uniform']:.1f} | {err['opr']:.1f} |",
             f"| team-match fuel error, median % (teams with >= 10) | **{pct(rel['video'])}** | {pct(rel['uniform'])} | {pct(rel['opr'])} |",
             "", f"* correlation of video vs labelled team-match fuel: {corr:.2f} ({len(tt)} team-matches)"]
    if pairs:
        lines.append(f"* labellers agree on the shooter set for {100 * agree:.0f} % of {len(pairs)} bursts labelled twice")
    lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS_section.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "score.json").write_text(json.dumps({"accuracy": acc, "team_mae": err, "team_median_rel": rel,
                                                "corr": corr, "bursts": len(rows), "balls": tot,
                                                "matches": n_matches, "labeller_agreement": agree},
                                               indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
