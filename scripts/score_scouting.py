"""Score the per-team fuel split against 5026's human match scouting (data/scouting/, autoscout/scouting.py).

Scouts log every shot (not makes) per robot, so per team-match the reference is the alliance's official
TBA fuel total split in proportion to scouted shots (assumes the three robots shot equally accurately).
Two scouts per robot give the noise floor of that reference.

Reported, over the scouted matches that have results in --out:
* team-match fuel error (mean, median %) and correlation, auto-scout vs reference, beside two baselines
  that need no video: an even 1/3 split and a fuel-OPR split
* top scorer of the alliance agreement
* burst level: for each scoreboard jump, the share of balls auto-scout gave to the robots scouts logged
  shooting just before it (scout taps sit ~2 s before the counter; their clock does not pause at teleop)

  python scripts/score_scouting.py [--out output] [--write]   (--write: data/exp_scouting/RESULTS_section.md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoscout import config, opr, scouting, tba  # noqa: E402

# The scouting app's timer runs straight through the ~3 s auto->teleop pause, so a scouted time is
# video time - auto start. Scout taps land ~0.8 s after the launch and ~2.0 s before the counter
# moves (failure-finder cycle 1, data/exp_audit/tools/c1b_scout_lag.py).
LAG, HALF = -2.0, 2.5  # scouted shot time - counter time (s), and the window half-width


def to_mt_fn(tl):
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")
    return lambda vt: (20 + vt - ts) if ts and vt >= ts else vt - a0


def tri(dt):
    return np.clip(1 - np.abs(dt - LAG) / HALF, 0, None)


def score(out_root: Path, keep_rows: bool = False) -> dict:
    bm = scouting.by_match(scouting.load())
    rows, oprs, bursts, noise = [], {}, [], []
    has_vonly = False
    for key, teams in sorted(bm.items()):
        rp = out_root / key / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        info = tba.match(key)
        if info.event_key not in oprs:
            oprs[info.event_key] = opr.fuel_opr(info.event_key)
        to_mt = to_mt_fn(res["timeline"])
        a0 = res["timeline"].get("auto_start") or 6.13
        for al in ("red", "blue"):
            ts = info.teams(al)
            hub = info.hub(al) or {}
            total = hub.get("totalCount")
            if not total or not all(teams.get(t) for t in ts):
                continue  # need every robot of the alliance scouted
            n = np.array([np.mean([r.n for r in teams[t]]) for t in ts])
            if n.sum() <= 0:
                continue
            ref = total * n / n.sum()
            tm_ = [res["teams"][str(t)] if str(t) in res["teams"] else res["teams"][t] for t in ts]
            vid = np.array([x["fuel"] for x in tm_], float)
            vonly = np.array([x.get("fuel_video", x["fuel"]) for x in tm_], float)  # without the model's OPR prior
            has_vonly = has_vonly or bool(np.any(np.abs(vonly - vid) > 0.05))
            o = np.array([oprs[info.event_key].get(t, np.nan) for t in ts])
            oshare = o / o.sum() if np.all(np.isfinite(o)) and o.sum() > 0 else np.full(3, 1 / 3)
            for i, t in enumerate(ts):
                rows.append({"key": key, "team": t, "al": al, "ref": ref[i], "video": vid[i], "vonly": vonly[i],
                             "uniform": total / 3, "opr": total * oshare[i], "shots": n[i]})
            # noise floor: the reference built from scout A vs scout B where a robot has two
            if all(len(teams[t]) >= 2 for t in ts):
                na = np.array([teams[t][0].n for t in ts], float)
                nb = np.array([teams[t][1].n for t in ts], float)
                if na.sum() > 0 and nb.sum() > 0:
                    noise += list(np.abs(total * na / na.sum() - total * nb / nb.sum()))
            # burst level: counter jumps vs scouted shots around them
            log = [(mt, tm, b) for mt, tm, b in res.get("fuel_log") or [] if tm in ts]
            timed = {t: [r for r in teams[t] if r.time_valid] for t in ts}
            if not all(timed.values()):
                continue  # burst timing needs usable timestamps for all three robots
            prev = 0
            for vt, c in res["timeline"]["counts"].get(al, []):
                if c <= prev:
                    continue
                k = c - prev
                prev = c
                st_c = vt - a0  # counter time on the scouts' clock
                w = np.array([np.mean([tri(r.shots - st_c).sum() for r in timed[t]]) for t in ts])
                if w.sum() <= 0:
                    continue
                tshare = w / w.sum()
                mt_log = to_mt(vt - 2.0)  # fuel_log is stamped at counter time - 2 s
                got = np.array([sum(b for m, tm, b in log if tm == t and abs(m - mt_log) < 0.06) for t in ts])
                if got.sum() <= 0:
                    continue
                vshare = got / got.sum()
                bursts.append((k, float(np.minimum(vshare, tshare).sum()), float(np.minimum(1 / 3, tshare).sum()),
                               float(np.minimum(oshare, tshare).sum())))
    if not rows:
        return {}
    R = {m: np.array([r[m] for r in rows]) for m in ("ref", "video", "vonly", "uniform", "opr")}
    out = {"team_matches": len(rows), "matches": len({r["key"] for r in rows})}
    big = R["ref"] >= 10
    for m in ("video", "vonly", "uniform", "opr"):
        err = np.abs(R[m] - R["ref"])
        out[m] = {"mae": float(err.mean()), "median_rel": float(np.median(err[big] / R["ref"][big])),
                  "corr": float(np.corrcoef(R[m], R["ref"])[0, 1])}
    # top scorer per alliance
    tops = {"video": 0, "vonly": 0, "opr": 0}
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["key"], r["al"]), []).append(i)
    for idx in groups.values():
        j = idx[int(np.argmax(R["ref"][idx]))]
        for m in tops:
            tops[m] += idx[int(np.argmax(R[m][idx]))] == j
    out["top_scorer"] = {m: tops[m] / len(groups) for m in tops}
    out["alliances"] = len(groups)
    out["has_vonly"] = has_vonly
    out["noise_mae"] = float(np.mean(noise)) if noise else float("nan")
    if bursts:
        B = np.array(bursts)
        out["burst_overlap"] = {"video": float(np.average(B[:, 1], weights=B[:, 0])),
                                "uniform": float(np.average(B[:, 2], weights=B[:, 0])),
                                "opr": float(np.average(B[:, 3], weights=B[:, 0])), "balls": float(B[:, 0].sum())}
    if keep_rows:
        out["rows"] = rows
    return out


def section(s: dict, label: str) -> str:
    v, u, o, w = s["video"], s["uniform"], s["opr"], s["vonly"]
    vo = s.get("has_vonly")  # the learned credit model is on: also show the video-only split
    lines = [f"## Per-team fuel vs 5026's human scouting ({label})", "",
             f"{s['team_matches']} team-matches in {s['matches']} Chezy Champs quals scouted by FRC 5026 "
             "(every robot scouted, mostly twice). Reference = the alliance's TBA fuel total split by scouted "
             "shots, so it assumes equal accuracy within an alliance.", "",
             ("| | auto-scout | older hand-tuned rule | even 1/3 split | fuel-OPR split |" if vo else
              "| | auto-scout | even 1/3 split | fuel-OPR split |"),
             "| --- | ---: | ---: | ---: |" + (" ---: |" if vo else ""),
             f"| team-match fuel error, mean | **{v['mae']:.1f}** |" + (f" {w['mae']:.1f} |" if vo else "") + f" {u['mae']:.1f} | {o['mae']:.1f} |",
             f"| team-match fuel error, median % | **{100 * v['median_rel']:.0f} %** |" + (f" {100 * w['median_rel']:.0f} % |" if vo else "")
             + f" {100 * u['median_rel']:.0f} % | {100 * o['median_rel']:.0f} % |",
             f"| correlation with the reference | **{v['corr']:.2f}** |" + (f" {w['corr']:.2f} |" if vo else "") + f" – | {o['corr']:.2f} |",
             f"| top scorer of the alliance | **{100 * s['top_scorer']['video']:.0f} %** |" + (f" {100 * s['top_scorer']['vonly']:.0f} % |" if vo else "")
             + f" 33 % | {100 * s['top_scorer']['opr']:.0f} % |"]
    if "burst_overlap" in s:
        b = s["burst_overlap"]
        lines.append(f"| balls per scoreboard jump given to the robots scouts saw shooting | **{100 * b['video']:.0f} %** | "
                     + ("– | " if vo else "") + f"{100 * b['uniform']:.0f} % | {100 * b['opr']:.0f} % |")
    lines += ["", f"* noise floor: the reference built from one scout vs the other differs by {s['noise_mae']:.1f} fuel "
              "per team-match on average"]
    if vo:
        lines.append("* auto-scout uses the learned credit model, which was fitted to these same scouted matches, "
                     "so this column is in-sample; its cross-validated figures are in \"Learned per-team credit split\"")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.OUTPUT))
    ap.add_argument("--label", default="current results")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    s = score(Path(a.out))
    if not s:
        print("no scouted matches with results in", a.out)
        return
    txt = section(s, a.label)
    print(txt)
    if a.write:
        d = config.DATA / "exp_scouting"
        d.mkdir(parents=True, exist_ok=True)
        (d / "RESULTS_section.md").write_text(txt, encoding="utf-8")
        (d / "score.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
