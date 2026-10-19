"""Evaluate processed matches against TBA.

* alliance level: overlay-timeline total, vision-only estimate (if present) vs TBA totalCount
* team level: correlation of attributed fuel with fuel-OPR shares; team 5026 summary

  python scripts/evaluate.py                       # output/ -> output/evaluation.json + RESULTS.md
  python scripts/evaluate.py --out data/exp_robots_v2/v2 [--matches k1 k2 ...]
      # an experiment dir and/or a match subset: prints, writes <out>/evaluation.json, never RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, opr, tba  # noqa: E402


def main(out: Path | str | None = None, matches: set[str] | None = None) -> dict:
    root = Path(out) if out else config.OUTPUT
    rows = []
    team_rows = []
    oprs = {}
    for f in sorted(root.glob("*/result.json")):
        if matches is not None and f.parent.name not in matches:
            continue
        r = json.loads(f.read_text())
        key = r["match"]
        info = tba.match(key)
        ev = info.event_key
        if ev not in oprs:
            try:
                oprs[ev] = opr.fuel_opr(ev)
            except Exception:
                oprs[ev] = {}
        for al in ("red", "blue"):
            meta = r["teams"][f"_{al}"]
            tba_total = info.hub(al).get("totalCount")
            vis = r.get("vision_only", {}).get(al)
            rows.append({"match": key, "alliance": al, "tba": tba_total, "timeline": meta.get("timeline_total"),
                         "source": meta.get("source"), "vision": vis})
        team_rows += team_rows_of(r, info, oprs[ev])
    print(f"{'match':22s} {'al':4s} {'TBA':>5s} {'overlay':>8s} {'err%':>6s} {'vision':>7s} {'err%':>6s}")
    errs_o, errs_v = [], []
    for x in rows:
        eo = ev_ = None
        if x["tba"] and x["timeline"] is not None and x["source"] == "overlay":
            eo = 100 * (x["timeline"] - x["tba"]) / max(1, x["tba"])
            errs_o.append(abs(eo))
        if x["tba"] and x["vision"] is not None:
            ev_ = 100 * (x["vision"] - x["tba"]) / max(1, x["tba"])
            errs_v.append(abs(ev_))
        vis_s = "" if x["vision"] is None else f"{x['vision']:7.0f}"
        eo_s = "" if eo is None else f"{eo:6.1f}"
        ev_s = "" if ev_ is None else f"{ev_:6.1f}"
        print(f"{x['match']:22s} {x['alliance']:4s} {x['tba'] or 0:5d} {x['timeline'] or 0:8.0f} "
              f"{eo_s:>6s} {vis_s:>7s} {ev_s:>6s}")
    if errs_o:
        print(f"\noverlay timeline: n={len(errs_o)} mean |err| {np.mean(errs_o):.1f}%  median {np.median(errs_o):.1f}%  "
              f"within 15%: {np.mean(np.array(errs_o) <= 15) * 100:.0f}%")
    if errs_v:
        print(f"vision-only:      n={len(errs_v)} mean |err| {np.mean(errs_v):.1f}%  median {np.median(errs_v):.1f}%  "
              f"within 15%: {np.mean(np.array(errs_v) <= 15) * 100:.0f}%")
    errs_o = errs_o if errs_o else []
    shares_a, shares_o, abs_err, base_err = alliance_shares(team_rows)
    if len(shares_a) >= 6:
        print(f"\nteam level ({len(shares_a)} team-matches): corr(video share, fuel-OPR share) = "
              f"{np.corrcoef(shares_a, shares_o)[0, 1]:.2f};  mean |video - OPR-share| = {np.mean(abs_err):.1f} fuel "
              f"(uniform-split baseline vs OPR-share: {np.mean(base_err):.1f})")
    mine = [x for x in team_rows if x["team"] == 5026]
    if mine:
        print("\nteam 5026:")
        for x in mine:
            print(f"  {x['match']:22s} fuel {x['fuel']:6.1f}   (OPR {x['opr']:.0f})")
    summary = {"overlay_mean_abs_err_pct": float(np.mean(errs_o)) if errs_o else None, **team_metrics(team_rows)}
    print(f"team metrics: {summary}")
    if out is not None or matches is None:
        Path(root / "evaluation.json").write_text(json.dumps({"alliances": rows, "teams": team_rows,
                                                              "summary": summary}, default=float, indent=1))
    if out is None and matches is None:
        write_results_md(rows, team_rows, errs_o, shares_a, shares_o, abs_err, base_err)
    return summary


def alliance_shares(team_rows: list[dict]):
    """Team level: within-alliance shares vs fuel-OPR shares (alliances with >= 40 fuel).
    -> (video shares, OPR shares, |video - OPR-share| in fuel, same for a uniform 1/3 split),
    three consecutive entries per alliance."""
    shares_a, shares_o, abs_err, base_err = [], [], [], []
    by_al = {}
    for x in team_rows:
        by_al.setdefault((x["match"], x["alliance"]), []).append(x)
    for (_, _), xs in by_al.items():
        fa = np.array([x["fuel"] for x in xs])
        op = np.array([x["opr"] for x in xs])
        if fa.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
            continue
        sa, so = fa / fa.sum(), op / op.sum()
        shares_a += sa.tolist()
        shares_o += so.tolist()
        abs_err += (np.abs(sa - so) * fa.sum()).tolist()
        base_err += (np.abs(1 / 3 - so) * fa.sum()).tolist()
    return shares_a, shares_o, abs_err, base_err


def team_metrics(team_rows: list[dict]) -> dict:
    """The RESULTS.md team-level numbers for a set of team rows."""
    sa, so, ae, be = alliance_shares(team_rows)
    n_al = len(sa) // 3
    top = sum(1 for k in range(0, len(sa), 3) if int(np.argmax(sa[k:k + 3])) == int(np.argmax(so[k:k + 3])))
    return {"team_matches": len(sa), "alliances": n_al,
            "corr": float(np.corrcoef(sa, so)[0, 1]) if len(sa) >= 6 else float("nan"),
            "top": top, "top_pct": 100.0 * top / max(1, n_al),
            "mae": float(np.mean(ae)) if ae else float("nan"),
            "uniform_mae": float(np.mean(be)) if be else float("nan")}


def team_rows_of(r: dict, info, opr_ev: dict) -> list[dict]:
    """Per-team rows (per-match fuel, the older hand-tuned rule's fuel, fuel OPR) of one result.json."""
    out = []
    for al in ("red", "blue"):
        tba_total = info.hub(al).get("totalCount")
        teams = info.teams(al)
        o = np.array([opr_ev.get(t, np.nan) for t in teams])
        rt = [r["teams"][str(t)] for t in teams]
        for t, oo, x in zip(teams, o, rt):
            share_o = oo / np.nansum(o) if np.nansum(o) > 0 else np.nan
            out.append({"match": r["match"], "team": t, "alliance": al, "fuel": x["fuel"],
                        "fuel_rule": x.get("fuel_video", x["fuel"]), "opr": oo,
                        "opr_share_pred": share_o * (tba_total or 0)})
    return out


def fuel_det_ab_section() -> list[str]:
    """RESULTS.md lines for the flying-fuel detector A/B (scripts/fuel_det_experiment.py eval --publish)."""
    p = config.DATA / "exp_fueldet" / "ab_published.json"
    if not p.exists():
        return []
    from autoscout import pipeline

    ab = json.loads(p.read_text())
    names = {"cached": "motion blobs only (baseline)", "motion": None,
             "zone_only": "no launch evidence (zone occupancy only; reference)"}
    lines = ["## Flying-fuel detector A/B (held-out matches)", "",
             "Launch arcs come from moving-yellow blobs (three-frame differencing). A flying-fuel YOLO "
             "(Roboflow airborne-fuel set + pseudo-labels from the shot tracks of the other matches) was "
             "tried as an extra blob source, merged with the motion blobs before flight tracking. Same "
             f"{ab['matches']} held-out matches ({', '.join(ab.get('events', []))}; never used for training), "
             "same code, only the blob source differs:", "",
             "| variant | share corr | top scorer | mean abs err | launches / scored ball | Δcorr 90 % CI | P(corr better) |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for k, m in ab["variants"].items():
        label = names.get(k, k.replace("union@", "motion + detector, conf ≥ ").replace("det_only@", "detector only, conf ≥ ")
                                .replace("confirm@", "motion tracks confirmed by detector, conf ≥ "))
        if label is None:
            continue
        b = m.get("vs_cached") or {}
        ci = f"[{b['d_corr_ci90'][0]:+.3f}, {b['d_corr_ci90'][1]:+.3f}]" if b else ""
        pb = f"{b['p_corr_better']:.2f}" if b else ""
        lines.append(f"| {label} | {m['corr']:.3f} | {m['top']}/{m['alliances']} ({m['top_pct']:.0f} %) | "
                     f"{m['mae']:.1f} | {m['shots_per_ball']:.2f} | {ci} | {pb} |")
    o = pipeline.Options()
    lines += ["", f"* pipeline default: " + (f"detector on (`{Path(o.flying_fuel_model).name}`, conf ≥ "
                                            f"{o.flying_fuel_conf})" if o.flying_fuel_model else
                                            "detector off (`Options.flying_fuel_model = None`); "
                                            "see the table for why"), ""]
    if ab.get("note"):
        lines += [f"* {ab['note']}", ""]
    return lines


def write_results_md(rows, team_rows, errs_o, shares_a, shares_o, abs_err, base_err):
    """Markdown summary of the latest evaluation (RESULTS.md in the repo root)."""
    keys = {r["match"] for r in rows}
    n5026 = sum(1 for k in keys if 5026 in tba.match(k).teams("red") + tba.match(k).teams("blue"))
    lines = ["# auto-scout results", "",
             f"Matches processed: {len(keys)} ({len(rows)} alliance-matches): every 2026 match of team 5026 "
             f"with a downloadable video ({n5026}; Silicon Valley, Glendale, NorCal State Champs, Johnson "
             f"division, Chezy Champs) plus {len(keys) - n5026} other Chezy Champs qualification matches that "
             "FRC 5026 scouted (human per-team reference). Generated by `scripts/evaluate.py`.", "",
             "**Summary.** Alliance FUEL totals read from the broadcast's live counter are within a few "
             "percent of TBA's official hub counts in every match (the acceptance bar was 15 %). Per-team "
             "numbers split that exact total by which robots were seen launching fuel while in their "
             "alliance zone, with weights from a small credit model fitted once to 5026's scouting. Each "
             "match's split uses only that match's video: no OPR or other team history moves balls between "
             "teams. Nothing uses an event's later matches either: the DINO robot recognizer only learns from "
             "matches already played, as it would live. A robot or launch no camera can identify goes, by "
             "elimination, to the one robot of its alliance not seen anywhere else. Against the scouting the per-team "
             "numbers are off by ~41 fuel per team-match held out (the older hand-tuned rule on the same evidence: "
             "~47); arcs credited to a teammate standing next to the shooter, and shooters no camera sees, are the "
             "main error sources. "
             "A pure vision hub counter "
             "(no overlay) was built and evaluated but is ~30 % off, so it is only a fallback.", ""]
    if errs_o:
        e = np.array(errs_o)
        lines += ["## Alliance FUEL totals: overlay timeline vs TBA", "",
                  f"* alliance-matches: {len(e)}",
                  f"* mean |error| {e.mean():.1f} %, median {np.median(e):.1f} %, "
                  f"within 15 %: {np.mean(e <= 15) * 100:.0f} %", ""]
    vpath = config.OUTPUT / "vision_counter_eval.json"
    if vpath.exists():
        ve = json.loads(vpath.read_text())
        ae = np.abs([x[-1] for x in ve if x[2] >= 20])
        if len(ae):
            lines += ["## Alliance FUEL totals: vision-only counter vs TBA (leave-one-match-out)", "",
                      f"* alliance-matches with >= 20 fuel: {len(ae)}",
                      f"* mean |error| {ae.mean():.1f} %, median {np.median(ae):.1f} %, "
                      f"within 15 %: {np.mean(ae <= 15) * 100:.0f} %", ""]
    ip = config.OUTPUT / "identity_holdout.json"
    if ip.exists():
        ih = json.loads(ip.read_text())
        lines += ["## Robot identity (bumper OCR held out)", "",
                  f"* identities assigned from half of the bumper reads agree with the other half for "
                  f"{100 * ih['agreement']:.1f} % of that evidence ({ih['matches']} matches). OCR reads "
                  f"themselves are ~98 % self-consistent, so this is close to the identity accuracy.", ""]
    if len(shares_a) >= 6:
        top = sum(1 for k in range(0, len(shares_a), 3)
                  if int(np.argmax(shares_a[k:k + 3])) == int(np.argmax(shares_o[k:k + 3])))
        lines += ["## Team-level attribution vs fuel OPR (independent per-team reference)", "",
                  "Fuel OPR (least squares over every qualification match at the event, from TBA) is the only "
                  "public per-team reference. It is a season-average, so it cannot be matched exactly in any "
                  "single match (defense, breakdowns, roles change match to match). The per-team numbers do "
                  "not use OPR, so this is an independent check.", "",
                  f"* team-matches: {len(shares_a)}",
                  f"* correlation of within-alliance share (video vs OPR): {np.corrcoef(shares_a, shares_o)[0, 1]:.2f}",
                  f"* top scorer of the alliance (video) = top fuel-OPR team: {top}/{len(shares_a) // 3} alliances "
                  f"({100 * top / max(1, len(shares_a) // 3):.0f} %; 33 % by chance)",
                  f"* mean |video - OPR-share| = {np.mean(abs_err):.1f} fuel per team-match "
                  f"(a uniform 1/3 split is {np.mean(base_err):.1f} off the OPR shares)", ""]
    # model / method experiments write their own section (kept across regenerations of this file)
    for p in sorted(config.DATA.glob("exp_*/RESULTS_section.md")):
        lines += [p.read_text(encoding="utf-8").strip(), ""]
    lines += fuel_det_ab_section()
    mine = [x for x in team_rows if x["team"] == 5026]
    if mine:
        lines += ["## Team 5026", "", "Fuel from each match's video alone; the event's fuel OPR is shown for reference.", "",
                  "| match | fuel | fuel OPR at event |", "| --- | ---: | ---: |"]
        for x in mine:
            lines.append(f"| {x['match']} | {x['fuel']:.0f} | {x['opr']:.0f} |")
        lines.append("")
    (config.ROOT / "RESULTS.md").write_text(chr(10).join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory of <match>/result.json (default output/)")
    ap.add_argument("--matches", nargs="+", default=None, help="only these match keys")
    a = ap.parse_args()
    main(out=a.out, matches=set(a.matches) if a.matches else None)
