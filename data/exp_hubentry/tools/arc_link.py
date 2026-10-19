"""Entries -> arcs -> robots: which entry tracks (balls dropping into a hub opening) can be tied to a launch arc that
the pipeline already linked to a robot (result.json shots), and what share of counted balls that covers.

Per view, an entry track is linked
  direct  when it IS a linked launch arc (same view, same first point), or
  chained when a linked launch arc in the same view ended shortly before it started and one projectile
          (u linear, v quadratic in t) fits both fragments: RMS <= FIT_PX px, downward image acceleration, gap
          <= MAX_GAP s, and the joint fit's speed matches the launch fragment's.
Coverage = linked entries (deduplicated across views, ENTRY_DEDUP s) / counted balls. Consistency = share of
linked entries whose robot is of the hub's alliance (a ball scored in a hub comes from that alliance).

  .venv/Scripts/python data/exp_hubentry/tools/arc_link.py -> data/exp_hubentry/arc_link.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lag_study import OUT, DST, tl_window  # noqa: E402

from autoscout import hubentry as H, scouting  # noqa: E402

FIT_PX = 4.0
MAX_GAP = 1.2
ENTRY_DEDUP = 0.25


def joint_fit(P):
    t = P[:, 0] - P[0, 0]
    A1 = np.vstack([t, np.ones_like(t)]).T
    A2 = np.vstack([t * t, t, np.ones_like(t)]).T
    bu = np.linalg.lstsq(A1, P[:, 1], rcond=None)[0]
    bv = np.linalg.lstsq(A2, P[:, 2], rcond=None)[0]
    res = np.sqrt(np.mean((A1 @ bu - P[:, 1]) ** 2 + (A2 @ bv - P[:, 2]) ** 2))
    return bu, bv, float(res)


def main():
    scouted = set(scouting.by_match(scouting.load()))
    tot = {}
    per_match = []
    for cp in sorted((DST / "cache").glob("*.pkl")):
        key = cp.stem
        r = json.loads((OUT / key / "result.json").read_text())
        tl = r["timeline"]
        lay = tl.get("layout") or "?"
        t0, t1 = tl_window(tl)
        c = pickle.loads(cp.read_bytes())
        shots = {}
        for s in r["shots"]:
            p = np.array(s["pts"], dtype=float)
            shots.setdefault(s["view"], []).append((p[0, 0], p[-1, 0], p, s))
        row = {"key": key, "layout": lay, "scouted": key in scouted}
        for al in ("red", "blue"):
            balls, prev = 0, 0
            for t, n in tl["counts"].get(al) or []:
                if n > prev and t0 <= t <= t1:
                    balls += n - prev
                prev = max(prev, n)
            if balls < 20:
                continue
            links = []  # (t_entry, view, kind, team, shot alliance)
            n_ent = 0
            for v, T in c["views"].items():
                if v not in ("main", f"{al}_end"):
                    continue
                e = np.flatnonzero(H.is_entry(T, al) & (T[:, H.C["t1"]] >= t0) & (T[:, H.C["t1"]] <= t1))
                n_ent += len(e)
                sv = shots.get(v, [])
                st0 = np.array([x[0] for x in sv]) if sv else np.zeros(0)
                st1 = np.array([x[1] for x in sv]) if sv else np.zeros(0)
                for i in e:
                    P = c["near_pts"][v].get(int(i))
                    if P is None:
                        continue
                    P = np.asarray(P, dtype=float)
                    te = float(P[-1, 0])
                    # direct: the entry track is itself a linked launch arc
                    j = np.flatnonzero(np.abs(st0 - P[0, 0]) < 0.002) if len(st0) else []
                    hit = None
                    for jj in j:
                        if abs(sv[jj][2][0, 1] - P[0, 1]) < 1.0:
                            hit = ("direct", sv[jj][3])
                            break
                    if hit is None and len(st1):
                        cand = np.flatnonzero((st1 < P[0, 0]) & (P[0, 0] - st1 <= MAX_GAP) & (st0 > te - 3.0))
                        best = None
                        for jj in cand:
                            Q = sv[jj][2]
                            J = np.vstack([Q, P[:, :3]])
                            bu, bv, res = joint_fit(J)
                            if res > FIT_PX or bv[0] <= 0:
                                continue
                            # launch fragment's own speed vs the joint fit's at its start
                            q = Q[-min(4, len(Q)):]
                            vq = np.hypot(*(q[-1, 1:3] - q[0, 1:3])) / max(q[-1, 0] - q[0, 0], 1e-3)
                            vj = np.hypot(bu[0], bv[1] + 2 * bv[0] * (q[0, 0] - J[0, 0]))
                            if not (0.6 < vq / max(vj, 1e-6) < 1.6):
                                continue
                            if best is None or res < best[0]:
                                best = (res, sv[jj][3])
                        if best is not None:
                            hit = ("chained", best[1])
                    if hit is not None:
                        links.append((te, v, hit[0], hit[1]["team"], hit[1]["alliance"], hit[1]["t"]))
            # dedupe across views: the same ball seen entering in two views
            links.sort(key=lambda x: x[0])
            uniq = []
            for x in links:
                if uniq and abs(x[0] - uniq[-1][0]) < ENTRY_DEDUP and x[3] == uniq[-1][3]:
                    continue
                uniq.append(x)
            same = sum(1 for x in uniq if x[4] == al)
            flight = [x[0] - x[5] for x in uniq]
            row[al] = {"balls": balls, "entries": n_ent, "linked": len(uniq),
                       "direct": sum(1 for x in uniq if x[2] == "direct"),
                       "chained": sum(1 for x in uniq if x[2] == "chained"), "same_alliance": same,
                       "flight_med": float(np.median(flight)) if flight else None}
            for kk in ("balls", "entries", "linked", "direct", "chained", "same_alliance"):
                tot.setdefault(lay, {}).setdefault(kk, 0)
                tot[lay][kk] += row[al][kk]
            tot.setdefault(lay, {}).setdefault("flights", []).extend(flight)
        per_match.append(row)
        print(key, {al: row.get(al, {}).get("linked") for al in ("red", "blue")}, flush=True)
    summ = {}
    for lay, d in tot.items():
        f = np.array(d.pop("flights"))
        summ[lay] = {**d, "linked_per_ball": d["linked"] / d["balls"], "entries_per_ball": d["entries"] / d["balls"],
                     "linked_per_entry": d["linked"] / max(d["entries"], 1),
                     "same_alliance_share": d["same_alliance"] / max(d["linked"], 1),
                     "flight_q10_50_90": np.percentile(f, [10, 50, 90]).round(2).tolist() if len(f) else None}
        print(lay, json.dumps(summ[lay]))
    (DST / "arc_link.json").write_text(json.dumps({"summary": summ, "rows": per_match}, indent=1))


if __name__ == "__main__":
    main()
