"""Scoring bursts (one alliance's counter rising over <= 6 s) on the scouted Chezy quals, with the scouted vs the
credited split of each. Generalises data/exp_audit/tools/team_volleys.py to every team and to whole bursts.

Per counter increment (raw overlay counts): scouted weights = scout taps kernel-weighted around the jump
(score_scouting.tri, mean over the robot's scouts), credited = fuel_log rows of that jump. A burst groups
increments no more than GAP s apart (split if it would span more than MAX_SPAN s). Balls moved = K * total
variation distance between the scouted and the credited shares.

  .venv/Scripts/python data/exp_visual_qa/tools/find_bursts.py [--src output|data/exp_visual_qa/cache] [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import attribution as A, scouting, tba  # noqa: E402
from score_scouting import to_mt_fn, tri  # noqa: E402

GAP, MAX_SPAN = 1.2, 6.0


def vis_frac(det_t: dict, team: int, t0: float, t1: float) -> float:
    """Fraction of a 0.5 s grid over [t0, t1] where some camera has a box labelled `team` within 0.3 s."""
    ts = det_t.get(team)
    if ts is None or not len(ts):
        return 0.0
    g = np.arange(t0, t1 + 1e-6, 0.5)
    j = np.clip(np.searchsorted(ts, g), 1, len(ts) - 1)
    d = np.minimum(np.abs(ts[j] - g), np.abs(ts[j - 1] - g))
    return float((d <= 0.3).mean())


def bursts_for(key: str, r: dict, bm: dict, dets: list | None = None) -> list[dict]:
    info = tba.match(key)
    det_t = {}
    if dets is not None:
        tmp = {}
        for row in dets:
            if row[7] is not None:
                tmp.setdefault(row[7], []).append(row[0])
        det_t = {k: np.array(sorted(v)) for k, v in tmp.items()}
    tl = r["timeline"]
    to_mt = to_mt_fn(tl)
    a0 = tl.get("auto_start") or 6.13
    log = r["fuel_log"]
    out = []
    for al in ("red", "blue"):
        ts = info.teams(al)
        timed = {t: [x for x in bm[key].get(t, []) if x.time_valid] for t in ts}
        if not all(timed.values()):
            continue
        shots = [s for s in r["shots"] if s["alliance"] == al]
        incs = []
        prev = 0
        for vt, c in tl["counts"].get(al, []):
            if c <= prev:
                continue
            k, prev = c - prev, c
            st_c = vt - a0
            w = np.array([np.mean([tri(x.shots - st_c).sum() for x in timed[t]]) for t in ts])
            mt_log = to_mt(vt - 2.0)
            got = np.array([sum(b for m, tm, b in log if tm == t and abs(m - mt_log) < 0.06) for t in ts])
            incs.append((vt, k, w, got))
        # group increments into bursts
        groups, cur = [], []
        for inc in incs:
            if cur and (inc[0] - cur[-1][0] > GAP or inc[0] - cur[0][0] > MAX_SPAN):
                groups.append(cur)
                cur = []
            cur.append(inc)
        if cur:
            groups.append(cur)
        for g in groups:
            K = sum(x[1] for x in g)
            ss = np.zeros(3)
            k_scouted = 0
            for vt, k, w, got in g:
                if w.sum() > 0:
                    ss += k * w / w.sum()
                    k_scouted += k
            cs = sum(x[3] for x in g)
            if k_scouted == 0 or cs.sum() <= 0:
                sshare = ss / max(ss.sum(), 1e-9)
                cshare = cs / max(cs.sum(), 1e-9)
                tv = None
            else:
                sshare = ss / ss.sum()
                cshare = cs / cs.sum()
                tv = 0.5 * float(np.abs(sshare - cshare).sum())
            t0, t1 = g[0][0], g[-1][0]
            ev = {}
            nl = {}
            for t in ts:
                ev[t] = round(float(sum(s["conf"] * A.kernel(np.array([vt - s["t"]]))[0] * k
                                        for vt, k, _, _ in g for s in shots if s["team"] == t) / K), 2)
                nl[t] = sum(1 for s in shots if s["team"] == t and t0 - A.LAG_HI <= s["t"] <= t1 - A.LAG_LO)
            n_unk = sum(1 for s in shots if s["team"] is None and t0 - A.LAG_HI <= s["t"] <= t1 - A.LAG_LO)
            vis = {t: round(vis_frac(det_t, t, t0 - A.LAG_HI, t1 - A.LAG_LO), 2) for t in ts} if det_t else {}
            out.append({"match": key, "alliance": al, "teams": ts, "t0": round(t0, 2), "t1": round(t1, 2),
                        "mt0": round(to_mt(t0 - 2.0), 1), "K": int(K), "k_scouted": int(k_scouted),
                        "sshare": [round(x, 3) for x in sshare], "cshare": [round(x, 3) for x in cshare],
                        "tv": None if tv is None else round(tv, 3),
                        "moved": None if tv is None else round(K * tv, 1),
                        "ev": ev, "n_launch": nl, "n_launch_unlabelled": n_unk, "vis": vis,
                        "scout_top": ts[int(np.argmax(sshare))], "credit_top": ts[int(np.argmax(cshare))]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output"))
    ap.add_argument("--out", default=str(ROOT / "data/exp_visual_qa/bursts_output.json"))
    ap.add_argument("--matches", nargs="*", default=None)
    a = ap.parse_args()
    bm = scouting.by_match(scouting.load())
    keys = a.matches or sorted(k for k in bm if k.startswith("2026cc_qm"))
    allb = []
    for key in keys:
        rp = Path(a.src) / key / "result.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
            dp = Path(a.src) / key / "dets.json"
            dets = json.loads(dp.read_text()) if dp.exists() else None
        except Exception as e:  # being rewritten
            print(key, "unreadable", e)
            continue
        b = bursts_for(key, r, bm, dets)
        allb += b
    Path(a.out).write_text(json.dumps(allb, indent=0))
    ok = [b for b in allb if b["tv"] is not None]
    print(len(allb), "bursts,", len(ok), "with scouting+credit;", sum(b["K"] for b in ok), "balls;",
          f"{sum(b['moved'] for b in ok):.0f} balls moved")
    big = sorted([b for b in ok if b["K"] >= 5 and max(b["sshare"]) >= 0.55 and b["scout_top"] != b["credit_top"]],
                 key=lambda b: -b["moved"])
    print(len(big), "bursts K>=5 with a clear scouted shooter that credit gives mostly to someone else")
    from collections import Counter
    print(Counter(b["match"] for b in big[:60]).most_common(20))
    for b in big[:40]:
        print(b["match"], b["alliance"], b["t0"], b["t1"], b["K"], b["teams"], b["sshare"], b["cshare"], b["moved"],
              b["n_launch"], b["n_launch_unlabelled"], b["vis"])
