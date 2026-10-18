"""Score per-robot launch signals (data/exp_launch/sig/*.pkl, from lsig.py) against 5026's scout taps.

  leval.py [--map cont|tl] [--win 4] [--sigs a,b,...]

1. Mapping check: alliance scout-tap series vs scoreboard increments, cross-correlation lag (auto / teleop).
2. Window level (alliances whose 3 robots all have time-valid scouting): W-second windows; windows with
   >= 3 scouted balls for the alliance. Per signal (event times shifted by the lag L that minimises the
   misassigned share, grid -2..2 s):
     cover   share of scouted balls in windows where the signal has anything for the alliance
     corr    pooled Pearson of within-alliance shares (windows where the signal is > 0)
     top     the most-scouted robot is the signal's top robot (same windows, unique scouted top)
     mis     balls misassigned: 0.5 * sum_r |share_r - scout share_r|, ball-weighted over ALL windows
             (an even 1/3 split where the signal is empty)
     rho     Spearman over all robot-windows (incl. zero windows): does the signal track the rate
3. Match level (alliances with all 3 scouted): signal totals -> shares vs scouted shares: TV, corr, top,
   fuel MAE (TBA total x share).
"""
import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
SIG = ROOT / "data" / "exp_launch" / "sig"
SIGS = ["cred", "rise", "be", "drain", "drainraw", "hub", "hubnear", "still"]


def rank(a):
    from scipy.stats import rankdata
    return rankdata(a)  # average ranks for ties


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load(extra=None):
    out = []
    for p in sorted(glob.glob(str(SIG / "*.pkl"))):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        if extra:
            for t, s in extra(d).items():
                d["sig"].setdefault(t, {}).update(s)
        out.append(d)
    return out


def binned(ev, edges, L=0.0):
    if not ev:
        return np.zeros(len(edges) - 1)
    a = np.asarray(ev, float)
    return np.histogram(a[:, 0] + L, bins=edges, weights=a[:, 1])[0]


def mapping_check(ms):
    for mp in ("cont", "tl"):
        for per in ("auto", "teleop"):
            acc = np.zeros(33)
            lags = np.arange(-2, 6.25, 0.25)
            for d in ms:
                for al, A_ in d["alliances"].items():
                    if not all(d["timed"].get(t) for t in A_["teams"]) or not A_["inc"]:
                        continue
                    lo, hi = (d["auto_start"], d["auto_start"] + 20) if per == "auto" else \
                        ((d["teleop_start"] or d["auto_start"] + 23) + 2, d["t1"] - 5)
                    g = np.arange(lo, hi, 0.25)
                    tap = sum(binned(d["truth"][t][mp], np.r_[g, g[-1] + 0.25]) for t in A_["teams"])
                    for li, L in enumerate(lags):
                        cnt = binned(A_["inc"], np.r_[g, g[-1] + 0.25], -L)  # counter at t + L
                        if tap.std() > 0 and cnt.std() > 0:
                            acc[li] += float(np.corrcoef(tap, cnt)[0, 1])
            print(f"  map {mp:4s} {per:6s}: counter lags taps by {lags[acc.argmax()]:+.2f} s (mean corr {acc.max() / max(1, len(ms)):.3f})")


def window_eval(ms, sig, mp, W, L):
    Y, S, rho_y, rho_s = [], [], [], []
    for d in ms:
        for al, A_ in d["alliances"].items():
            tm = A_["teams"]
            if not all(d["timed"].get(t) for t in tm):
                continue
            edges = np.arange(d["t0"], d["t1"] + W, W)
            y = np.array([binned(d["truth"][t][mp], edges) for t in tm])
            s = np.array([binned(d["sig"][t].get(sig, []) if sig != "even" else [], edges, L) for t in tm])
            if sig == "even":
                s = np.ones_like(y)
            rho_y.append(y.ravel()); rho_s.append(s.ravel())
            keep = y.sum(0) >= 3
            Y.append(y[:, keep].T); S.append(s[:, keep].T)
    Y, S = np.concatenate(Y), np.concatenate(S)
    yt, st = Y.sum(1), S.sum(1)
    ys = Y / yt[:, None]
    ss = np.where(st[:, None] > 0, S / np.maximum(st, 1e-12)[:, None], 1 / 3)
    has = st > 0
    mis = float((0.5 * np.abs(ss - ys).sum(1) * yt).sum() / yt.sum())
    cover = float(yt[has].sum() / yt.sum())
    corr = float(np.corrcoef(ss[has].ravel(), ys[has].ravel())[0, 1]) if has.sum() > 3 else float("nan")
    uniq = has & (np.sort(ys, 1)[:, -1] > np.sort(ys, 1)[:, -2])
    top = float((ss[uniq].argmax(1) == ys[uniq].argmax(1)).mean()) if uniq.any() else float("nan")
    rho = spearman(np.concatenate(rho_s), np.concatenate(rho_y))
    ok = has & (np.ptp(ys, 1) > 1e-9) & (np.ptp(ss, 1) > 1e-9)
    wrho = float(np.nanmean([np.corrcoef(rank(a_), rank(b_))[0, 1] for a_, b_ in zip(ss[ok], ys[ok])])) if ok.any() else float("nan")
    return {"mis": mis, "cover": cover, "corr": corr, "top": top, "rho": rho, "wrho": wrho, "n_win": int(len(yt))}


def ceiling(ms, W):
    """Two scouts of the same robot (rows with time-valid taps): robot-window Spearman between them, and the
    window-level misassigned share of scout set A vs set B where all 3 robots of an alliance have 2 scouts."""
    sys.path.insert(0, str(ROOT))
    from autoscout import scouting
    bm = scouting.by_match(scouting.load())
    ra, rb, YA, YB = [], [], [], []
    for d in ms:
        for al, A_ in d["alliances"].items():
            edges = np.arange(d["t0"], d["t1"] + W, W)
            pairs = []
            for t in A_["teams"]:
                rows = [x for x in bm.get(d["key"], {}).get(t, []) if x.time_valid and x.n > 0]
                if len(rows) >= 2:
                    a_, b_ = (binned([(d["auto_start"] + v - 0.8, 1.0) for v in x.shots], edges) for x in rows[:2])
                    ra.append(a_); rb.append(b_); pairs.append((a_, b_))
            if len(pairs) == 3:
                YA.append(np.array([p_[0] for p_ in pairs]).T); YB.append(np.array([p_[1] for p_ in pairs]).T)
    out = {"rho": spearman(np.concatenate(ra), np.concatenate(rb)), "n_robots": len(ra)}
    if YA:
        YA, YB = np.concatenate(YA), np.concatenate(YB)
        k = (YA.sum(1) >= 3) & (YB.sum(1) >= 3)
        ya, yb = YA[k] / YA[k].sum(1, keepdims=True), YB[k] / YB[k].sum(1, keepdims=True)
        w = (YA[k].sum(1) + YB[k].sum(1)) / 2
        out.update(mis=float((0.5 * np.abs(ya - yb).sum(1) * w).sum() / w.sum()),
                   corr=float(np.corrcoef(ya.ravel(), yb.ravel())[0, 1]), n_win=int(k.sum()), n_alliances=len(YA))
    return out


def match_eval(ms, sig):
    P_, R_, T_ = [], [], []
    for d in ms:
        for al, A_ in d["alliances"].items():
            tm = A_["teams"]
            if not all(d["scout_n"].get(t) is not None for t in tm) or not A_["tba_total"]:
                continue
            ref = np.array([d["scout_n"][t] for t in tm], float)
            if ref.sum() <= 0:
                continue
            s = np.ones(3) if sig == "even" else np.array([sum(w for _, w in d["sig"][t].get(sig, [])) for t in tm])
            P_.append(s / s.sum() if s.sum() > 0 else np.ones(3) / 3); R_.append(ref / ref.sum()); T_.append(A_["tba_total"])
    P_, R_, T_ = np.array(P_), np.array(R_), np.array(T_, float)
    return {"tv": float((0.5 * np.abs(P_ - R_).sum(1)).mean()), "corr": float(np.corrcoef(P_.ravel(), R_.ravel())[0, 1]),
            "top": float((P_.argmax(1) == R_.argmax(1)).mean()), "mae": float((np.abs(P_ - R_) * T_[:, None]).mean()),
            "n": int(len(P_))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="cont")
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--sigs", default=",".join(["even"] + SIGS))
    ap.add_argument("--nomapcheck", action="store_true")
    ap.add_argument("--extra", default="", help="comma list of dirs under data/exp_launch with <key>.pkl {team: {sig: events}}")
    a = ap.parse_args()

    def extra(d):
        out = {}
        for dn in [x for x in a.extra.split(",") if x]:
            p_ = ROOT / "data" / "exp_launch" / dn / f"{d['key']}.pkl"
            if p_.exists():
                with open(p_, "rb") as fh:
                    for t, s_ in pickle.load(fh).items():
                        out.setdefault(t, {}).update(s_)
        return out
    ms = load(extra if a.extra else None)
    print(f"{len(ms)} matches")
    if not a.nomapcheck:
        mapping_check(ms)
    print(f"\nwindow {a.win:g} s, scout mapping {a.map}")
    print(f"scout-vs-scout ceiling: {ceiling(ms, a.win)}")
    print(f"{'signal':10s} {'lag':>5s} {'mis':>6s} {'cover':>6s} {'corr':>6s} {'top':>5s} {'rho':>6s} {'wrho':>6s} | match: {'TV':>6s} {'corr':>6s} {'top':>5s} {'MAE':>6s}")
    for sg in a.sigs.split(","):
        best = None
        for L in ([0.0] if sg == "even" else np.arange(-3, 3.01, 0.25)):
            m = window_eval(ms, sg, a.map, a.win, L)
            if best is None or m["mis"] < best[1]["mis"]:
                best = (L, m)
        L, m = best
        mm = match_eval(ms, sg)
        print(f"{sg:10s} {L:+5.2f} {100 * m['mis']:5.1f}% {100 * m['cover']:5.1f}% {m['corr']:6.3f} {100 * m['top']:4.0f}% {m['rho']:6.3f} {m['wrho']:6.3f} |"
              f"        {mm['tv']:6.3f} {mm['corr']:6.3f} {100 * mm['top']:4.0f}% {mm['mae']:6.1f}   (win n={m['n_win']}, match n={mm['n']})")


if __name__ == "__main__":
    main()
