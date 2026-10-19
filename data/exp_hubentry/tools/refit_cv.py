"""Fair "best estimate" comparison: the learned credit model (models/attr_model.json) was fitted on today's
broad-kernel launch evidence for these same 54 scouted matches, so the base arm is in-sample. This refits the
model's six numbers (prior, hidden, b, gz, g_dist, g_opr) leave-one-match-out for each arm's launch evidence and
scores the held-out matches the way score_scouting does (team-match fuel error vs the scouted split of the TBA
total; top scorer). Same objective for every arm: cross-entropy of the alliance's match share vs the scouted share,
plus a small L2 pull to today's values.

  .venv/Scripts/python data/exp_hubentry/tools/refit_cv.py --arms base,entry_q,entry_q_mix0.3
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import attribution as A, config, hubentry as H  # noqa: E402
from score_scouting import score  # noqa: E402
import ab  # noqa: E402

DST = config.DATA / "exp_hubentry"
LAM = 1e-3


def arm_spec(name):
    """(kind, kwargs) describing how pieces / kernels are formed, mirroring ab.ARMS."""
    specs = {"base": ("base", {}),
             "entry_q": ("entry", {"kern": (H.FLIGHT_LO, H.FLIGHT, H.FLIGHT_HI)}),
             "entry_q_mix0.3": ("entry", {"kern": (H.FLIGHT_LO, H.FLIGHT, H.FLIGHT_HI), "mix": 0.3}),
             "entry_q_mix0.5": ("entry", {"kern": (H.FLIGHT_LO, H.FLIGHT, H.FLIGHT_HI), "mix": 0.5}),
             "entry_q_wide": ("entry", {"kern": (0.0, 0.9, 2.5)}),
             "const_lag1.6": ("const", {"kern": (H.FLIGHT_LO, H.FLIGHT, H.FLIGHT_HI), "lag": 1.6})}
    return specs[name]


def features(name, k, cap, cache, hf):
    """Per scouted alliance of match k: list of pieces with per-robot features for the learned model."""
    kind, kw = arm_spec(name)
    a0 = cap["auto_start"] or 6.13
    ts_ = cap["teleop_start"] or a0 + 25.1
    t0, t1 = a0 - 1.0, ts_ + config.TELEOP_LEN + 8.0
    tl = ab.TL(cap["auto_start"], cap["teleop_start"])
    out = {}
    for c in cap["calls"]:
        if c["model"] is None:
            continue
        teams = c["teams"]
        al = "red" if teams[0].team in cap["red"] else "blue"
        inc = c["inc"]
        if kind == "base":
            ents = {t: [H.Retimed(t, t, n, 1.0)] for t, n in inc if n > 0}
        else:
            if kind == "const":
                lst = [H.Retimed(t, t - H.READ_GAP / 2 - kw["lag"], n, 1.0) for t, n in inc if n > 0]
            else:
                rt, rate = H.entry_series(cache["views"], hf, al, t0, t1)
                lst = H.retime_queue(inc, rt, rate)
            ents = {}
            for e in lst:
                ents.setdefault(e.count_t, []).append(e)
        if kind != "base":
            lo_, pk, hi_ = kw["kern"]
            kern = ab.tri_kernel(lo_, pk, hi_, H.KERNEL_AREA)
        rows = []
        for t, n in inc:
            if n <= 0:
                continue
            mt = tl.match_time(t - 2.0)
            per = A.period_of(mt) if mt is not None else "unknown"
            gate = A.AUTO_ZONE_GATE or per != "auto"
            pieces = ents.get(t) or [H.Retimed(t, t, n, 0.0)]
            tot = sum(p.n for p in pieces)
            for p in pieces:
                te = p.entry_t
                if kind == "base" or p.conf <= 0:
                    kf, lo, hi = A.kernel, t - A.LAG_HI, t - A.LAG_LO
                    dt0 = t
                elif kw.get("mix"):
                    m = kw["mix"]
                    off = t - te

                    def kf(dt, off=off, m=m):
                        return (1 - m) * kern(dt) + m * A.kernel(dt + off)
                    lo, hi = min(te - hi_, t - A.LAG_HI), max(te - lo_, t - A.LAG_LO)
                    dt0 = te
                else:
                    kf, lo, hi = kern, te - hi_, te - lo_
                    dt0 = te
                F = np.zeros((3, 7))  # s, z, vis, pres, prior_on, hid_on, dz
                for i, ts in enumerate(teams):
                    F[i, 0] = float((kf(dt0 - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
                    F[i, 1] = ts.zone_frac(lo, hi) if gate else 1.0
                    F[i, 2] = ts.vis_frac(lo, hi)
                    F[i, 3] = 1.0 - ts.absent_frac(lo, hi) if ts.absent else 1.0
                    F[i, 4] = 0.0 if ts.prior_off else 1.0
                    F[i, 5] = 0.0 if ts.no_hidden else 1.0
                    d = ts.dist_mean(lo, hi)
                    F[i, 6] = 0.0 if np.isnan(d) else (d - 3.5) / 2.0
                rows.append((n * p.n / tot, F))
        opr = c["opr"]
        o = np.array([opr.get(ts.team, np.nan) for ts in teams], float) if opr else np.full(3, np.nan)
        out[al] = {"teams": [ts.team for ts in teams], "W": np.array([r[0] for r in rows]),
                   "F": np.stack([r[1] for r in rows]) if rows else np.zeros((0, 3, 7)), "opr": o}
    return out


THETA0_KEYS = ("lp", "lh", "lb", "gz", "g_dist", "g_opr")


def theta0():
    m = A.load_model()
    return np.array([np.log(m["prior"]), np.log(m["hidden"]), np.log(m["b"]), m["gz"], m["g_dist"], m["g_opr"]])


def shares(th, al):
    p, h, b = np.exp(th[0]), np.exp(th[1]), np.exp(th[2])
    gz, gd, go = th[3], th[4], th[5]
    F = al["F"]
    if not len(F):
        return np.full(3, 1 / 3)
    s, z, vis, pres, pon, hon, dz = (F[:, :, j] for j in range(7))
    E = s + (p * pon * vis + h * hon * (1 - vis)) * pres
    lw = b * np.log(E + 1e-9) + gz * np.log(z + 1e-3) + gd * dz
    w = np.exp(lw - lw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    tot = (al["W"][:, None] * w).sum(0)
    sh = tot / tot.sum()
    o = al["opr"]
    if go and np.all(np.isfinite(o)) and o.sum() > 0:
        l2 = np.log(sh + 1e-9) + go * np.log(np.maximum(o / o.sum(), 0.02))
        sh = np.exp(l2 - l2.max())
        sh /= sh.sum()
    return sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,entry_q,entry_q_mix0.3")
    a = ap.parse_args()
    # scouted alliances and the reference (exactly score_scouting's rows)
    s = score(DST / "ab" / "base", keep_rows=True)
    ref = {}
    for r in s["rows"]:
        ref.setdefault((r["key"], r["al"]), {})[r["team"]] = r["ref"]
    keys = sorted({k for k, _ in ref})
    print(len(keys), "matches", len(ref), "alliances", flush=True)
    res = {}
    for arm in a.arms.split(","):
        data = {}
        for k in keys:
            cap, base, cache, hf = ab.load_ctx(k)
            f = features(arm, k, cap, cache, hf)
            for al, d in f.items():
                if (k, al) in ref:
                    d["ref"] = np.array([ref[(k, al)][t] for t in d["teams"]])
                    data[(k, al)] = d
        th0 = theta0()

        def loss(th, ks):
            L = 0.0
            for kk in ks:
                d = data[kk]
                q = d["ref"] / d["ref"].sum()
                L -= float((q * np.log(shares(th, d) + 1e-9)).sum())
            return L / len(ks) + LAM * float(((th - th0) ** 2).sum())

        def evaluate(th_by_match):
            err, tops, n, na = 0.0, 0, 0, 0
            for kk, d in data.items():
                sh = shares(th_by_match[kk[0]], d)
                tot = d["ref"].sum()
                err += float(np.abs(sh * tot - d["ref"]).sum())
                n += 3
                tops += int(np.argmax(sh) == np.argmax(d["ref"]))
                na += 1
            return err / n, tops / na

        ins = evaluate({k: th0 for k in keys})
        th_all = minimize(loss, th0, args=(list(data),), method="L-BFGS-B").x
        fitted_in = evaluate({k: th_all for k in keys})
        th_cv = {}
        for k in keys:
            tr = [kk for kk in data if kk[0] != k]
            th_cv[k] = minimize(loss, th_all, args=(tr,), method="L-BFGS-B").x
        cv = evaluate(th_cv)
        per_match = {}
        for kk, d in data.items():
            sh = shares(th_cv[kk[0]], d)
            per_match.setdefault(kk[0], [0.0, 0])
            per_match[kk[0]][0] += float(np.abs(sh * d["ref"].sum() - d["ref"]).sum())
            per_match[kk[0]][1] += 3
        res[arm] = {"today_model_in_sample": {"mae": ins[0], "top": ins[1]},
                    "refit_all_in_sample": {"mae": fitted_in[0], "top": fitted_in[1], "theta": th_all.tolist()},
                    "refit_lomo": {"mae": cv[0], "top": cv[1]}, "per_match": per_match}
        print(f"{arm:16s} today's model {ins[0]:.2f} / top {100 * ins[1]:.1f} % | refit in-sample {fitted_in[0]:.2f} / "
              f"{100 * fitted_in[1]:.1f} % | refit LOMO {cv[0]:.2f} / {100 * cv[1]:.1f} % | theta {np.round(th_all, 3)}",
              flush=True)
    if "base" in res:
        rng = np.random.default_rng(0)
        ks = sorted(res["base"]["per_match"])
        for arm in res:
            if arm == "base":
                continue
            pa, pb = res[arm]["per_match"], res["base"]["per_match"]
            d = []
            for _ in range(2000):
                bs = rng.choice(ks, len(ks))
                d.append(sum(pa[x][0] for x in bs) / sum(pa[x][1] for x in bs)
                         - sum(pb[x][0] for x in bs) / sum(pb[x][1] for x in bs))
            res[arm]["lomo_diff_vs_base_ci95"] = np.percentile(d, [2.5, 97.5]).tolist()
            print(f"{arm}: LOMO MAE diff vs base {res[arm]['refit_lomo']['mae'] - res['base']['refit_lomo']['mae']:+.2f} "
                  f"CI {np.round(np.percentile(d, [2.5, 97.5]), 2)}", flush=True)
    (DST / "refit_cv.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
