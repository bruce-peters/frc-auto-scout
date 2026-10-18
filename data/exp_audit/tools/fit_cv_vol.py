"""Learn the per-robot credit weights of split_increments from 5026's scouting; leave-one-match-out CV.

Weight of robot r for a scoreboard jump (alliance total stays exact, the weights are normalised over the 3):

    w_r = (zone_r + 1e-3)^gz * E_r^b * exp(g_opr * log opr_share_r + g_typ * log typ_share_r)

with launch evidence E_r in one of two forms
    lin: E = sum_k a_k S_k + p * vis + h * (1 - vis)                  (old rule: a=1, p=0.12, h=2, b=gz=1, g=0)
    gp : E = (sum_k a_k S_k + alpha) / (cv_own + r_oth * cv_other + r_none * cv_none)   (Gamma-Poisson rate:
         observed launches over expected recall of the views the robot was in)
S_k = kernel-weighted launch confidence split by link kind or by camera view.
Positive parameters are fitted in log space with an L2 penalty pulling them to the old rule's values.

Loss: cross-entropy of the match share (sum over jumps) against the scouted share, optionally plus a per-jump
cross-entropy against kernel-weighted scouted shot times. CV: each scouted match held out in turn; the
ridge strength is picked inside each training fold by grouped 5-fold CV (nested), so held-out numbers are
honest.

  .venv/Scripts/python data/exp_attrmodel/tools/fit_cv.py DS.pkl [--quick] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

LOG_FLOOR = 0.02
PERIODS = ["auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame", "unknown"]


# ----------------------------------------------------------------------------- data
def load(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return d["feats"], d["recs"]


def prior_logs(rec):
    o = np.asarray(rec["opr"], float)
    so = o / o.sum() if np.all(np.isfinite(o)) and o.sum() > 0 else np.full(3, 1 / 3)
    t = np.asarray(rec["typ"], float)
    t = np.where(np.isfinite(t), t, 1 / 3)
    st = t / t.sum()
    return np.log(np.maximum(so, LOG_FLOOR)), np.log(np.maximum(st, LOG_FLOOR)), so, st


class Pack:
    """All alliances' jumps concatenated, for fast full-batch fitting."""

    def __init__(self, recs, feats):
        self.recs = recs
        fi = {f: i for i, f in enumerate(feats)}
        X, N, A, AUTO, LO, LT, Q, QW, POFF = [], [], [], [], [], [], [], [], []
        for ai, r in enumerate(recs):
            J = len(r["n"])
            X.append(r["X"])
            N.append(r["n"])
            A.append(np.full(J, ai))
            AUTO.append(np.array([p == "auto" for p in r["period"]], float))
            lo, lt, _, _ = prior_logs(r)
            LO.append(np.tile(lo, (J, 1)))
            LT.append(np.tile(lt, (J, 1)))
            POFF.append(np.tile(np.asarray(r.get("prior_off", np.zeros(3)), float), (J, 1)))  # parked robots
            if "scout_inc" in r:
                q = r["scout_inc"]
                s = q.sum(1, keepdims=True)
                Q.append(np.where(s > 0, q / np.maximum(s, 1e-9), 1 / 3))
                QW.append(r["n"] * np.minimum(1.0, s[:, 0]))  # jumps with scouted shots nearby
            else:
                Q.append(np.full((J, 3), 1 / 3))
                QW.append(np.zeros(J))
        self.fi = fi
        self.X = torch.tensor(np.concatenate(X))
        self.N = torch.tensor(np.concatenate(N))
        self.A = torch.tensor(np.concatenate(A), dtype=torch.long)
        self.auto = torch.tensor(np.concatenate(AUTO))
        self.lopr = torch.tensor(np.concatenate(LO))
        self.ltyp = torch.tensor(np.concatenate(LT))
        self.poff = torch.tensor(np.concatenate(POFF))
        self.Q = torch.tensor(np.concatenate(Q))
        self.QW = torch.tensor(np.concatenate(QW))
        self.nA = len(recs)
        self.lopr_A = torch.tensor(np.array([prior_logs(r)[0] for r in recs]))
        self.AP = torch.tensor(np.concatenate([ai * len(PERIODS) + np.array([PERIODS.index(p) if p in PERIODS else 0
                                                                            for p in r["period"]], int)
                                               for ai, r in enumerate(recs)]), dtype=torch.long)
        ref = np.array([r["scout_n"] / r["scout_n"].sum() if "scout_n" in r else np.full(3, 1 / 3) for r in recs])
        self.ref = torch.tensor(ref)

    def f(self, name):
        return self.X[:, :, self.fi[name]]

    @property
    def dist_z(self):
        """distance to own hub in the window, (d - 3.5 m) / 2 m, 0 where the position is unknown."""
        if not hasattr(self, "_dz"):
            d = self.f("dist")
            self._dz = torch.where(torch.isnan(d), torch.zeros_like(d), (d - 3.5) / 2.0)
        return self._dz

    @property
    def lM(self):
        """log relative match-level launch rate of each robot within its alliance."""
        if not hasattr(self, "_lM"):
            m = self.f("s_match") + 1.0
            self._lM = torch.log(m / m.mean(1, keepdim=True))
        return self._lM


# ----------------------------------------------------------------------------- model
SETS = {"all": ["s_all"], "kind": ["s_box", "s_proj", "s_projh"], "view": ["s_own", "s_oth", "s_main"], "beq": ["s_beq"]}


class Spec:
    def __init__(self, form="lin", sset="all", priors=(), auto_b=False, fit_zone=True, fixed=False, extras=()):
        self.form, self.sset, self.priors, self.auto_b, self.fit_zone, self.fixed = form, sset, tuple(priors), auto_b, fit_zone, fixed
        self.extras = tuple(extras)  # "mid": + c_mid * s_mid; "mrate": pseudo-shots * M^gam; "mprior": M^g_m

    def name(self):
        if self.fixed:
            return "old rule (rebuilt)"
        return f"{self.form}/{self.sset}" + "".join(f"+{e}" for e in self.extras) + \
            (f"+{'+'.join(self.priors)}" if self.priors else "") + \
            ("+auto_b" if self.auto_b else "") + ("" if self.fit_zone else "-zone")

    def init(self):
        """Parameter -> (init value = old rule, regularised?)"""
        th = {}
        ks = SETS[self.sset]
        for i, k in enumerate(ks):
            if i > 0 or self.form == "gp":  # first gain fixed at 1 for lin (scale redundancy)
                th[f"la_{k}"] = 0.0
        if self.form == "lin":
            th["lp"] = float(np.log(0.12))
            th["lh"] = float(np.log(2.0))
            if "cov" in self.extras:  # pseudo-shots by exclusive camera coverage instead of seen / hidden
                th["lp_oth"] = float(np.log(0.12))
        else:
            th["lalpha"] = float(np.log(0.3))
            th["lr_oth"] = float(np.log(0.4))
            th["lr_none"] = float(np.log(0.2))
        th["lb"] = 0.0
        if self.auto_b:
            th["lb_auto"] = 0.0
        if self.fit_zone:
            th["gz"] = 1.0
        for p in self.priors:
            th[f"g_{p}"] = 0.0
        if "mid" in self.extras:
            th["lc_mid"] = float(np.log(0.01))
        if "mrate" in self.extras:
            th["gam"] = 0.0
        if "mprior" in self.extras:
            th["g_m"] = 0.0
        if "dist" in self.extras or "pos" in self.extras:
            th["g_dist"] = 0.0
        if "pos" in self.extras:
            th["g_zl"] = 0.0
            th["g_cvown"] = 0.0
        if "cvown" in self.extras:
            th["g_cvown"] = 0.0
        if "mlprior" in self.extras:
            th["g_ml"] = 0.0
        if "vol" in self.extras:  # failure-finder cycle 27: volley ball-equivalents added to the launch evidence
            th["lc_vol"] = float(np.log(0.02))
        if "volexp" in self.extras:  # ... or as a multiplicative volume factor (s_beq + 1)^g_vol
            th["g_vol"] = 0.0
        if "plprior" in self.extras:
            th["g_pl"] = 0.0
        return th

    def logw(self, th, P: Pack):
        ks = SETS[self.sset]
        S = 0.0
        for i, k in enumerate(ks):
            a = torch.exp(th[f"la_{k}"]) if f"la_{k}" in th else 1.0
            S = S + a * P.f(k)
        if "mid" in th:
            S = S + torch.exp(th["lc_mid"]) * P.f("s_mid")
        if "lc_vol" in th:
            S = S + torch.exp(th["lc_vol"]) * P.f("s_beq")
        Mr = P.lM if ("gam" in th or "g_m" in th) else None
        if self.form == "lin":
            if "lp_oth" in th:
                pseudo = (torch.exp(th["lp"]) * P.f("cv_own") + torch.exp(th["lp_oth"]) * P.f("cv_other")
                          + torch.exp(th["lh"]) * P.f("cv_none"))
            else:
                vis = P.f("vis")
                pseudo = torch.exp(th["lp"]) * vis * (1 - P.poff) + torch.exp(th["lh"]) * (1 - vis)
            if "gam" in th:
                pseudo = pseudo * torch.exp(th["gam"] * Mr)
            E = S + pseudo
        else:
            den = P.f("cv_own") + torch.exp(th["lr_oth"]) * P.f("cv_other") + torch.exp(th["lr_none"]) * P.f("cv_none")
            alpha = torch.exp(th["lalpha"]) * (torch.exp(th["gam"] * Mr) if "gam" in th else 1.0)
            E = (S + alpha) / (den + 0.05)
        b = torch.exp(th["lb"])
        if self.auto_b:
            b = b + (torch.exp(th["lb_auto"]) - b) * P.auto[:, None]
        lw = b * torch.log(E + 1e-9)
        gz = th["gz"] if "gz" in th else 1.0
        lw = lw + gz * torch.log(P.f("zone") + 1e-3)
        if "g_opr" in th:
            lw = lw + th["g_opr"] * P.lopr
        if "g_typ" in th:
            lw = lw + th["g_typ"] * P.ltyp
        if "g_m" in th:
            lw = lw + th["g_m"] * Mr
        if "g_dist" in th:
            lw = lw + th["g_dist"] * P.dist_z
        if "g_zl" in th:
            lw = lw + th["g_zl"] * P.f("zone")
        if "g_cvown" in th:
            lw = lw + th["g_cvown"] * P.f("cv_own")
        if "g_vol" in th:
            lw = lw + th["g_vol"] * torch.log1p(P.f("s_beq"))
        return lw

    def jump_p(self, th, P):
        return torch.softmax(self.logw(th, P), dim=1)

    def shares(self, th, P):
        p = self.jump_p(th, P)
        tot = torch.zeros(P.nA, 3).index_add_(0, P.A, P.N[:, None] * p)
        sh = tot / tot.sum(1, keepdim=True).clamp_min(1e-9)
        if "g_ml" in th:  # match-level prior on the summed shares
            sh = torch.softmax(torch.log(sh + 1e-9) + th["g_ml"] * P.lopr_A, dim=1)
        if "g_pl" in th:  # the same prior applied to each scoring period's shares (period totals stay exact)
            nAP = P.nA * len(PERIODS)
            tp = torch.zeros(nAP, 3).index_add_(0, P.AP, P.N[:, None] * p)
            nt = tp.sum(1, keepdim=True)
            spp = tp / nt.clamp_min(1e-9)
            lo = P.lopr_A.repeat_interleave(len(PERIODS), dim=0)
            spp = torch.softmax(torch.log(spp + 1e-9) + th["g_pl"] * lo, dim=1)
            tot = (spp * nt).reshape(P.nA, len(PERIODS), 3).sum(1)
            sh = tot / tot.sum(1, keepdim=True).clamp_min(1e-9)
        return sh


def old_theta():
    return Spec("lin", "all", fit_zone=True).init()


def fit(spec: Spec, P: Pack, lam: float, mu: float = 0.0, iters: int = 200):
    th0 = spec.init()
    names = list(th0)
    x0 = torch.tensor([th0[k] for k in names])
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-7)

    def closure():
        opt.zero_grad()
        th = dict(zip(names, x))
        sh = spec.shares(th, P)
        loss = -(P.ref * torch.log(sh + 1e-9)).sum(1).mean()
        if mu > 0 and P.QW.sum() > 0:
            p = spec.jump_p(th, P)
            ce = -(P.Q * torch.log(p + 1e-9)).sum(1)
            loss = loss + mu * (P.QW * ce).sum() / P.QW.sum()
        loss = loss + lam * ((x - x0) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {k: float(v) for k, v in zip(names, x.detach())}


def predict(spec: Spec, th, P: Pack) -> np.ndarray:
    with torch.no_grad():
        return spec.shares({k: torch.tensor(v) for k, v in th.items()}, P).numpy()


# ----------------------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, recs) -> dict:
    ref = np.array([r["scout_n"] / r["scout_n"].sum() for r in recs])
    tot = np.array([r["tba_total"] for r in recs], float)
    tv = 0.5 * np.abs(pred - ref).sum(1)
    top = (pred.argmax(1) == ref.argmax(1)).mean()
    corr = np.corrcoef(pred.ravel(), ref.ravel())[0, 1]
    mae = (np.abs(pred - ref) * tot[:, None]).mean()
    fcorr = np.corrcoef((pred * tot[:, None]).ravel(), (ref * tot[:, None]).ravel())[0, 1]
    # slope of predicted vs reference share deviations from 1/3 (compression)
    slope = np.polyfit((ref - 1 / 3).ravel(), (pred - 1 / 3).ravel(), 1)[0]
    return {"tv": float(tv.mean()), "corr": float(corr), "top": float(top), "fuel_mae": float(mae),
            "fuel_corr": float(fcorr), "slope": float(slope), "n": len(recs)}


def fmt(m):
    return (f"TV {m['tv']:.3f}  corr {m['corr']:.3f}  top {100 * m['top']:.0f} %  fuel MAE {m['fuel_mae']:.1f}  "
            f"fuel corr {m['fuel_corr']:.3f}  slope {m['slope']:.2f}")


def baselines(recs):
    out = {}
    vs = np.array([r["video_share"] for r in recs])
    old = np.array([r["old_unscaled"] / r["old_unscaled"].sum() for r in recs])
    so = np.array([prior_logs(r)[2] for r in recs])
    st = np.array([prior_logs(r)[3] for r in recs])
    out["video only (stored)"] = vs
    out["old rule (rebuilt)"] = old
    out["even 1/3"] = np.full_like(vs, 1 / 3)
    out["fuel-OPR split"] = so
    out["typical video share"] = st
    out["0.7 video + 0.3 OPR"] = 0.7 * vs + 0.3 * so
    return out


# ----------------------------------------------------------------------------- CV
def groups(recs):
    keys = sorted({r["key"] for r in recs})
    return keys, np.array([keys.index(r["key"]) for r in recs])


def nested_lam(spec, recs, feats, lams, mu, k=5, seed=0):
    keys, g = groups(recs)
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(keys)) % k
    best, best_l = None, None
    for lam in lams:
        loss = []
        for f in range(k):
            tr = [r for r, gi in zip(recs, g) if fold_of[gi] != f]
            te = [r for r, gi in zip(recs, g) if fold_of[gi] == f]
            th = fit(spec, Pack(tr, feats), lam, mu)
            pr = predict(spec, th, Pack(te, feats))
            ref = np.array([r["scout_n"] / r["scout_n"].sum() for r in te])
            loss.append(-(ref * np.log(pr + 1e-9)).sum(1))
        loss = float(np.concatenate(loss).mean())
        if best is None or loss < best:
            best, best_l = loss, lam
    return best_l


def lomo(spec, recs, feats, lams, mu=0.0, nested=True):
    keys, g = groups(recs)
    pred = np.zeros((len(recs), 3))
    thetas, chosen = [], []
    for gi in range(len(keys)):
        tr = [r for r, gg in zip(recs, g) if gg != gi]
        te_idx = [i for i, gg in enumerate(g) if gg == gi]
        lam = nested_lam(spec, tr, feats, lams, mu) if (nested and len(lams) > 1) else lams[0]
        th = fit(spec, Pack(tr, feats), lam, mu)
        pred[te_idx] = predict(spec, th, Pack([recs[i] for i in te_idx], feats))
        thetas.append(th)
        chosen.append(lam)
    return pred, thetas, chosen


def match_level_lomo(recs, use=("opr",)):
    """Baseline: share ∝ video_share^b * opr_share^g (fitted LOMO), a match-level sharpen + prior."""
    keys, g = groups(recs)
    vs = np.log(np.maximum(np.array([r["video_share"] for r in recs]), LOG_FLOOR))
    lo = np.array([prior_logs(r)[0] for r in recs])
    lt = np.array([prior_logs(r)[1] for r in recs])
    ref = np.array([r["scout_n"] / r["scout_n"].sum() for r in recs])
    F = [vs] + ([lo] if "opr" in use else []) + ([lt] if "typ" in use else [])
    F = torch.tensor(np.stack(F, -1))
    R = torch.tensor(ref)
    pred = np.zeros_like(ref)
    coefs = []
    for gi in range(len(keys)):
        m = torch.tensor(g != gi)
        x = torch.tensor([1.0] + [0.0] * (F.shape[-1] - 1), requires_grad=True)
        opt = torch.optim.LBFGS([x], max_iter=100, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            p = torch.softmax((F[m] * x).sum(-1), 1)
            loss = -(R[m] * torch.log(p)).sum(1).mean()
            loss.backward()
            return loss

        opt.step(cl)
        with torch.no_grad():
            pred[g == gi] = torch.softmax((F[torch.tensor(g == gi)] * x).sum(-1), 1).numpy()
        coefs.append(x.detach().numpy().tolist())
    return pred, np.array(coefs)


def parse_spec(txt: str) -> Spec:
    head, *toks = txt.split("+")
    form, sset = head.split("/")
    pri = tuple(t for t in toks if t in ("opr", "typ"))
    ext = tuple(t for t in toks if t in ("mrate", "mprior", "mid", "cov", "dist", "pos", "cvown", "mlprior", "plprior", "vol", "volexp"))
    return Spec(form, sset, priors=pri, extras=ext, auto_b="auto_b" in toks, fit_zone="-zone" not in txt)


def boot_diff(pa, pb, recs, n=2000, seed=1):
    """Paired bootstrap over matches of metric(pa) - metric(pb): 95 % intervals."""
    keys, g = groups(recs)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("tv", "corr", "top", "fuel_mae")}
    idx_of = [np.where(g == i)[0] for i in range(len(keys))]
    for _ in range(n):
        pick = np.concatenate([idx_of[i] for i in rng.integers(0, len(keys), len(keys))])
        rr = [recs[i] for i in pick]
        ma, mb = metrics(pa[pick], rr), metrics(pb[pick], rr)
        for k in out:
            out[k].append(ma[k] - mb[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


DEFAULT_SPECS = ("lin/all;lin/kind;lin/view;lin/all+cov;gp/view;lin/all+mrate;"
                 "lin/all+opr;lin/kind+opr;lin/view+opr;lin/all+cov+opr;gp/view+opr;lin/all+mrate+opr;"
                 "lin/all+opr+typ;lin/all+opr+auto_b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--quick", action="store_true", help="fixed lambda, no nesting")
    ap.add_argument("--out")
    ap.add_argument("--specs", default=DEFAULT_SPECS)
    ap.add_argument("--mus", default="0")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--lam", type=float, help="fixed lambda (no nesting), e.g. export_model.py's 3e-4")
    a = ap.parse_args()
    feats, recs = load(a.ds)
    sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
          and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
    print(f"{len(sc)} scouted alliances in {len({r['key'] for r in sc})} matches "
          f"({sum('scout_inc' in r for r in sc)} with timed scouting)")
    res = {"baselines": {}, "models": {}, "n_alliances": len(sc), "n_matches": len({r['key'] for r in sc})}
    base = baselines(sc)
    for name, pr in base.items():
        m = metrics(pr, sc)
        res["baselines"][name] = m
        print(f"  {name:40s} {fmt(m)}")
    for use in (("opr",), ("opr", "typ")):
        pr, co = match_level_lomo(sc, use)
        m = metrics(pr, sc)
        nm = "match-level video^b*" + "*".join(f"{u}^g" for u in use) + " (LOMO)"
        res["baselines"][nm] = dict(m, coef_mean=co.mean(0).tolist())
        print(f"  {nm:40s} {fmt(m)}  coef {co.mean(0).round(2)}")
    blend = base["0.7 video + 0.3 OPR"]
    lams = [0.003] if a.quick else [0.0, 3e-5, 3e-4, 3e-3, 3e-2]
    for txt in a.specs.split(";"):
        spec = parse_spec(txt)
        for mu in [float(x) for x in a.mus.split(",")]:
            pred, ths, lam = (lomo(spec, sc, feats, [a.lam], mu, nested=False) if a.lam is not None
                              else lomo(spec, sc, feats, lams, mu, nested=not a.quick))
            m = metrics(pred, sc)
            mean_th = {k: float(np.mean([t[k] for t in ths])) for k in ths[0]}
            nm = spec.name() + (f" mu={mu}" if mu else "")
            entry = dict(m, theta=mean_th, theta_sd={k: float(np.std([t[k] for t in ths])) for k in ths[0]},
                         lams=[float(x) for x in lam], pred=pred.tolist())
            print(f"  {nm:40s} {fmt(m)}", flush=True)
            print(f"      theta {json.dumps({k: round(v, 2) for k, v in mean_th.items()})}  "
                  f"lam {dict(zip(*[x.tolist() for x in np.unique(lam, return_counts=True)]))}", flush=True)
            if a.boot:
                entry["vs_blend_ci"] = boot_diff(pred, blend, sc)
                print(f"      vs blend 95% CI: {json.dumps({k: [round(x, 3) for x in v] for k, v in entry['vs_blend_ci'].items()})}",
                      flush=True)
            res["models"][nm] = entry
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
