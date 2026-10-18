"""Cycle 1: the launch -> overlay-counter timing kernel (attribution.LAG_LO/PEAK/HI).

1. reproduce analyse()'s split from result.json (sanity check of common.resplit)
2. empirical launch->counter impulse response: pooled NNLS deconvolution of the counter increments on
   the detected launches (each alliance-match normalised to gain 1), per event family / period / view
3. sweep kernel shapes through split_increments -> share metrics vs fuel OPR
Writes data/exp_audit/c1_kernel.json. Usage: c1_kernel.py [v2|v1]
"""
import json
import sys

import numpy as np
from scipy.optimize import nnls

from common import A, AUDIT, load, match_keys, resplit, share_metrics, tba, team_signals, TL

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
keys = [k for k in match_keys(src)]
BIN = 0.2
LAGS = np.arange(-10, 51)  # bins: -2 s .. +10 s

# ---------------- 1. reproduction
rep = []
R, INFO = {}, {}
for k in keys:
    r = load(k, src)
    info = tba.match(k)
    R[k], INFO[k] = r, info
    fu = resplit(r, info)
    for t in info.red + info.blue:
        rep.append((fu[t], r["teams"][str(t)]["fuel"]))
rep = np.array(rep)
print(f"{len(keys)} matches ({src}); reproduction: max |diff| {np.abs(rep[:, 0] - rep[:, 1]).max():.2f} fuel, "
      f"corr {np.corrcoef(rep.T)[0, 1]:.4f}")


# ---------------- 2. deconvolution
def design(r, al, sel):
    tl = TL(r)
    if not tl.counts.get(al):
        return None
    inc = A.increments_from_counts(tl.counts[al])
    t0, t1 = tl.t0(), tl.t1()
    nb = int((t1 - t0) / BIN) + 1
    y = np.zeros(nb)
    for t, n in inc:
        i = int((t - t0) / BIN)
        if 0 <= i < nb:
            y[i] += n
    L = np.zeros(nb)
    for s in r["shots"]:
        if s["alliance"] == al and s["team"] is not None and s["toward"] and sel(s, tl):
            i = int((s["t"] - t0) / BIN)
            if 0 <= i < nb:
                L[i] += 1
    if L.sum() < 10 or y.sum() < 10:
        return None
    L *= y.sum() / L.sum()
    X = np.zeros((nb, len(LAGS)))
    for j, l in enumerate(LAGS):
        if l >= 0:
            X[l:, j] = L[:nb - l]
        else:
            X[:nb + l, j] = L[-l:]
    return X, y


def deconv(sel=lambda s, tl: True, keys_=None):
    Xs, ys = [], []
    for k in keys_ or keys:
        for al in ("red", "blue"):
            d = design(R[k], al, sel)
            if d is not None:
                Xs.append(d[0]); ys.append(d[1])
    if not Xs:
        return None
    X, y = np.vstack(Xs), np.concatenate(ys)
    # second-difference smoothness penalty (keeps NNLS from spiking)
    n = X.shape[1]
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1, -2, 1)
    lam = 0.3 * np.sqrt((X ** 2).sum() / n)
    h, _ = nnls(np.vstack([X, lam * D]), np.concatenate([y, np.zeros(n - 2)]))
    return h, len(Xs)


def describe(h):
    lag = LAGS * BIN
    pos = lag >= 0
    m = h.sum()
    cdf = np.cumsum(h) / m
    q = lambda p: float(lag[np.searchsorted(cdf, p)])
    mode = float(lag[np.argmax(h)])
    # share of the response mass inside the current kernel's support, and kernel-weighted overlap
    ker = A.kernel(lag)
    inside = float(h[(lag >= A.LAG_LO) & (lag <= A.LAG_HI)].sum() / m)
    neg = float(h[~pos].sum() / m)
    return {"mode": mode, "q10": q(0.1), "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "q90": q(0.9),
            "mass_in_kernel_support": round(inside, 3), "mass_negative_lag": round(neg, 3),
            "gain": round(float(m), 3)}


def is_chezy(k):
    return R[k]["timeline"]["layout"] == "chezy"


out = {"src": src, "n_matches": len(keys)}
variants = {
    "all": (lambda s, tl: True, None),
    "first_layout": (lambda s, tl: True, [k for k in keys if not is_chezy(k)]),
    "chezy_layout": (lambda s, tl: True, [k for k in keys if is_chezy(k)]),
    "auto_only": (lambda s, tl: (tl.match_time(s["t"]) or 99) < 20, None),
    "teleop_only": (lambda s, tl: (tl.match_time(s["t"]) or 0) >= 20, None),
    "view_main": (lambda s, tl: s["view"] == "main", None),
    "view_end": (lambda s, tl: s["view"] != "main", None),
    "how_box": (lambda s, tl: s["how"] == "box", None),
    "how_proj": (lambda s, tl: s["how"] in ("proj", "proj_hidden"), None),
}
print("\nempirical launch->counter response (s):")
for name, (sel, ks) in variants.items():
    if ks is not None and not ks:
        continue
    res = deconv(sel, ks)
    if res is None:
        continue
    h, n = res
    d = describe(h)
    d["alliance_matches"] = n
    d["h"] = [round(float(x), 4) for x in h]
    out[name] = d
    print(f"  {name:14s} n={n:3d} mode {d['mode']:4.1f} q10 {d['q10']:4.1f} q25 {d['q25']:4.1f} med {d['median']:4.1f} "
          f"q75 {d['q75']:4.1f} q90 {d['q90']:4.1f}  in [{A.LAG_LO},{A.LAG_HI}]: {d['mass_in_kernel_support']:.2f} "
          f"neg-lag {d['mass_negative_lag']:.2f}")
h = np.array(out["all"]["h"])
print("  lag(s): " + " ".join(f"{l * BIN:4.1f}" for l in LAGS[::3]))
print("  h     : " + " ".join(f"{x / h.max():4.2f}" for x in h[::3]))

# ---------------- 3. kernel sweep
SIG = {k: team_signals(R[k], INFO[k]) for k in keys}
base = (A.LAG_LO, A.LAG_PEAK, A.LAG_HI)
grid = [base, (0.5, 2.0, 4.0), (0.5, 1.8, 3.5), (0.6, 2.2, 4.5), (0.8, 2.4, 4.5), (1.0, 2.5, 4.0), (0.3, 1.5, 3.0),
        (0.8, 2.8, 4.5), (0.8, 2.0, 5.5), (1.2, 3.2, 6.0)]
sweep = []
print("\nkernel sweep (share corr vs fuel OPR / top scorer / MAE):")
for lo, pk, hi in grid:
    A.LAG_LO, A.LAG_PEAK, A.LAG_HI = lo, pk, hi
    pm = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in keys}
    m = share_metrics(pm)
    m["kernel"] = (lo, pk, hi)
    sweep.append(m)
    print(f"  LO {lo:3.1f} PEAK {pk:3.1f} HI {hi:3.1f}: corr {m['corr']:.3f}  top {m['top']}/{m['alliances']}  "
          f"MAE {m['mae']:.1f}")
A.LAG_LO, A.LAG_PEAK, A.LAG_HI = base
out["sweep"] = sweep
(AUDIT / f"c1_kernel_{src}.json").write_text(json.dumps(out, indent=1))
print("wrote", AUDIT / f"c1_kernel_{src}.json")
