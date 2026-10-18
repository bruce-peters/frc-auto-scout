"""Append per-jump launch-signal features to the v5 credit-model dataset -> data/exp_launch/ds_launch.pkl.

For each alliance record of data/exp_audit/ds_vol_v5.pkl (= your ds_v5.pkl + s_beq) and each robot:
  s_<name> = sum(attribution.kernel(t_jump - (t_event + shift)) * w)
over that robot's events in data/exp_launch/sig*/<key>.pkl. shift = the signal's fitted window lag minus the
credited launches' (leval.py, 4 s windows), so every signal is aligned with launch times like s_all.
Records without signals (unscouted matches) get 0.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import attribution as A  # noqa: E402

L = ROOT / "data" / "exp_launch"
SIGS = {"dino": ("sig_dino", "dino", 0.5), "dinodm": ("sig_dino", "dino_dm", 0.5),
        "dinolto": ("sig_dino", "dinolto", 0.5), "dinoltodm": ("sig_dino", "dinolto_dm", 0.5),
        "dinoc": ("sig_dino", "dinoc", 0.5), "dinoclto": ("sig_dino", "dinoclto", 0.5), "still": ("sig", "still", -1.0),
        "drain": ("sig", "drain", 0.5), "drainraw": ("sig", "drainraw", 0.5), "rise": ("sig", "rise", 0.75),
        "flux": ("sig_flux", "flux", float(os.environ.get("FLUX_SHIFT", "0"))),
        "fluxall": ("sig_flux", "fluxall", float(os.environ.get("FLUX_SHIFT", "0")))}
with open(ROOT / "data" / "exp_audit" / "ds_vol_v5.pkl", "rb") as fh:
    d = pickle.load(fh)
feats = list(d["feats"]) + [f"s_{n}" for n in SIGS]
cache = {}
n_with = 0
for rec in d["recs"]:
    k = rec["key"]
    if k not in cache:
        cache[k] = {}
        for dn in sorted({v[0] for v in SIGS.values()}):
            p = L / dn / f"{k}.pkl"
            if not p.exists() and dn == "sig":
                p = L / "sig_other" / f"{k}.pkl"  # unscouted matches (lsig.py --other)
            if not p.exists() and dn == "sig_dino" and os.environ.get("DINO_FULL"):
                p = L / "sig_dino_full" / f"{k}.pkl"  # unscouted matches: the full-data ridge
            if p.exists():
                with open(p, "rb") as fh:
                    x = pickle.load(fh)
                cache[k][dn] = x["sig"] if dn == "sig" else x
    if "X" not in rec:
        continue
    X = np.asarray(rec["X"])
    if X.ndim != 3 or not len(X):
        rec["X"] = X.reshape(0, 3, len(feats)) if X.size == 0 else X
        continue
    T = np.asarray(rec["t"], float)
    add = np.zeros((len(T), 3, len(SIGS)))
    for ri, team in enumerate(rec["teams"]):
        for si, (n, (dn, key, sh)) in enumerate(SIGS.items()):
            ev = cache[k].get(dn, {}).get(team, {}).get(key, [])
            if ev:
                a = np.asarray(ev, float)
                add[:, ri, si] = (A.kernel(T[:, None] - (a[None, :, 0] + sh)) * a[None, :, 1]).sum(1)
    n_with += bool(cache[k])
    rec["X"] = np.concatenate([X, add], 2)
d["feats"] = feats
out_ds = Path(os.environ.get("DS_OUT", L / "ds_launch.pkl"))
with open(out_ds, "wb") as fh:
    pickle.dump(d, fh)
print(f"{len(d['recs'])} records ({n_with} with launch signals), feats + {list(SIGS)} -> {out_ds}")
