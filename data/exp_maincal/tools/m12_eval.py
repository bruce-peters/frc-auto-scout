"""Before / after on the scan copies (data/exp_maincal/before/<match>/scan.pkl), in memory: pipeline.analyse with
each main-camera variant. OCR: cached reads only (as failure-finder cycle 8), so every variant sees the same reads.

Variants:
  before        stored cameras, the old CameraTrack.at (rvecs blended across branches)
  at_fix        stored cameras, the fixed CameraTrack.at
  guard         at_fix + maincal.guard (window poses putting >= 20 % of their boxes off the field dropped)
  static        maincal static fit (shared K / centre, rotation per window, tags only) + guard
  static_floor  static fit with the floor term (best of its starts) + guard
  manual        static fit + the two hand-picked centre-line points (cc only; the bias reference) + guard
  event         leave-one-out event prior (median K / distortion / centre of the other matches' static fits),
                rotation per window about that centre (maincal.static_from_prior) + guard
Writes data/exp_maincal/eval/<variant>/<match>/result.json and eval/<match>.json.
Usage: m12_eval.py [--variants a,b] match...
"""
import gc
import json
import pickle
import sys
import time

import numpy as np

from common import EXP, F, M, MANUAL_PTS, P, TAGS, load_extract, main_boxes
from autoscout import ocr

ocr.read_numbers = lambda img: []
args = sys.argv[1:]
variants = None
if args and args[0] == "--variants":
    variants, args = args[1].split(","), args[2:]
OUT = EXP / "eval"
OUT.mkdir(exist_ok=True)
AT_NEW = F.CameraTrack.at


def at_old(self, t):
    ts = np.asarray(self.times)
    if len(ts) == 1 or t <= ts[0]:
        return F.Camera(self.K, self.dist, self.rvecs[0], self.tvecs[0], self.size, self.rms, self.n_tags)
    if t >= ts[-1]:
        return F.Camera(self.K, self.dist, self.rvecs[-1], self.tvecs[-1], self.size, self.rms, self.n_tags)
    j = int(np.searchsorted(ts, t))
    a = (t - ts[j - 1]) / max(1e-9, ts[j] - ts[j - 1])
    rv = (1 - a) * np.asarray(self.rvecs[j - 1]) + a * np.asarray(self.rvecs[j])
    tv = (1 - a) * np.asarray(self.tvecs[j - 1]) + a * np.asarray(self.tvecs[j])
    return F.Camera(self.K, self.dist, rv.tolist(), tv.tolist(), self.size, self.rms, self.n_tags)


def loo_prior(key: str) -> dict | None:
    """Median K / distortion / centre of the same event's other matches' static fits (static_fits.json)."""
    fits = json.loads((EXP / "static_fits.json").read_text())
    ev = key.split("_")[0]
    ent = [e for k, e in fits.items() if k.split("_")[0] == ev and k != key]
    if len(ent) < 3:
        return None
    f = float(np.median([e["K"][0][0] for e in ent]))
    ref = min(ent, key=lambda e: abs(e["K"][0][0] - f))
    return {"n": len(ent), "size": tuple(ent[0]["size"]), "f": f,
            "K": [[f, 0, ent[0]["K"][0][2]], [0, f, ent[0]["K"][1][2]], [0, 0, 1]],
            "dist": np.median([e["dist"] for e in ent], axis=0).tolist(),
            "centre": np.median([e["centre"] for e in ent], axis=0).tolist(), "rvec": ref["rvec"]}


def camera_variants(ex, which):
    plain = ex["cams"].get("main")
    obs = [(t, d) for t, d in ex["tag_obs"] if d]
    bx = main_boxes(ex)
    out, info = {}, {}
    K, D, C = M.track_start(plain)

    def g(name, tr):
        if tr is None:
            return
        tr2, rep = M.guard(tr, bx, P.ROBOT_HALF_DEPTH)
        out[name] = tr2
        info[name] = {"f": round(tr.K[0][0], 1), "centre": np.round(F.track_centre(tr), 3).tolist(), "rms": round(tr.rms, 3),
                      **rep}

    for name in which:
        if name in ("before", "at_fix"):
            out[name] = plain
        elif name == "guard":
            g(name, plain)
        elif name == "static":
            g(name, M.fit_static(obs, TAGS, ex["size"], K, D, C))
        elif name == "static_floor":
            c = M.static_candidates(obs, TAGS, ex["size"], plain=plain, boxes=bx, half_depth=P.ROBOT_HALF_DEPTH)
            g(name, next((tr for n, tr in c if n.startswith("static")), None))
        elif name == "event":
            pr = loo_prior(ex["key"])
            if pr is not None:
                g(name, M.static_from_prior(obs, TAGS, pr))
        elif name == "event_lm" and ex["key"].startswith("2026cc"):
            pr = json.loads((EXP / "prior_lm.json").read_text())
            g(name, M.static_from_prior(obs, TAGS, pr))
        elif name == "manual" and ex["key"].startswith("2026cc"):
            g(name, M.fit_static(obs, TAGS, ex["size"], K, D, C, points=MANUAL_PTS, w_points=2.0))
    return out, info, bx


ALL = ["before", "at_fix", "guard", "static", "static_floor", "manual"]
for key in args:
    tic = time.time()
    ex = load_extract(key)
    which = variants or (ALL if key.startswith("2026cc") else ["before", "static", "static_floor"])
    cams, info, bx = camera_variants(ex, which)
    obs_f, size_f = [(t, d) for t, d in ex["tag_obs"] if d], ex["size"]
    del ex
    with open(EXP / "before" / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    plain = md.cams.get("main")
    for fname in [n for n in which if n.startswith("final")]:  # the production path (pipeline.process_match)
        # final: static fit + leave-one-out median event prior (untrusted); final_lm: + the landmark camera (trusted)
        if M.centre_scatter(plain) <= M.MAIN_SCATTER_M:
            cams[fname] = plain  # not triggered: the stored camera, untouched
            info[fname] = {"accepted": None, "triggered": False}
            continue
        if fname == "final_lm":
            pr, trusted = M.load_main_prior(None, size_f, path=EXP / "prior_lm.json")
        else:
            pr, trusted = loo_prior(key), False
        cands = [c for c in M.main_candidates(obs_f, TAGS, size_f, plain) if c[0] == "static"]
        if pr is not None:
            et = M.static_from_prior(obs_f, TAGS, pr)
            if et is not None and et.rms < F.MAX_RMS:
                cands.append(("event_prior", et))
        r_ = P.accept_main_cam(md, cands, prefer="event_prior", trusted=trusted)
        cams[fname], md.cams["main"] = md.cams["main"], plain
        info[fname] = {"accepted": r_["accepted"], "flag": r_["flag"], "triggered": True,
                       "cands": [(c["how"], c["off"], c["d_xview"], c["pass"]) for c in r_["candidates"]]}
    rep = {"key": key, "variants": {}}
    jp = OUT / f"{key}.json"
    if jp.exists():
        rep = json.loads(jp.read_text())
    for name, cam in cams.items():
        F.CameraTrack.at = at_old if name == "before" else AT_NEW
        md.cams["main"] = cam
        res = P.analyse(md, P.Options())
        d = OUT / name / key
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=0, default=P._json_default), encoding="utf-8")
        agree = mass = 0.0
        for hk in (0, 1):
            rh = P.analyse(md, P.Options(ocr_holdout=hk))
            for al in ("red", "blue"):
                tm = md.info.teams(al)
                for s in rh["identity"][al].values():
                    if s["team"] is not None:
                        agree += s["held"][tm.index(s["team"])]
                        mass += sum(s["held"])
        xv = {}
        for v in ("blue_end", "red_end"):
            if v in md.cams:
                g_ = P.cross_view_error(md, v, md.cams[v])
                xv[v] = [round(g_["median"], 3), g_["n"]]
        off = float(M.off_field(cam, bx, P.ROBOT_HALF_DEPTH).mean())
        rep["variants"][name] = {"off": round(off, 4), "agree": agree, "mass": mass, "xview": xv,
                                 "fuel": {str(t): res["teams"][t]["fuel"] for t in md.info.red + md.info.blue},
                                 "n_segments": res.get("n_segments"), **info.get(name, {})}
        print(f"{key} {name:12s} off {off:.3f} held-out {agree / max(mass, 1e-9):.3f} xv {xv} {info.get(name, '')}", flush=True)
    F.CameraTrack.at = AT_NEW
    md.cams["main"] = plain
    jp.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(f"{key}: {time.time() - tic:.0f}s", flush=True)
    del md
    gc.collect()
