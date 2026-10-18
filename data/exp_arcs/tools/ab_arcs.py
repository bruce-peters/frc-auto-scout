"""Pass 3: A/B of the 3-D arc term in the launch link on the scouted Chezy quals (output/ is not written).

Each scan is analysed in memory once per variant with arc_hook.ArcTerm (pipeline.analyse_arc); result.json goes
to data/exp_arcs/ab/<variant>/<match>/. The baseline arm is pass 1's data/exp_arcs/base (the unchanged analysis,
current defaults LINK_X_TIE = 1, LINK_X_CROSS = True). Scored with scripts/score_scouting.score, paired by match.

  .venv/Scripts/python -u data/exp_arcs/tools/ab_arcs.py --variants "w1_s100:W=1,SIGMA2=100" [--score-only]

Upper-case keys in a variant set pipeline module constants for that arm only (e.g. SPLIT_TRACKS=True,
GATE_TRACE=True; reset after the arm). LOG=<dir> runs that arm with arc_hook.Logger (the unchanged analysis) and
writes its contested launches to data/exp_arcs/<dir>/<match>.pkl. --out names the arm folder (default ab).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, pipeline, scouting  # noqa: E402
import arc_hook  # noqa: E402

EXP = config.DATA / "exp_arcs"


def parse(spec):
    out = {}
    for item in spec.split(";"):
        name, _, kv = item.partition(":")
        kw = {}
        for p in filter(None, kv.split(",")):
            k, _, v = p.partition("=")
            try:
                kw[k] = float(v) if v not in ("None",) else None
            except ValueError:
                kw[k] = v
        out[name] = kw
    return out


def per_match(s):
    """{match: rows} from score(keep_rows=True)."""
    by = {}
    for r in s["rows"]:
        by.setdefault(r["key"], []).append(r)
    return by


def metrics(rows):
    R = {m: np.array([r[m] for r in rows]) for m in ("ref", "video", "vonly")}
    out = {}
    for m in ("video", "vonly"):
        out[m + "_mae"] = float(np.abs(R[m] - R["ref"]).mean())
        out[m + "_corr"] = float(np.corrcoef(R[m], R["ref"])[0, 1])
    g = {}
    for i, r in enumerate(rows):
        g.setdefault((r["key"], r["al"]), []).append(i)
    for m in ("video", "vonly"):
        out[m + "_top"] = float(np.mean([idx[int(np.argmax(R[m][idx]))] == idx[int(np.argmax(R["ref"][idx]))]
                                         for idx in g.values()]))
    return out


def paired(base, var, n=4000, seed=0):
    """Variant minus base per metric, with a paired bootstrap over matches (95 % CI)."""
    bm, vm = per_match(base), per_match(var)
    ks = sorted(set(bm) & set(vm))
    d0 = {k: v - metrics([r for kk in ks for r in bm[kk]])[k] for k, v in metrics([r for kk in ks for r in vm[kk]]).items()}
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in d0}
    for _ in range(n):
        s = [ks[i] for i in rng.choice(len(ks), len(ks))]
        a, b = metrics([r for kk in s for r in vm[kk]]), metrics([r for kk in s for r in bm[kk]])
        for k in d0:
            bs[k].append(a[k] - b[k])
    mae_b = {k: np.abs(np.array([r["vonly"] - r["ref"] for r in bm[k]])).mean() for k in ks}
    mae_v = {k: np.abs(np.array([r["vonly"] - r["ref"] for r in vm[k]])).mean() for k in ks}
    better = sum(mae_v[k] < mae_b[k] - 1e-9 for k in ks)
    worse = sum(mae_v[k] > mae_b[k] + 1e-9 for k in ks)
    return {k: (d0[k], *np.percentile(bs[k], [2.5, 97.5])) for k in d0}, len(ks), better, worse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help='"name:W=1,SIGMA2=100;name2:..."')
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--base", default="ab/base0", help="baseline arm folder under data/exp_arcs (run it in the same pass)")
    ap.add_argument("--out", default="ab", help="arm folders go to data/exp_arcs/<out>/<arm>/")
    ap.add_argument("--scored-only", action="store_true", help="only matches that score_scouting can use (every robot scouted)")
    a = ap.parse_args()
    variants = parse(a.variants)
    for kw in variants.values():
        if "gate_base" in kw:
            kw["gate_base"] = bool(kw["gate_base"])
    keys = a.matches or sorted(k for k in scouting.by_match(scouting.load()) if (config.OUTPUT / k / "scan.pkl").exists())
    if a.scored_only:
        from score_scouting import score as _score  # noqa: E402
        keys = sorted({r["key"] for r in _score(EXP / "ab" / "base0", keep_rows=True)["rows"]} & set(keys))
    consts, logs = {}, {}
    for name, kw in variants.items():
        const_keys = [k for k in kw if k.isupper() and k not in ("W", "SIGMA2", "LOG")]
        for k in const_keys:
            assert hasattr(pipeline, k), f"pipeline has no constant {k}"
        consts[name] = {k: (kw[k] == "True" if kw[k] in ("True", "False") else kw[k]) for k in const_keys}
        logs[name] = kw.pop("LOG", None)
        for k in const_keys:
            kw.pop(k)
    OUT = EXP / a.out
    if not a.score_only:
        arc_hook.build()
        arc_hook.install_gate_wrapper()
        import hashlib
        from autoscout import attribution
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
        am = attribution.ATTR_MODEL
        amp = Path(am) if Path(am).is_absolute() else ROOT / am
        print("pipeline.py", sha(ROOT / "autoscout" / "pipeline.py"), "attribution.py", sha(ROOT / "autoscout" / "attribution.py"),
              "credit model", am, sha(amp) if amp.exists() else None, "arcs.py", sha(ROOT / "autoscout" / "arcs.py"), flush=True)
        print(len(keys), "scouted matches", flush=True)
        tic = time.time()
        stats = {v: [0, 0] for v in variants}
        for k in keys:
            with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
                raw = fh.read()
            md = pickle.loads(raw)
            del raw
            pipeline.load_dino(md, config.OUTPUT / k)
            for name, kw in variants.items():
                saved = {c: getattr(pipeline, c) for c in consts[name]}
                for c, v in consts[name].items():
                    setattr(pipeline, c, v)
                hook = arc_hook.ArcTerm(**kw)
                if logs[name]:
                    lg = arc_hook.Logger()
                    pipeline._ARC_HOOK = lg
                else:
                    pipeline._ARC_HOOK = hook if kw.get("W", 1.0) else None  # W = 0: the unchanged analysis
                res = pipeline.analyse_arc(md, pipeline.Options())
                pipeline._ARC_HOOK = None
                for c, v in saved.items():
                    setattr(pipeline, c, v)
                if logs[name]:
                    (EXP / logs[name]).mkdir(parents=True, exist_ok=True)
                    with open(EXP / logs[name] / f"{k}.pkl", "wb") as fh:
                        pickle.dump(lg.events, fh, protocol=pickle.HIGHEST_PROTOCOL)
                od = OUT / name / k
                od.mkdir(parents=True, exist_ok=True)
                (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
                stats[name][0] += hook.n_used
                stats[name][1] += hook.n_changed
            print(k, f"{time.time() - tic:.0f}s", {n: tuple(s) for n, s in stats.items()}, flush=True)
            del md
        json.dump(stats, open(OUT / "hook_stats.json", "w"))

    from score_scouting import score  # noqa: E402

    base = score(EXP / a.base, keep_rows=True)
    b = base
    print(f"base: video only MAE {b['vonly']['mae']:.2f} corr {b['vonly']['corr']:.3f} top {100 * b['top_scorer']['vonly']:.0f} % | "
          f"best estimate MAE {b['video']['mae']:.2f} corr {b['video']['corr']:.3f} | burst {100 * b['burst_overlap']['video']:.1f} % "
          f"| {b['matches']} matches", flush=True)
    out = {"base": {k: v for k, v in base.items() if k != "rows"}}
    for name in variants:
        if (OUT / name).resolve() == (EXP / a.base).resolve():
            continue
        s = score(OUT / name, keep_rows=True)
        d, n, better, worse = paired(base, s)
        out[name] = {"score": {k: v for k, v in s.items() if k != "rows"}, "paired": d, "n": n, "better": better, "worse": worse}
        print(f"{name}: video only MAE {s['vonly']['mae']:.2f} corr {s['vonly']['corr']:.3f} top {100 * s['top_scorer']['vonly']:.0f} % | "
              f"best estimate MAE {s['video']['mae']:.2f} corr {s['video']['corr']:.3f} | burst {100 * s['burst_overlap']['video']:.1f} %", flush=True)
        for k, (m, lo, hi) in d.items():
            print(f"    {k:11s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"    video-only MAE better in {better} matches, worse in {worse} (of {n})", flush=True)
    json.dump(out, open(OUT / f"ab_summary_vs_{Path(a.base).name}.json", "w"), indent=1, default=float)
