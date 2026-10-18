"""A/B two robot detectors on the same matches (dirs written by scripts/rescan_robots.py).

For each dir: re-analyse every cached scan with the current code (so both sides use identical
tracking / identity / attribution code), then
  * held-out identity agreement   (scripts/identity_holdout.py --out <dir>)
  * team-level shares vs fuel OPR  (scripts/evaluate.py --out <dir>)
  * detection / continuity stats   (robots per 10 Hz step by view, fused-robot coverage,
                                    segments per match, OCR evidence)
and a paired bootstrap over matches for the differences B - A. Writes <out>/compare.json.

  python scripts/compare_detectors.py --a data/exp_robots_v2/v1 --b data/exp_robots_v2/v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
import identity_holdout  # noqa: E402
from autoscout import pipeline, robots  # noqa: E402

# which analysis code both sides were scored with (other sessions edit autoscout/ concurrently)
_SRC = Path(pipeline.__file__).parent
CODE = {"fusion": getattr(robots, "FUSION", None),
        "src_sha1": __import__("hashlib").sha1(b"".join(p.read_bytes() for p in sorted(_SRC.glob("*.py")))).hexdigest()[:12],
        "w_cls": getattr(pipeline.Options(), "w_cls", None)}


def det_stats(md) -> dict:
    """Detections per 10 Hz step by view, and how many of the 3 robots per alliance are observed."""
    by_t: dict[float, list] = {}
    for d in md.dets:
        by_t.setdefault(round(d.t, 3), []).append(d)
    n = max(1, len(by_t))
    out = {f"dets_{v}": sum(d.view == v for d in md.dets) / n for v in ("main", "blue_end", "red_end")}
    out["dets_end_xy"] = sum(d.view != "main" and d.xy is not None for d in md.dets) / n
    cover = []
    for ds in by_t.values():
        fused = robots.fuse(ds)
        for al in ("red", "blue"):
            cover.append(min(3, sum(1 for f in fused if f.get("alliance") == al)) / 3)
    out["coverage"] = float(np.mean(cover)) if cover else 0.0
    return out


def side(d: Path, keys: list[str], reanalyse: bool) -> dict:
    per = {}
    for k in keys:
        if reanalyse:
            pipeline.reanalyse(k, pipeline.Options(out_dir=d))
        with open(d / k / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        pipeline.analyse(md, pipeline.Options(out_dir=d))  # fills d.xy for the stats
        r = json.loads((d / k / "result.json").read_text())
        st = det_stats(md)
        st["segments"] = r.get("n_segments")
        st["ocr_evidence"] = float(sum(sum(s["ocr"]) for al in ("red", "blue") for s in r["identity"][al].values()))
        per[k] = st
    identity_holdout.main(out=d, quiet=True)
    ih = json.loads((d / "identity_holdout.json").read_text())
    summ = evaluate.main(out=d)
    ev = json.loads((d / "evaluation.json").read_text())
    return {"per_match": per, "identity": ih, "team": summ, "team_rows": ev["teams"]}


def boot(keys, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ks = list(rng.choice(keys, len(keys)))
        try:
            vals.append(fn(ks))
        except Exception:
            continue
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline dir (e.g. v1 rescans)")
    ap.add_argument("--b", required=True, help="candidate dir (e.g. v2 rescans)")
    ap.add_argument("--out", default=None, help="where compare.json goes (default: parent of --b)")
    ap.add_argument("--no-reanalyse", action="store_true")
    a = ap.parse_args()
    da, db = Path(a.a), Path(a.b)
    keys = sorted({p.parent.name for p in da.glob("*/scan.pkl")} & {p.parent.name for p in db.glob("*/scan.pkl")})
    print(f"{len(keys)} matches in both dirs", flush=True)
    A, B = side(da, keys, not a.no_reanalyse), side(db, keys, not a.no_reanalyse)

    def ident(S, ks):
        pm = S["identity"]["per_match"]
        return sum(pm[k][0] for k in ks) / max(1e-9, sum(pm[k][1] for k in ks))

    def team(S, ks, field):
        # a bootstrap sample can draw a match twice: tag each draw so its alliances stay separate
        rows = [{**r, "match": f"{k}#{i}"} for i, k in enumerate(ks) for r in S["team_rows"] if r["match"] == k]
        return evaluate.team_metrics(rows)[field]

    # both sides were analysed in this one process, i.e. with the same imported code snapshot
    res = {"matches": keys, "a": str(da), "b": str(db), "metrics": {}, "code": CODE}
    print(f"\n{'metric':28s} {'A':>9s} {'B':>9s} {'B-A':>8s}  95% CI of B-A (paired bootstrap)  P(B>A)")
    rows = [("identity agreement %", lambda S, ks: 100 * ident(S, ks)),
            ("team share corr", lambda S, ks: team(S, ks, "corr")),
            ("top scorer match %", lambda S, ks: team(S, ks, "top_pct")),
            ("|video-OPR share| fuel", lambda S, ks: team(S, ks, "mae"))]
    for f in ("coverage", "dets_main", "dets_blue_end", "dets_red_end", "dets_end_xy", "segments", "ocr_evidence"):
        rows.append((f, lambda S, ks, f=f: float(np.mean([S["per_match"][k][f] for k in ks]))))
    for name, fn in rows:
        va, vb = fn(A, keys), fn(B, keys)
        lo, hi, p = boot(keys, lambda ks: fn(B, ks) - fn(A, ks))
        res["metrics"][name] = {"a": va, "b": vb, "diff": vb - va, "ci95": [lo, hi], "p_b_gt_a": p}
        print(f"{name:28s} {va:9.3f} {vb:9.3f} {vb - va:+8.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:.2f}")
    events = sorted({k.split("_")[0] for k in keys})
    res["by_event"] = {}
    for e in events:
        ks = [k for k in keys if k.startswith(e + "_")]
        res["by_event"][e] = {"n": len(ks), "identity_a": 100 * ident(A, ks), "identity_b": 100 * ident(B, ks),
                              "coverage_a": float(np.mean([A["per_match"][k]["coverage"] for k in ks])),
                              "coverage_b": float(np.mean([B["per_match"][k]["coverage"] for k in ks]))}
        x = res["by_event"][e]
        print(f"  {e:12s} n={len(ks):2d} identity {x['identity_a']:5.1f} -> {x['identity_b']:5.1f}   "
              f"coverage {x['coverage_a']:.3f} -> {x['coverage_b']:.3f}")
    res["per_match"] = {k: {"a": {**A["per_match"][k], "identity": A["identity"]["per_match"][k]},
                            "b": {**B["per_match"][k], "identity": B["identity"]["per_match"][k]}} for k in keys}
    out = Path(a.out) if a.out else db.parent
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out / "compare.json")


if __name__ == "__main__":
    main()
