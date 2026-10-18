"""Scouting A/B of flight-tracker variants (link_x_ab.py / ab_arcs.py style; output/ is not written).

Each scouted scan is analysed in memory once per arm with its flight tracks replaced by a tracker variant re-run
on the stored blob log (data/exp_track/tools/track_lab.py), pipeline module constants set for that arm only, and
optionally the 3-D arc check (data/exp_arcs/tools/arc_hook.ArcTerm(W=0.3, SIGMA2=1, prior="soft",
gate_base=True), via pipeline.analyse_arc). result.json goes to data/exp_track/ab/<arm>/<match>/, scored with
scripts/score_scouting.score and paired by match against the first arm (ab_arcs.paired).

  python data/exp_track/tools/track_ab.py --arms "base:TRACK=stored;bss:TRACK=ballistic+split+stitch,GATE_TRACE=True"
      TRACK=stored keeps the scan's own tracks; ARC=1 adds the 3-D arc check.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "scripts", ROOT / "data" / "exp_arcs" / "tools", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))
from autoscout import config, pipeline, scouting  # noqa: E402
import arc_hook  # noqa: E402
import track_lab as TL  # noqa: E402

EXP = config.DATA / "exp_track"


def parse(spec: str) -> dict:
    arms = {}
    for item in spec.split(";"):
        name, _, kv = item.partition(":")
        d = {}
        for p in filter(None, kv.split(",")):
            k, _, v = p.partition("=")
            d[k] = {"True": True, "False": False}.get(v, v)
        arms[name] = d
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--out", default="ab")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    arms = parse(a.arms)
    OUT = EXP / a.out
    bym = scouting.by_match(scouting.load())
    keys = a.matches or sorted(k for k in bym if (config.OUTPUT / k / "scan.pkl").exists()
                               and (TL.BLOBS / f"{k}.pkl").exists() and k != "2026cc_qm14")
    if not a.score_only:
        arc_hook.build()
        arc_hook.install_gate_wrapper()
        print(len(keys), "scouted matches", flush=True)
        tic = time.time()
        for k in keys:
            todo = [n for n in arms if not (OUT / n / k / "result.json").exists()]
            if not todo:
                continue
            with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
                md = pickle.load(fh)
            pipeline.load_dino(md, config.OUTPUT / k)
            stored = md.tracks_all
            with open(TL.BLOBS / f"{k}.pkl", "rb") as fh:
                log = pickle.load(fh)
            cache = {}
            for name in todo:
                kw = arms[name]
                tv = kw.get("TRACK", "stored")
                if tv == "stored":
                    md.tracks_all = stored
                else:
                    if tv not in cache:
                        cache[tv] = TL.run(tv, log["blob_log"], md, log["fuel_step"])
                    md.tracks_all = cache[tv]
                consts = {c: v for c, v in kw.items() if c.isupper() and c not in ("TRACK", "ARC")}
                saved = {c: getattr(pipeline, c) for c in consts}
                for c, v in consts.items():
                    setattr(pipeline, c, float(v) if isinstance(v, str) and v.replace(".", "", 1).isdigit() else v)
                pipeline._ARC_HOOK = (arc_hook.ArcTerm(W=0.3, SIGMA2=1.0, prior="soft", gate_base=True)
                                      if str(kw.get("ARC", "0")) == "1" else None)
                res = pipeline.analyse_arc(md, pipeline.Options())
                pipeline._ARC_HOOK = None
                for c, v in saved.items():
                    setattr(pipeline, c, v)
                od = OUT / name / k
                od.mkdir(parents=True, exist_ok=True)
                (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
            md.tracks_all = stored
            del md, log, cache
            print(k, f"{time.time() - tic:.0f}s", flush=True)

    from score_scouting import score  # noqa: E402
    from ab_arcs import paired  # noqa: E402

    names = list(arms)
    base = score(OUT / names[0], keep_rows=True)
    b = base
    print(f"{names[0]}: video only MAE {b['vonly']['mae']:.2f} corr {b['vonly']['corr']:.3f} top {100 * b['top_scorer']['vonly']:.0f} % | "
          f"best estimate MAE {b['video']['mae']:.2f} corr {b['video']['corr']:.3f} top {100 * b['top_scorer']['video']:.0f} % | "
          f"burst {100 * b['burst_overlap']['video']:.1f} % | {b['matches']} matches", flush=True)
    summary = {names[0]: {k: v for k, v in base.items() if k != "rows"}}
    for name in names[1:]:
        s = score(OUT / name, keep_rows=True)
        d, n, better, worse = paired(base, s)
        summary[name] = {"score": {k: v for k, v in s.items() if k != "rows"}, "paired": d, "n": n,
                         "better": better, "worse": worse}
        print(f"{name}: video only MAE {s['vonly']['mae']:.2f} corr {s['vonly']['corr']:.3f} top {100 * s['top_scorer']['vonly']:.0f} % | "
              f"best estimate MAE {s['video']['mae']:.2f} corr {s['video']['corr']:.3f} top {100 * s['top_scorer']['video']:.0f} % | "
              f"burst {100 * s['burst_overlap']['video']:.1f} %", flush=True)
        for k_, (m, lo, hi) in d.items():
            print(f"    {k_:11s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"    video-only MAE better in {better} matches, worse in {worse} (of {n})", flush=True)
    json.dump(summary, open(OUT / f"summary_vs_{names[0]}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
