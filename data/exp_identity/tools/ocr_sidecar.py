"""Bumper reads missing from a scan's ocr_cache (reanalyse re-reads them on the GPU and never saves the
scan; on the CPU EasyOCR takes ~2 s per crop). This tool finds them and fills a sidecar cache
data/exp_identity/ocr_sidecar/<match>.pkl ({crop_id: reads}) that absent_eval.py merges into md.ocr_cache.

  ocr_sidecar.py count [--matches ...]   # list misses per match (fast: OCR stubbed)
  ocr_sidecar.py fill  [--matches ...]   # read the misses on the CPU (slow), resumable per match
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from absent_eval import wait_mem  # noqa: E402
from autoscout import ocr, pipeline  # noqa: E402

SIDE = ROOT / "data" / "exp_identity" / "ocr_sidecar"
MISS = ROOT / "data" / "exp_identity" / "ocr_misses.json"


def load_side(key):
    p = SIDE / f"{key}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["count", "fill"])
    ap.add_argument("--matches", nargs="*", default=[])
    a = ap.parse_args()
    SIDE.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.parent.name for p in (ROOT / "output").glob("*/scan.pkl"))
    if a.matches:
        keys = [k for k in keys if k in a.matches]
    misses = json.loads(MISS.read_text()) if MISS.exists() else {}
    if a.mode == "count":
        real = ocr.read_numbers
        ocr.read_numbers = lambda img: []
        for key in keys:
            if key in misses:
                continue
            wait_mem()
            try:
                md = pickle.load(open(ROOT / "output" / key / "scan.pkl", "rb"))
            except Exception as e:
                print(key, "load failed", e, flush=True)
                continue
            md.ocr_cache = dict(md.ocr_cache or {})
            side = load_side(key)
            md.ocr_cache.update(side)
            before = set(md.ocr_cache)
            pipeline.analyse(md, pipeline.Options())
            new = sorted(set(md.ocr_cache) - before)
            misses[key] = new
            MISS.write_text(json.dumps(misses))
            print(f"{key}: {len(new)} misses (cache {len(before)})", flush=True)
            md = None
        ocr.read_numbers = real
        print("total misses", sum(len(v) for v in misses.values()))
    else:
        for key in sorted(keys, key=lambda k: len(misses.get(k, []))):
            todo = [c for c in misses.get(key, [])]
            side = load_side(key)
            todo = [c for c in todo if c not in side]
            if not todo:
                continue
            wait_mem()
            md = pickle.load(open(ROOT / "output" / key / "scan.pkl", "rb"))
            tic = time.time()
            for i, cid in enumerate(todo):
                img = pipeline._img(md.crops[cid][0])
                side[cid] = ocr.read_numbers(img) if img is not None else []
                if (i + 1) % 50 == 0 or i + 1 == len(todo):
                    pickle.dump(side, open(SIDE / f"{key}.pkl", "wb"))
            print(f"{key}: filled {len(todo)} in {time.time() - tic:.0f}s", flush=True)
            md = None


if __name__ == "__main__":
    main()
