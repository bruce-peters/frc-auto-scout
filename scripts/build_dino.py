"""DINOv2 appearance embeddings for every stored robot crop (identity evidence).

The cached scans keep each crop's detection box and time but not the pixels, so this re-reads
the needed frames from the match video, crops the robot (letterboxed to a square), and embeds
it with frozen DINOv2 ViT-S/14 (CLS token + mean patch token, L2-normalised). Runs on CPU so it
does not compete with GPU training. Writes <dir>/<match>/dino.npz with `ids` and `feats`.

    python scripts/build_dino.py                  # all matches in data/eval_snapshot and output/
    python scripts/build_dino.py --workers 4 --size 112
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MODEL = None


def _model(threads: int):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(threads)
        repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        _MODEL = torch.hub.load(str(repo), "dinov2_vits14", source="local", verbose=False).eval()
    return _MODEL


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def embed(crops: list[np.ndarray], threads: int) -> np.ndarray:
    import torch
    m = _model(threads)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.from_numpy(np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[i:i + 64]]))
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            f = m.forward_features(x)
            v = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], dim=1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def one(args):
    key, size, threads, dirs = args
    from autoscout.video import Video, video_path
    targets = [Path(d) / key / "dino.npz" for d in dirs if (Path(d) / key).exists()]
    if all(t.exists() for t in targets):
        return key, "have"
    src = next(Path(d) / key / "scan.pkl" for d in dirs if (Path(d) / key / "scan.pkl").exists())
    with open(src, "rb") as fh:
        md = pickle.load(fh)
    want = sorted(((d.t, d.crop_id, d.box) for d in md.dets if d.crop_id is not None), key=lambda x: x[0])
    if not want:
        return key, "no crops"
    v = Video(video_path(key))
    by_idx: dict[int, list] = {}
    for t, cid, box in want:
        by_idx.setdefault(int(round(t * v.fps)), []).append((cid, box))
    idxs = sorted(by_idx)
    ids, crops = [], []
    for idx, frame in v.iter_frames(idxs[0], idxs[-1] + 1):
        if idx not in by_idx:
            continue
        for cid, box in by_idx[idx]:
            x0, y0, x1, y1 = [int(round(c)) for c in box]
            c = frame[max(0, y0):y1, max(0, x0):x1]
            if c.size == 0:
                continue
            ids.append(cid)
            crops.append(letterbox(c, size))
    v.close()
    feats = embed(crops, threads).astype(np.float16)
    for t in targets:
        np.savez_compressed(t, ids=np.array(ids, np.int32), feats=feats)
    return key, f"{len(ids)} crops"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--dirs", nargs="*", default=[str(ROOT / "data" / "eval_snapshot"), str(ROOT / "output")])
    ap.add_argument("--matches", nargs="*", default=[])
    a = ap.parse_args()
    keys = a.matches or sorted({p.parent.name for d in a.dirs for p in Path(d).glob("*/scan.pkl")})
    with ProcessPoolExecutor(a.workers) as ex:
        for key, msg in ex.map(one, [(k, a.size, a.threads, a.dirs) for k in keys]):
            print(key, msg, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
