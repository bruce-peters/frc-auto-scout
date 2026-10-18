"""Cross-match DINOv2 robot recognition for team identity (identity evidence `Options.w_dino`).

Each team plays ~10-12 matches per event. A linear probe on frozen DINOv2 features, trained on a
team's crops from its *other* matches (labels: segments with clear bumper OCR), sees the robot
against many backgrounds and positions, so it has to learn the robot itself. Three steps:

  embed   re-read the crop frames from data/videos/<match>.mp4 and embed every stored robot crop
          with frozen DINOv2 ViT-S/14 on the GPU, several crop strategies at once
          -> data/exp_dino/emb/<match>.npz
  labels  pipeline.analyse() with the default Options -> per crop: alliance, segment, the segment's
          bumper-OCR evidence over the alliance's 3 teams, assigned team
          -> data/exp_dino/labels/<match>.npz
  probe   per event, leave-one-match-out: train a softmax probe over the event's teams on the
          OCR-confident crops of the event's other matches, apply it to the held-out match and
          store per crop the log-probabilities over each alliance's 3 teams
          -> data/exp_dino/probe/<match>.npz  (read by pipeline.analyse when Options.w_dino > 0)

    python scripts/build_dino_probe.py embed  [--matches k ...] [--batch 64]
    python scripts/build_dino_probe.py labels [--matches k ...]
    python scripts/build_dino_probe.py probe  [--feat box_inner] [--name default] [--events 2026cc ...]

Results: data/exp_dino/RESULTS_section.md. Use with Options(w_dino=2) and, to let the probe fix duplicate
tracks, the duplicate-capable Viterbi (absent_cost=2, absent_cost_dead=2, absent_near_bonus=2). --pool
(train across events) was worse than per-event training.

Crop strategies (feature names in emb/<match>.npz, 384-d each, float16):
  box_cls, box_mean   whole box stretched to 224x224: CLS token / mean of all patch tokens
  box_inner           mean of the patch tokens inside the box interior (x 20-80 %, y 15-85 %)
  box_fg              mean of the patch tokens that are mostly robot: not carpet (colour far from the
                      ring around the box) and not FUEL yellow (hopper contents change all match)
  box_low             mean of the patch tokens of the lower 45 % (bumper + chassis)
  ctr_cls, ctr_mean   centre crop (x 20-80 %, y 20-80 %) re-embedded at 224x224
  low_cls, low_mean   lower 45 % (bumper + chassis), full width, re-embedded at 112x224
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "data" / "exp_dino"
EMB, LAB, PROBE = EXP / "emb", EXP / "labels", EXP / "probe"
FEATS = ["box_cls", "box_mean", "box_inner", "box_fg", "box_low", "ctr_cls", "ctr_mean", "low_cls", "low_mean"]
VIEWS = ["main", "red_end", "blue_end"]
S = 224  # input side (16 x 16 patches of 14 px)
G = S // 14


def free_gb() -> float:
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                             capture_output=True, text=True, timeout=60).stdout
        return int(out.strip()) / 1024 ** 2
    except Exception:
        return 99.0


def wait_mem(min_gb: float):
    while (f := free_gb()) < min_gb:
        print(f"  waiting for memory: {f:.1f} GB free < {min_gb}", flush=True)
        time.sleep(60)


# ---------------------------------------------------------------- embed
def _model(dev: str):
    import torch
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    m = torch.hub.load(str(repo), "dinov2_vits14", source="local", verbose=False).eval().to(dev)
    return m.half() if dev == "cuda" else m


def _rs(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _patch_frac(mask: np.ndarray) -> np.ndarray:
    """(S, S) bool pixel mask -> (G*G,) fraction of each 14x14 patch that is set."""
    return mask.reshape(G, 14, G, 14).mean(axis=(1, 3)).ravel()


_PC = (np.arange(G) + 0.5) / G
_PX, _PY = np.meshgrid(_PC, _PC)
INNER = ((_PX >= 0.2) & (_PX <= 0.8) & (_PY >= 0.15) & (_PY <= 0.85)).ravel().astype(np.float32)
LOWER = (_PY >= 0.55).ravel().astype(np.float32)


def crop_views(frame: np.ndarray, box) -> tuple | None:
    """Whole box, centre crop and lower part of one detection, plus the patch mask of robot pixels."""
    from autoscout.fuel import yellow_mask
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(c)) for c in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return None
    c = frame[y0:y1, x0:x1]
    boxi = _rs(c, S, S)
    ctr = _rs(c[int(0.2 * h):max(int(0.2 * h) + 2, int(0.8 * h)), int(0.2 * w):max(int(0.2 * w) + 2, int(0.8 * w))], S, S)
    low = _rs(c[int(0.55 * h):, :], S, S // 2)
    # carpet colour: median Lab of a ring around the box (mostly floor), robot = far from it
    mx, my = max(4, w // 4), max(4, h // 4)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    ring = frame[Y0:Y1, X0:X1].copy()
    rmask = np.ones(ring.shape[:2], bool)
    rmask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    lab_ring = cv2.cvtColor(ring, cv2.COLOR_BGR2LAB)[rmask].astype(np.float32)
    lab = cv2.cvtColor(boxi, cv2.COLOR_BGR2LAB).astype(np.float32)
    if len(lab_ring) > 20:
        ref = np.median(lab_ring, axis=0)
        carpet = np.linalg.norm(lab - ref, axis=2) < 18.0
    else:
        carpet = np.zeros((S, S), bool)
    robot = ~carpet & (yellow_mask(boxi) == 0)
    fg = (_patch_frac(robot) >= 0.4).astype(np.float32)
    if fg.sum() < 8:
        fg = INNER
    return boxi, ctr, low, fg


def _forward(model, imgs: list[np.ndarray], dev: str):
    import torch
    x = torch.from_numpy(np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs])).to(dev)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std
    if dev == "cuda":
        x = x.half()
    f = model.forward_features(x)
    return f["x_norm_clstoken"].float(), f["x_norm_patchtokens"].float()


def _pool(tok, m: np.ndarray):
    import torch
    mt = torch.from_numpy(np.asarray(m, np.float32)).to(tok.device)
    if mt.ndim == 1:
        mt = mt.unsqueeze(0).expand(tok.shape[0], -1)
    return (mt.unsqueeze(2) * tok).sum(1) / mt.sum(1, keepdim=True).clamp(min=1.0)


def embed_batch(model, items: list[tuple], dev: str, batch: int) -> dict[str, np.ndarray]:
    import torch
    out = {k: [] for k in FEATS}
    with torch.no_grad():
        for i in range(0, len(items), batch):
            part = items[i:i + batch]
            cls, tok = _forward(model, [p[0] for p in part], dev)
            fg = np.stack([p[3] for p in part])
            out["box_cls"].append(cls)
            out["box_mean"].append(tok.mean(1))
            out["box_inner"].append(_pool(tok, INNER))
            out["box_fg"].append(_pool(tok, fg))
            out["box_low"].append(_pool(tok, LOWER))
            cls, tok = _forward(model, [p[1] for p in part], dev)
            out["ctr_cls"].append(cls)
            out["ctr_mean"].append(tok.mean(1))
            cls, tok = _forward(model, [p[2] for p in part], dev)
            out["low_cls"].append(cls)
            out["low_mean"].append(tok.mean(1))
    return {k: torch.cat(v).cpu().numpy().astype(np.float16) for k, v in out.items()}


def embed_match(key: str, model, dev: str, batch: int, chunk: int = 512) -> str:
    from autoscout.video import Video, video_path
    out = EMB / f"{key}.npz"
    if out.exists():
        return "have"
    with open(ROOT / "output" / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    want = [(d.t, d.crop_id, tuple(d.box), d.view) for d in md.dets if d.crop_id is not None]
    bh = {cid: c[2] for cid, c in md.crops.items()}
    del md
    if not want:
        return "no crops"
    v = Video(video_path(key))
    by_idx: dict[int, list] = {}
    for t, cid, box, view in want:
        by_idx.setdefault(int(round(t * v.fps)), []).append((t, cid, box, view))
    idxs = sorted(by_idx)
    ids, ts, vws, hs, feats = [], [], [], [], {k: [] for k in FEATS}
    pend: list = []

    def flush():
        if pend:
            r = embed_batch(model, [p[1] for p in pend], dev, batch)
            for k in FEATS:
                feats[k].append(r[k])
            for meta, _ in pend:
                ids.append(meta[0]); ts.append(meta[1]); vws.append(meta[2]); hs.append(meta[3])
            pend.clear()

    cap = v.cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
    need = set(idxs)
    idx = idxs[0]
    while idx <= idxs[-1]:
        if idx in need:
            ok, frame = cap.read()
            if not ok:
                break
            for t, cid, box, view in by_idx[idx]:
                cv_ = crop_views(frame, box)
                if cv_ is not None:
                    pend.append(((cid, t, VIEWS.index(view) if view in VIEWS else -1, bh.get(cid, 0)), cv_))
            if len(pend) >= chunk:
                flush()
        else:
            if not cap.grab():
                break
        idx += 1
    flush()
    v.close()
    if not ids:
        return "no frames"
    EMB.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_tmp.npz")
    np.savez(tmp, ids=np.array(ids, np.int32), t=np.array(ts, np.float32), view=np.array(vws, np.int8),
             bh=np.array(hs, np.int16), **{k: np.concatenate(feats[k]) for k in FEATS})
    os.replace(tmp, out)
    return f"{len(ids)}/{len(want)} crops"


def cmd_embed(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    model = _model(dev)
    keys = a.matches or sorted(p.parent.name for p in (ROOT / "output").glob("*/scan.pkl"))
    for k in keys:
        if (EMB / f"{k}.npz").exists():
            continue
        wait_mem(a.min_free)
        tic = time.time()
        try:
            msg = embed_match(k, model, dev, a.batch)
        except Exception as e:  # keep going; a rerun retries the failures
            msg = f"FAIL {e!r}"
        print(f"{time.strftime('%H:%M')} {k}: {msg} ({time.time() - tic:.0f}s)", flush=True)


# ---------------------------------------------------------------- labels
def labels_match(key: str) -> str:
    """Per crop: segment, segment alliance, assigned team and the segment's bumper-OCR evidence, from the
    finalized output/<match>/result.json + dets.json (default Options) and the scan's crop ids."""
    import json
    od = ROOT / "output" / key
    with open(od / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    res = json.loads((od / "result.json").read_text())
    rows = json.loads((od / "dets.json").read_text())
    if len(rows) != len(md.dets):
        return f"FAIL dets.json has {len(rows)} rows, scan {len(md.dets)} dets"
    info = md.info
    teams6 = list(info.red) + list(info.blue)
    seg_al, seg_ocr, seg_team = {}, {}, {}
    for ai, al in enumerate(("red", "blue")):
        for sid, s in res["identity"][al].items():
            seg_al[int(sid)] = ai
            seg_ocr[int(sid)] = s["ocr"]
            seg_team[int(sid)] = s["team"] if s["team"] is not None else -1
    ids, seg, al, team, ocr_ev = [], [], [], [], []
    for d, r in zip(md.dets, rows):
        if d.crop_id is None:
            continue
        sid = r[8]
        ids.append(d.crop_id)
        if sid is None or int(sid) not in seg_al:
            seg.append(-1); al.append(-1); team.append(-1); ocr_ev.append([0.0, 0.0, 0.0])
        else:
            sid = int(sid)
            seg.append(sid); al.append(seg_al[sid]); team.append(seg_team[sid]); ocr_ev.append(seg_ocr[sid])
    # per-segment mean ResNet-18 crop embedding (the vector pipeline.analyse uses for appearance), for a
    # leave-one-match-out ResNet gallery baseline
    cnn = getattr(md, "cnn_cache", None) or {}
    seg_ids = sorted({s for s in seg if s >= 0})
    seg_res = np.zeros((len(seg_ids), 512), np.float32)
    seg_n = np.zeros(len(seg_ids), np.int32)
    pos = {s: i for i, s in enumerate(seg_ids)}
    for cid, s in zip(ids, seg):
        if s >= 0 and cid in cnn:
            seg_res[pos[s]] += cnn[cid]
            seg_n[pos[s]] += 1
    seg_res /= np.maximum(seg_n, 1)[:, None]
    LAB.mkdir(parents=True, exist_ok=True)
    np.savez(LAB / f"{key}.npz", ids=np.array(ids, np.int32), seg=np.array(seg, np.int32),
             al=np.array(al, np.int8), team=np.array(team, np.int32), ocr=np.array(ocr_ev, np.float32),
             teams=np.array(teams6, np.int32), event=np.array(info.event_key),
             seg_ids=np.array(seg_ids, np.int32), seg_res=seg_res.astype(np.float16), seg_n=seg_n)
    return f"{len(ids)} crops, {len(seg_ids)} segments"


def cmd_labels(a):
    keys = a.matches or sorted(p.parent.name for p in (ROOT / "output").glob("*/scan.pkl"))
    for k in keys:
        if (LAB / f"{k}.npz").exists() and not a.force:
            continue
        try:
            msg = labels_match(k)
        except Exception as e:
            msg = f"FAIL {e!r}"
        print(f"{k}: {msg}", flush=True)


# ---------------------------------------------------------------- probe
def confident(ocr: np.ndarray, team_idx: np.ndarray, min_mass: float = 2.0, purity: float = 0.8) -> np.ndarray:
    """Segments whose bumper reads are strong and clear and agree with the assigned team (build_galleries rule)."""
    s = ocr.sum(1)
    return (s >= min_mass) & (ocr.max(1) >= purity * s) & (ocr.argmax(1) == team_idx)


def load_match(key: str, feats: list[str]) -> dict | None:
    """Embeddings + labels of one match, aligned by crop id. X: L2-normalised features (concatenated)."""
    pe, pl = EMB / f"{key}.npz", LAB / f"{key}.npz"
    if not (pe.exists() and pl.exists()):
        return None
    E, L = np.load(pe), np.load(pl)
    parts = []
    for f in feats:
        x = E[f].astype(np.float32)
        parts.append(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6))
    X = np.concatenate(parts, axis=1)
    pos = {int(c): i for i, c in enumerate(L["ids"])}
    j = np.array([pos.get(int(c), -1) for c in E["ids"]])
    ok = j >= 0
    j = j[ok]
    teams6 = L["teams"]
    al, seg, team, ocr = L["al"][j], L["seg"][j], L["team"][j], L["ocr"][j]
    # assigned team as an index 0..2 within the segment's alliance (-1: none)
    tidx = np.full(len(j), -1)
    for a_ in (0, 1):
        tri = teams6[3 * a_:3 * a_ + 3]
        m = al == a_
        tidx[m] = np.array([list(tri).index(t) if t in tri else -1 for t in team[m]])
    conf = confident(ocr, tidx) & (al >= 0)
    return {"key": key, "event": str(L["event"]), "ids": E["ids"][ok], "X": X[ok], "view": E["view"][ok],
            "bh": E["bh"][ok], "al": al, "seg": seg, "team": team, "tidx": tidx, "ocr": ocr, "conf": conf,
            "teams6": teams6}


def crop_weights(seg: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1/sqrt(crops in the segment) (long segments do not dominate), then every class gets equal total weight."""
    w = np.zeros(len(seg), np.float64)
    _, inv, cnt = np.unique(seg, return_inverse=True, return_counts=True)
    w = 1.0 / np.sqrt(cnt[inv])
    for c in np.unique(labels):
        m = labels == c
        w[m] /= w[m].sum()
    return w / w.sum() * len(w)


def train_probe(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_cls: int, l2: float, dev: str, iters: int = 200):
    """Multinomial logistic regression (full-batch L-BFGS, torch) on standardised features."""
    import torch
    Xt = torch.from_numpy(X).to(dev)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-4
    Xt = (Xt - mu) / sd
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev)
    W = torch.zeros(X.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ W + b
        loss = (wt * torch.nn.functional.cross_entropy(z, yt, reduction="none")).mean() + 0.5 * l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (W.detach(), b.detach(), mu, sd)


def apply_probe(model, X: np.ndarray, dev: str) -> np.ndarray:
    import torch
    W, b, mu, sd = model
    with torch.no_grad():
        z = ((torch.from_numpy(X).to(dev) - mu) / sd) @ W + b
        return torch.softmax(z, 1).cpu().numpy()


FLOOR = 0.02  # per-crop probability floor: one odd crop cannot veto a team


def alliance_logq(P: np.ndarray, classes: list[int], teams3) -> np.ndarray:
    """(n, C) event-team probabilities -> (n, 3) log-probabilities over the alliance's 3 teams.

    Teams the probe never saw (no confident crops in the event's other matches) share the probability
    mass the probe gives to "none of the alliance's known teams". NaN when none of the 3 is known."""
    col = {t: i for i, t in enumerate(classes)}
    known = [col.get(int(t), -1) for t in teams3]
    k = sum(c >= 0 for c in known)
    n = len(P)
    if k == 0:
        return np.full((n, 3), np.nan, np.float32)
    q = np.zeros((n, 3))
    pk = np.stack([P[:, c] if c >= 0 else np.zeros(n) for c in known], 1)
    if k == 3:
        q = pk
    else:
        rest = np.clip(1.0 - pk.sum(1), 1e-6, None) / (3 - k)
        q = np.where(np.array(known)[None, :] >= 0, pk, rest[:, None])
    q = q / q.sum(1, keepdims=True)
    q = np.clip(q, FLOOR, None)
    q = q / q.sum(1, keepdims=True)
    return np.log(q).astype(np.float32)


def train_for(train: list[dict], policy: str, l2: float, dev: str, min_view_h: int = 0):
    """Probe over the event teams seen in `train` (other matches only)."""
    Xs, ys, segs = [], [], []
    for mi, m in enumerate(train):
        if policy == "ocr":
            sel = m["conf"]
        elif policy == "assigned":
            sel = (m["team"] >= 0) & (m["al"] >= 0)
        else:
            raise ValueError(policy)
        sel = sel & (m["bh"] >= min_view_h)
        if not sel.any():
            continue
        Xs.append(m["X"][sel])
        ys.append(m["team"][sel])
        segs.append(mi * 100000 + m["seg"][sel])
    if not Xs:
        return None, []
    y_team = np.concatenate(ys)
    classes = sorted(set(y_team.tolist()))
    if len(classes) < 2:
        return None, classes
    cidx = {t: i for i, t in enumerate(classes)}
    y = np.array([cidx[t] for t in y_team])
    w = crop_weights(np.concatenate(segs), y)
    return train_probe(np.concatenate(Xs), y, w, len(classes), l2, dev), classes


def probe_logq(model, classes, m: dict, dev: str) -> np.ndarray:
    """(n, 6) per crop: log q over the red teams (cols 0-2) and the blue teams (cols 3-5)."""
    n = len(m["X"])
    if model is None:
        return np.full((n, 6), np.nan, np.float32)
    P = apply_probe(model, m["X"], dev)
    return np.concatenate([alliance_logq(P, classes, m["teams6"][:3]), alliance_logq(P, classes, m["teams6"][3:])], 1)


def event_matches() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted(EMB.glob("*.npz")):
        if p.stem.endswith("_tmp") or not (LAB / p.name).exists():
            continue
        out.setdefault(p.stem.split("_")[0], []).append(p.stem)
    return out


class Pool:
    """OCR-confident crops of every match (all events), for leave-one-match-out probes trained across events:
    a team's robot looks the same at its other events, which helps the events of which we only have one
    team's schedule (most teams there appear in a single match)."""

    def __init__(self, feats: list[str], keys: list[str]):
        Xs, ys, segs, mids = [], [], [], []
        self.keys = keys
        for mi, k in enumerate(keys):
            m = load_match(k, feats)
            if m is None:
                continue
            sel = m["conf"]
            Xs.append(m["X"][sel]); ys.append(m["team"][sel]); segs.append(mi * 100000 + m["seg"][sel])
            mids.append(np.full(int(sel.sum()), mi))
        self.X, self.team, self.seg, self.mid = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(segs),
                                                 np.concatenate(mids))

    def train(self, hold: str, l2: float, dev: str):
        keep = self.mid != self.keys.index(hold)
        y_team = self.team[keep]
        classes = sorted(set(y_team.tolist()))
        cidx = {t: i for i, t in enumerate(classes)}
        y = np.array([cidx[t] for t in y_team])
        w = crop_weights(self.seg[keep], y)
        return train_probe(self.X[keep], y, w, len(classes), l2, dev), classes


def cmd_probe(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feats = a.feat.split("+")
    name = a.name or f"{a.feat}_{a.policy}" + ("_pool" if a.pool else "") + ("_causal" if a.causal else "")
    od = PROBE / name
    od.mkdir(parents=True, exist_ok=True)
    evm = event_matches()
    pool = Pool(feats, [k for ks in evm.values() for k in ks]) if a.pool else None
    for ev, keys in evm.items():
        if a.events and ev not in a.events:
            continue
        if a.causal:  # no hindsight: in schedule order, each match's probe sees only the matches played before it
            from autoscout import opr, tba
            keys = sorted(keys, key=lambda k: opr.match_order(tba.match(k)))
        data = [m for m in (load_match(k, feats) for k in keys) if m is not None]
        for i, m in enumerate(data):
            if pool is not None:
                model, classes = pool.train(m["key"], a.l2, dev)
            elif a.causal:
                model, classes = train_for(data[:i], a.policy, a.l2, dev)
            else:
                model, classes = train_for(data[:i] + data[i + 1:], a.policy, a.l2, dev)
            lq = probe_logq(model, classes, m, dev)
            np.savez(od / f"{m['key']}.npz", ids=m["ids"].astype(np.int32), teams=m["teams6"].astype(np.int32),
                     logq=lq.astype(np.float16))
        print(f"{ev}: {len(data)} matches -> {od}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--matches", nargs="*", default=[])
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--min_free", type=float, default=6.0)
    lb = sub.add_parser("labels")
    lb.add_argument("--matches", nargs="*", default=[])
    lb.add_argument("--force", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--feat", default="box_inner", help="feature name(s) joined by +, e.g. box_cls+box_fg")
    p.add_argument("--policy", default="ocr", choices=["ocr", "assigned"])
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--events", nargs="*", default=[])
    p.add_argument("--name", default=None, help="output folder under data/exp_dino/probe/")
    p.add_argument("--pool", action="store_true", help="train on every other match of every event (ocr labels)")
    p.add_argument("--causal", action="store_true",
                   help="no hindsight: train each match's probe only on the event's earlier matches (live use)")
    a = ap.parse_args()
    {"embed": cmd_embed, "labels": cmd_labels, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
