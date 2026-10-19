"""Assign team numbers to tracker segments.

Per alliance there are 3 tracker slots and 3 teams. Time is cut into *epochs* at every
segment boundary; within an epoch each slot holds one segment (or nothing). The hidden state
is the permutation slot -> team (6 states). Emission = evidence that each slot's segment is
that team (bumper OCR + appearance similarity); transitions may only re-assign slots whose
segment ended at the epoch boundary, with a small penalty, so identities persist through
continuous tracking and get re-decided after ambiguity (close passes, re-acquisition).
Viterbi gives the best labelling.
"""
from __future__ import annotations

import itertools

import numpy as np

from .robots import Segment

PERMS = list(itertools.permutations(range(3)))


def _epochs(segs: list[Segment]) -> list[tuple[float, float, dict[int, Segment]]]:
    """Split time at segment starts; return [(t0, t1, {slot: segment})]."""
    bounds = sorted({s.t0 for s in segs} | {max(s.t1, s.t0) + 1e-3 for s in segs})
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mid = (a + b) / 2
        active = {}
        for s in segs:
            if s.t0 <= mid <= s.t1 + 1e-3:
                # one segment per slot at a time by construction
                active[s.slot] = s
        if active:
            out.append((a, b, active))
    return out


def assign(segs: list[Segment], evidence: dict[int, np.ndarray], switch_cost: float = 0.6
           ) -> dict[int, int]:
    """Return {segment id: team index 0..2} for one alliance.

    evidence[seg.id] is a length-3 score vector (higher = more likely that team).
    """
    eps = _epochs(segs)
    if not eps:
        return {}
    n = len(eps)
    S = len(PERMS)
    score = np.full((n, S), -1e18)
    back = np.zeros((n, S), dtype=int)

    def emit(active: dict[int, Segment], perm) -> float:
        tot = 0.0
        for slot, seg in active.items():
            ev = evidence.get(seg.id)
            if ev is not None:
                tot += float(ev[perm[slot]])
        return tot

    for k in range(S):
        score[0, k] = emit(eps[0][2], PERMS[k])
    for i in range(1, n):
        prev_active = eps[i - 1][2]
        cur_active = eps[i][2]
        # slots whose segment continues across the boundary must keep their team
        cont = [sl for sl, sg in cur_active.items() if sl in prev_active and prev_active[sl].id == sg.id]
        for k in range(S):
            e = emit(cur_active, PERMS[k])
            best, arg = -1e18, 0
            for j in range(S):
                if any(PERMS[j][sl] != PERMS[k][sl] for sl in cont):
                    continue
                c = score[i - 1, j] - (0.0 if j == k else switch_cost)
                if c > best:
                    best, arg = c, j
            score[i, k] = best + e
            back[i, k] = arg
    k = int(np.argmax(score[-1]))
    path = [k]
    for i in range(n - 1, 0, -1):
        k = back[i, k]
        path.append(k)
    path.reverse()
    labels: dict[int, list[int]] = {}
    for (a, b, active), k in zip(eps, path):
        for slot, seg in active.items():
            labels.setdefault(seg.id, []).append(PERMS[k][slot])
    # a segment spanning several epochs keeps one label (continuity constraint) - take mode
    return {sid: max(set(v), key=v.count) for sid, v in labels.items()}


def appearance_vector(crop: np.ndarray) -> np.ndarray | None:
    """HSV colour histogram of the robot body (top 55 % of the box, above the bumpers)."""
    import cv2

    if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None
    body = crop[: max(4, int(crop.shape[0] * 0.55))]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = np.sqrt(hist / (hist.sum() + 1e-9))  # Hellinger
    return hist.astype(np.float32)


class CNNEmbedder:
    """ImageNet ResNet-18 features of robot crops (instance-level appearance, view tolerant)."""

    def __init__(self):
        import torch
        import torchvision

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, crops: list, batch: int = 128) -> np.ndarray:
        import cv2

        out = []
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(crops), batch):
                arr = np.stack([cv2.cvtColor(cv2.resize(c, (112, 112)), cv2.COLOR_BGR2RGB) for c in crops[i:i + batch]])
                x = torch.from_numpy(arr).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
                x = (x - self.mean) / self.std
                f = self.model(x)
                f = torch.nn.functional.normalize(f, dim=1)
                out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def train_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1.0,
                  sample_w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression (L-BFGS) -> (W [d, k], b [k])."""
    from scipy.optimize import minimize

    n, dim = X.shape
    Y = np.eye(n_classes)[y]
    sw = np.ones(n) if sample_w is None else sample_w
    sw = sw / sw.sum()

    def f(theta):
        W = theta[: dim * n_classes].reshape(dim, n_classes)
        b = theta[dim * n_classes:]
        Z = X @ W + b
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -(sw * (Y * np.log(P + 1e-12)).sum(1)).sum() + 0.5 * l2 * (W ** 2).sum()
        G = (P - Y) * sw[:, None]
        gW = X.T @ G + l2 * W
        gb = G.sum(0)
        return loss, np.concatenate([gW.ravel(), gb])

    res = minimize(f, np.zeros(dim * n_classes + n_classes), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200})
    th = res.x
    return th[: dim * n_classes].reshape(dim, n_classes), th[dim * n_classes:]


def classifier_evidence(seg_feats: dict[int, np.ndarray], ocr_ev: dict[int, np.ndarray], min_mass: float = 2.0,
                        purity: float = 0.8, l2: float = 1.0) -> dict[int, np.ndarray]:
    """Per-match appearance classifier for one alliance's 3 robots.

    Trains on the crops of segments whose bumper-OCR evidence is strong and clear, then returns,
    for every segment, centred mean log-probabilities over its crops (classes without any
    training segment get a neutral score). Empty dict when there is too little to train on.
    """
    Xs, ys, ws = [], [], []
    for sid, F in seg_feats.items():
        e = ocr_ev.get(sid)
        if e is None or e.sum() < min_mass or e.max() < purity * e.sum():
            continue
        Xs.append(F)
        ys += [int(np.argmax(e))] * len(F)
        ws += [1.0 / np.sqrt(len(F))] * len(F)  # long segments should not dominate
    if not Xs:
        return {}
    X, y, w = np.concatenate(Xs).astype(np.float64), np.array(ys), np.array(ws)
    present = sorted(set(y.tolist()))
    if len(present) < 2:
        return {}
    remap = {c: i for i, c in enumerate(present)}
    W, b = train_softmax(X, np.array([remap[c] for c in y]), len(present), l2=l2, sample_w=w)
    out = {}
    for sid, F in seg_feats.items():
        Z = F.astype(np.float64) @ W + b
        Z = Z - Z.max(1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        m = logp.mean(0)
        full = np.full(3, np.nan)
        for c, i in remap.items():
            full[c] = m[i]
        full = np.where(np.isfinite(full), full, np.nanmean(full))
        out[sid] = (full - full.mean()) * min(1.0, len(F) / 3.0)
    return out



def classifier_evidence_by_view(seg_view_feats: dict[str, dict[int, np.ndarray]], ocr_ev: dict[int, np.ndarray],
                                min_mass: float = 2.0, purity: float = 0.8, l2: float = 1.0
                                ) -> dict[int, np.ndarray]:
    """Like classifier_evidence, but one classifier per camera view.

    Bumper OCR mostly succeeds in the close end cameras, so a single classifier over all views
    learns "which camera" rather than "which robot". Training and scoring within each view keeps
    the comparison like-for-like; evidence from the views a segment appears in is summed.
    """
    out: dict[int, np.ndarray] = {}
    for view, feats in seg_view_feats.items():
        ev_v = classifier_evidence(feats, ocr_ev, min_mass=min_mass, purity=purity, l2=l2)
        for sid, e in ev_v.items():
            out[sid] = out.get(sid, np.zeros(3)) + e
    return out


# ---------------- absent / unknown state (data/exp_identity/RESULTS_section.md; off by default)
# assign() gives every team of the alliance a slot whenever three slots are active, so the label of a dead or
# absent robot that is never read lands on a teammate's duplicate track (2026cc_qm18: 3045's label on boxes
# showing 6647) and takes that robot's launches. assign_absent() lets a team be ABSENT: the hidden state is any
# map slot -> team (27 states). A team that labels no slot, active or inactive, is absent in that epoch and costs
# absent_cost[team]; two slots sharing a label is the duplicate-track case. A team on an inactive slot is hidden,
# as in assign(), at no cost.
STATES27 = list(itertools.product(range(3), repeat=3))
_M27 = np.array(STATES27)
_USED27 = np.array([[t in s for t in range(3)] for s in STATES27])  # state -> team labels some slot
_PAIRS = ((0, 1), (0, 2), (1, 2))
_SAME27 = {p: (_M27[:, p[0]] == _M27[:, p[1]]).astype(float) for p in _PAIRS}


def _mean_dist(sa: Segment, sb: Segment, a: float, b: float) -> float:
    """Mean distance (m) between two segments' tracked positions over [a, b] (its first 5 s)."""
    ts = np.linspace(a, min(b, a + 5.0), 3)
    pa, pb = np.asarray(sa.points, float), np.asarray(sb.points, float)
    ta, tb = np.clip(ts, sa.t0, sa.t1), np.clip(ts, sb.t0, sb.t1)
    dx = np.interp(ta, pa[:, 0], pa[:, 1]) - np.interp(tb, pb[:, 0], pb[:, 1])
    dy = np.interp(ta, pa[:, 0], pa[:, 2]) - np.interp(tb, pb[:, 0], pb[:, 2])
    return float(np.mean(np.hypot(dx, dy)))


def assign_absent(segs: list[Segment], evidence: dict[int, np.ndarray], switch_cost: float = 0.6,
                  absent_cost=(np.inf, np.inf, np.inf), near_m: float = 0.0, near_bonus: float = 0.0,
                  far_m: float = 0.0, far_cost: float = 0.0
                  ) -> tuple[dict[int, int], dict[int, list[tuple[float, float]]]]:
    """Like assign(), but a team may have no segment (absent) at absent_cost[team] per epoch (inf = never).

    near_bonus is added per epoch to states that give two concurrent segments less than near_m apart the
    same label (co-located tracks are one robot seen twice). far_cost is subtracted per epoch from states that
    give two concurrent segments more than far_m apart the same label: one robot cannot be in two places, so by
    elimination the other track is a team without one. Returns ({segment id: team index},
    {team index: [(t0, t1), ...] stretches in which the team is absent}).
    """
    eps = _epochs(segs)
    if not eps:
        return {}, {0: [], 1: [], 2: []}
    n, S = len(eps), len(STATES27)
    ac = np.array([c if np.isfinite(c) else 1e9 for c in absent_cost], float)
    base = -np.where(_USED27, 0.0, ac[None, :]).sum(1)
    score = np.full((n, S), -1e18)
    back = np.zeros((n, S), dtype=int)
    stay = np.where(np.eye(S, dtype=bool), 0.0, switch_cost)

    def emit(a: float, b: float, active: dict[int, Segment]) -> np.ndarray:
        e = base.copy()
        for slot, seg in active.items():
            ev = evidence.get(seg.id)
            if ev is not None:
                e += np.asarray(ev, float)[_M27[:, slot]]
        if (near_m > 0 and near_bonus) or (far_m > 0 and far_cost):
            for p in _PAIRS:
                if p[0] in active and p[1] in active:
                    dm = _mean_dist(active[p[0]], active[p[1]], a, b)
                    if near_m > 0 and near_bonus and dm < near_m:
                        e += near_bonus * _SAME27[p]
                    elif far_m > 0 and far_cost and dm > far_m:
                        e -= far_cost * _SAME27[p]
        return e

    score[0] = emit(*eps[0])
    for i in range(1, n):
        prev_active, cur_active = eps[i - 1][2], eps[i][2]
        cont = [sl for sl, sg in cur_active.items() if sl in prev_active and prev_active[sl].id == sg.id]
        ok = np.ones((S, S), bool)
        for sl in cont:  # slots whose segment continues must keep their team
            ok &= _M27[:, None, sl] == _M27[None, :, sl]
        c = np.where(ok, score[i - 1][:, None] - stay, -1e30)
        back[i] = np.argmax(c, axis=0)
        score[i] = c[back[i], np.arange(S)] + emit(*eps[i])
    k = int(np.argmax(score[-1]))
    path = [k]
    for i in range(n - 1, 0, -1):
        k = back[i, k]
        path.append(k)
    path.reverse()
    labels: dict[int, list[int]] = {}
    absent: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
    for (a, b, active), k in zip(eps, path):
        for slot, seg in active.items():
            labels.setdefault(seg.id, []).append(STATES27[k][slot])
        for t in range(3):
            if not _USED27[k, t]:
                if absent[t] and abs(absent[t][-1][1] - a) < 1e-6:
                    absent[t][-1] = (absent[t][-1][0], b)
                else:
                    absent[t].append((a, b))
    return {sid: max(set(v), key=v.count) for sid, v in labels.items()}, absent


def dead_teams(segs: list[Segment], ocr_ev: dict[int, np.ndarray], labels: dict[int, int],
               own_max: float = 0.5, mate_min: float = 3.0, foreign_min: float = 1.0) -> np.ndarray:
    """Dead-robot evidence per team index: < own_max bumper evidence over all the alliance's segments while
    every teammate has >= mate_min, and the segments labelled with it carry >= foreign_min of teammates' reads."""
    own = np.zeros(3)
    for s in segs:
        own += ocr_ev.get(s.id, np.zeros(3))
    out = np.zeros(3, bool)
    for t in range(3):
        mates = [u for u in range(3) if u != t]
        if own[t] < own_max and all(own[u] >= mate_min for u in mates):
            foreign = sum(float(ocr_ev[sid][u]) for sid, lab in labels.items() if lab == t and sid in ocr_ev
                          for u in mates)
            out[t] = foreign >= foreign_min
    return out
