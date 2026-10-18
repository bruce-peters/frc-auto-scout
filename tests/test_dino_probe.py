import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_dino_probe as B  # noqa: E402
from autoscout import pipeline  # noqa: E402


def test_alliance_logq_all_known_renormalises_over_the_three():
    classes = [10, 20, 30, 40]
    P = np.array([[0.6, 0.2, 0.1, 0.1]])
    lq = B.alliance_logq(P, classes, [10, 20, 30])
    q = np.exp(lq[0])
    assert abs(q.sum() - 1) < 1e-5 and q.argmax() == 0
    assert abs(q[0] / q[1] - 3.0) < 1e-3  # ratios kept


def test_alliance_logq_unknown_teams_share_the_rest_and_floor():
    classes = [10, 40, 50]
    # a crop that looks like none of the alliance's known teams: the unknown ones get the mass
    P = np.array([[0.05, 0.9, 0.05], [0.999, 0.0005, 0.0005]])
    lq = B.alliance_logq(P, classes, [10, 20, 30])
    q = np.exp(lq)
    assert q[0, 0] < q[0, 1] and abs(q[0, 1] - q[0, 2]) < 1e-6
    assert q[1].argmax() == 0 and q[1].min() >= B.FLOOR * 0.9  # floor: one crop cannot veto a team
    assert np.isnan(B.alliance_logq(P, classes, [1, 2, 3])).all()  # nothing known -> no evidence


def test_crop_weights_balance_classes_and_long_segments():
    seg = np.array([0, 0, 0, 0, 1, 2])
    y = np.array([0, 0, 0, 0, 0, 1])
    w = B.crop_weights(seg, y)
    assert abs(w[y == 0].sum() - w[y == 1].sum()) < 1e-9
    assert w[4] > w[0]  # a crop of a short segment counts more than one of a long segment


def test_load_dino_probe_maps_crops_to_teams(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DINO_PROBE_DIR", tmp_path)
    lq = np.log(np.array([[0.7, 0.2, 0.1, 0.3, 0.3, 0.4]], np.float32))
    np.savez(tmp_path / "2026xx_qm1.npz", ids=np.array([5]), teams=np.array([1, 2, 3, 4, 5, 6]), logq=lq.astype(np.float16))
    d = pipeline.load_dino_probe("2026xx_qm1")
    assert set(d) == {5} and abs(np.exp(d[5][1]) - 0.7) < 1e-2 and abs(np.exp(d[5][6]) - 0.4) < 1e-2
    assert pipeline.load_dino_probe("2026xx_qm2") == {}
    assert pipeline.Options().w_dino == 2.0  # on by default (with the absent state)
