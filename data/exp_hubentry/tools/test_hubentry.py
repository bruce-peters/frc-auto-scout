"""Unit tests for autoscout/hubentry.py on synthetic data (no scans needed). Move to tests/ when porting.

  .venv/Scripts/python -m pytest data/exp_hubentry/tools/test_hubentry.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from autoscout import attribution as A, hubentry as H  # noqa: E402


def _teams():
    out = []
    rng = np.random.default_rng(1)
    for i, team in enumerate((1, 2, 3)):
        ts = A.TeamSignals(team)
        ts.shots = np.sort(rng.uniform(10, 60, 30 - 8 * i))
        ts.shot_w = rng.uniform(0.3, 1.0, len(ts.shots))
        ts.zone_t = np.arange(0, 80, 0.5)
        ts.zone_v = (rng.random(len(ts.zone_t)) < 0.8).astype(float)
        ts.vis_t, ts.vis_v = ts.zone_t, (rng.random(len(ts.zone_t)) < 0.9).astype(float)
        out.append(ts)
    return out


def _inc():
    return [(12.0 + 0.334 * k, float(1 + k % 3)) for k in range(40)] + [(40.0 + 0.334 * k, 2.0) for k in range(10)]


def test_fallback_reproduces_split_increments():
    teams, inc = _teams(), _inc()
    for model in (None, A.load_model()):
        keep = A.ATTR_MODEL
        A.ATTR_MODEL = None if model is None else keep
        try:
            l1, l2 = [], []
            a = A.split_increments(inc, teams, lambda t: t - 5.0, log=l1)
            ents = [H.Retimed(t, t, n, 0.0) for t, n in inc]
            b = H.split_entries(inc, teams, lambda t: t - 5.0, ents, log=l2, model=model)
        finally:
            A.ATTR_MODEL = keep
        for tm in a:
            for p in a[tm]:
                assert abs(a[tm][p] - b[tm][p]) < 1e-9
        assert l1 == l2


def test_retime_queue_keeps_count_order_and_bounds():
    # a hopper dump: 60 balls drop in around 10.8 s within ~1 s, the hub counts them at ~12/s from 12.0 s
    inc = [(12.0 + 0.334 * k, 4.0) for k in range(15)] + [(30.0 + 0.334 * k, 2.0) for k in range(5)]
    t = np.arange(0, 50, 0.1)
    rate = np.exp(-0.5 * ((t - 10.8) / 0.4) ** 2) * 5 + np.exp(-0.5 * ((t - 28.6) / 0.4) ** 2) * 2 + 0.02
    ents = H.retime_queue(inc, t, rate)
    assert abs(sum(e.n for e in ents) - sum(n for _, n in inc)) < 1e-9
    et = [e.entry_t for e in ents]
    assert all(b >= a - 1e-9 for a, b in zip(et, et[1:]))  # first in, first out
    for e in ents:
        assert 0.6 - 1e-9 <= e.count_t - e.entry_t <= H.D_MAX + H.READ_GAP + 1e-9
    dump = [e for e in ents if e.count_t < 20]
    assert abs(np.average([e.entry_t for e in dump], weights=[e.n for e in dump]) - 10.8) < 0.5
    delay = [e.count_t - e.entry_t for e in dump]
    assert delay[-1] - delay[0] > 2.0  # the queue: the last ball of the dump waits ~3 s longer than the first


def test_retime_queue_falls_back_without_evidence():
    inc = _inc()
    t = np.arange(0, 70, 0.1)
    ents = H.retime_queue(inc, t, np.zeros(len(t)))
    assert all(e.conf == 0.0 for e in ents)
    assert all(abs(e.count_t - H.READ_GAP / 2 - 1.35 - e.entry_t) < 1e-9 for e in ents)


def test_entry_kernel_area():
    dt = np.linspace(-1, 3, 40001)
    k = H.entry_kernel(dt)
    assert abs(np.trapezoid(k, dt) - H.KERNEL_AREA) < 1e-3
    assert H.entry_kernel(np.array([H.FLIGHT_LO - 0.01, H.FLIGHT_HI + 0.01])).sum() == 0
