import json
import os
import pickle

import numpy as np
import pytest

from autoscout import config, fuel, pipeline, tracks60


def _log():
    """A small blob log: two balls rising across three views' worth of frames at 60 Hz."""
    rows = []
    for i in range(20):
        t = 10.0 + i / 60
        mb = np.array([(500 + 3 * i, 400 - 9 * i + 0.25 * i * i, 30, 6, 6, 497 + 3 * i, 397 - 9 * i),
                       (800 - 2 * i, 420 - 8 * i + 0.25 * i * i, 30, 6, 6, 797 - 2 * i, 417 - 8 * i)], float)
        rows.append((t, mb, np.zeros((0, 5), np.float32)))
    return {"fuel_step": 1, "blob_log": {"main": rows, "blue_end": rows[:5], "red_end": []}}


def test_tracks_from_log_is_the_greedy_tracker_on_the_logged_blobs():
    log = _log()
    out = tracks60.tracks_from_log(log)
    ref = fuel.FlightTracker(gate=45.0)
    for t, mb, _ in log["blob_log"]["main"]:
        ref.update(t, [fuel.Blob(*map(float, r)) for r in mb])
    ref = ref.flush()
    assert set(out) == {"main", "blue_end", "red_end"}
    assert len(out["main"]) == len(ref) == 2
    assert all(np.allclose(np.array(a.pts), np.array(b.pts)) for a, b in zip(out["main"], ref))
    assert out["red_end"] == []


def test_load_fuel60_keeps_30hz_tracks_without_a_log(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BLOBS60", tmp_path)
    monkeypatch.setattr(pipeline, "FUEL_60HZ", True)

    class MD:
        key = "2026xx_qm1"
        tracks_all = {"main": ["stored"]}
    md = MD()
    assert pipeline.load_fuel60(md) == 30
    assert md.tracks_all == {"main": ["stored"]}


KEY = "2026cc_qm24"
AB = config.DATA / "exp_track" / "ab60e" / "greedy60" / KEY / "result.json"


@pytest.mark.skipif(os.environ.get("AUTOSCOUT_DATA_TESTS") != "1" or tracks60.log_path(KEY) is None
                    or not AB.exists() or not (config.OUTPUT / KEY / "scan.pkl").exists(),
                    reason="needs AUTOSCOUT_DATA_TESTS=1, the 60 Hz blob log, the scan and the A/B arm result")
def test_loader_reproduces_the_60hz_ab_arm(monkeypatch):
    """The production loader reproduces the adopted A/B arm (data/exp_track/ab60e/greedy60, analysis defaults of
    2026-10-19 01:00: process-of-elimination linking). If analysis defaults change, re-run that arm."""
    with open(config.OUTPUT / KEY / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    pipeline.load_dino(md, config.OUTPUT / KEY)
    monkeypatch.setattr(pipeline, "FUEL_60HZ", True)
    assert pipeline.load_fuel60(md) == 60
    assert {v: len(t) for v, t in md.tracks_all.items()} == {"main": 37654, "blue_end": 24362, "red_end": 21848}
    res = json.loads(json.dumps(pipeline.analyse(md, pipeline.Options()), default=pipeline._json_default))
    ab = json.loads(AB.read_text())
    assert len(res["shots"]) == len(ab["shots"])
    # the pure-video split (the hand-tuned credit rule: conftest switches the learned model off in tests)
    assert all(abs(res["teams"][t]["fuel"] - ab["teams"][t]["fuel_video"]) < 1e-6
               for t in ab["teams"] if not t.startswith("_"))
