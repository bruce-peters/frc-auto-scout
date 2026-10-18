import json

import pytest

from autoscout import labels


def _res(counts_red, counts_blue=()):
    return {"timeline": {"auto_start": 6.0, "teleop_start": 31.0,
                         "counts": {"red": list(counts_red), "blue": list(counts_blue)}}}


def test_bursts_group_by_gap_and_length():
    # red: 10->12->15 within 1 s (one burst), pause, then a long run that must be split at 5 s
    red = [(10.0, 2), (10.5, 5), (11.0, 9), (20.0, 10)] + [(20.0 + 0.5 * k, 10 + k) for k in range(1, 16)]
    b = labels.bursts(_res(red))
    assert len(b) >= 3  # the 20-27.5 s run is split into <= 5 s pieces
    assert b[0]["t0"] == 10.0 and b[0]["t1"] == 11.0 and b[0]["balls"] == 9
    assert all(x["t1"] - x["t0"] <= labels.MAX_LEN_S for x in b)
    assert sum(x["balls"] for x in b) == 25  # every counted ball is in exactly one burst
    assert b[0]["play0"] == pytest.approx(10.0 - labels.LEAD_S)


def test_bursts_drop_post_match_and_never_count_decreases():
    red = [(40.0, 5), (40.4, 3), (40.8, 7), (200.0, 9)]  # misread 3 ignored; 200 s is after the match
    b = labels.bursts(_res(red))
    assert len(b) == 1 and b[0]["balls"] == 7


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "LABELS", tmp_path)
    labs = {"red-1000": {"alliance": "red", "t0": 10, "t1": 11, "balls": 9, "status": "labeled", "shooters": [5026]},
            "red-2000": {"alliance": "red", "t0": 20, "t1": 21, "balls": 3, "status": "bogus", "shooters": []}}
    doc = labels.save("2026casnv_qm2", "Bruce", labs)
    assert list(doc["labels"]) == ["red-1000"]  # invalid status dropped
    back = labels.load("2026casnv_qm2", "Bruce")
    assert back["labels"]["red-1000"]["shooters"] == [5026]
    assert labels.progress("2026casnv_qm2") == {"Bruce": 1}
    with pytest.raises(ValueError):
        labels.save("2026casnv_qm2", "../evil", labs)
    assert json.loads((tmp_path / "2026casnv_qm2__Bruce.json").read_text())["labeller"] == "Bruce"
