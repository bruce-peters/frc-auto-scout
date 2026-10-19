import numpy as np

from autoscout import season, tba


def _mi(key, red, blue, yr, yb):
    bd = {"red": {"hubScore": {"totalCount": yr}}, "blue": {"hubScore": {"totalCount": yb}}}
    return tba.MatchInfo(key, key.split("_")[0], "qm", int(key.split("qm")[1]), 1, red, blue, [], bd, yr, yb)


def test_prior_events_only_finished_before_the_event(monkeypatch):
    monkeypatch.setattr(season, "event_simple", lambda k: {"start_date": "2026-09-18", "year": 2026})
    monkeypatch.setattr(tba, "team_events", lambda team, year: [
        {"key": "2026a", "end_date": "2026-03-08", "event_type": 1},
        {"key": "2026pre", "end_date": "2026-02-28", "event_type": 100},  # preseason
        {"key": "2026cc", "end_date": "2026-09-20", "event_type": 99},  # the event itself
        {"key": "2026b", "end_date": "2026-09-18", "event_type": 99},  # ends the day it starts
        {"key": "2026c", "end_date": "2026-07-26", "event_type": 99},
    ])
    assert [e["key"] for e in season.prior_events(1, "2026cc")] == ["2026a", "2026c"]


def test_season_epa_follows_statbotics_update(monkeypatch):
    def raw(key, t, level, red, blue, yr, yb):
        return {"key": key, "comp_level": level, "set_number": 1, "match_number": 1, "actual_time": t,
                "alliances": {"red": {"team_keys": [f"frc{x}" for x in red]}, "blue": {"team_keys": [f"frc{x}" for x in blue]}},
                "score_breakdown": {"red": {"hubScore": {"totalCount": yr}}, "blue": {"hubScore": {"totalCount": yb}}}}

    data = {"/events/2026/simple": [{"key": "2026a", "start_date": "2026-03-06", "end_date": "2026-03-08", "event_type": 1}],
            "/event/2026a/matches": [raw("2026a_qm1", 100, "qm", [1, 2, 3], [4, 5, 6], 60, 30),
                                     raw("2026a_sf1m1", 200, "sf", [1, 2, 3], [4, 5, 6], 90, 0)]}
    monkeypatch.setattr(tba, "get", lambda path: data[path])
    monkeypatch.setattr(season, "EPA_INIT_MATCHES", 1)
    season.season_epa.cache_clear()
    try:
        h = season.season_epa(2026)
    finally:
        season.season_epa.cache_clear()
    init = (60 + 30) / 2 / 3  # 15 per team
    a1 = init + 1 / 3 * (60 - 3 * init) / 3  # first qual: learning rate 1/3 of the alliance error share
    assert abs(h[1][0][1] - a1) < 1e-9 and abs(h[4][0][1] - (init + 1 / 3 * (30 - 3 * init) / 3)) < 1e-9
    a2 = a1 + 1 / 3 * (1 / 3) * (90 - 3 * a1) / 3  # playoff: weight 1/3, qual count stays at 1 (rate still 1/3)
    assert abs(h[1][1][1] - a2) < 1e-9 and [x[4] for x in h[1]] == [True, False]


def test_prior_oprs_blends_season_rates_with_the_matches_played():
    truth = {1: 30.0, 2: 60.0, 3: 90.0, 4: 40.0, 5: 50.0, 6: 70.0}
    sched = [([1, 2, 3], [4, 5, 6]), ([1, 4, 5], [2, 3, 6]), ([1, 2, 6], [3, 4, 5]), ([1, 3, 5], [2, 4, 6]),
             ([2, 5, 6], [1, 3, 4]), ([3, 5, 6], [1, 2, 4]), ([4, 6, 2], [1, 5, 3]), ([1, 6, 4], [2, 5, 3])]
    ms = [_mi(f"2026x_qm{i + 1}", r, b, sum(truth[t] for t in r), sum(truth[t] for t in b)) for i, (r, b) in enumerate(sched)]
    ms.append(_mi("2026x_qm99", [1, 2, 3], [4, 5, 6], 0, 0))
    by_key = {m.key: m for m in ms}
    old = tba.event_matches, tba.match
    tba.event_matches, tba.match = (lambda ev: ms), (lambda k: by_key[k])
    try:
        # season rates in other units (half the event's scale), a different ranking, team 6 unknown
        rates = {t: season.SeasonRate(r, 10.0) for t, r in {1: 40.0, 2: 20.0, 3: 30.0, 4: 10.0, 5: 25.0}.items()}
        rates[6] = None
        start = season.prior_oprs("2026x", None, rates=rates)
        assert start[1] == 40.0 and start[6] == np.mean([40, 20, 30, 10, 25])  # unknown team: the mean
        assert season.prior_oprs("2026x", "2026x_qm1", rates=rates) == start  # nothing played yet
        weak = season.prior_oprs("2026x", "2026x_qm99", season_weight=1e-6, rates=rates)
        # the event's own matches decide (team 6 keeps opr.fuel_opr's small ridge toward the event mean)
        assert all(abs(weak[t] - truth[t]) < (2.0 if t == 6 else 1.0) for t in truth)
        strong = season.prior_oprs("2026x", "2026x_qm99", season_weight=1e4, rates=rates)
        k = strong[1] / 40.0
        assert 1.5 < k < 3.0  # season units rescaled to the event's alliance totals
        assert all(abs(strong[t] - k * rates[t].rate) < 0.5 for t in (1, 2, 3, 4, 5))
    finally:
        tba.event_matches, tba.match = old
