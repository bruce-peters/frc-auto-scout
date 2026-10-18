import numpy as np
import pytest

from autoscout import attribution, field, identity, ocr, overlay
from autoscout.robots import FieldTracker, Segment


def test_clean_counts_rejects_misreads_and_keeps_monotonic():
    raw = [(0, 0), (1, 0), (2, 5), (2.25, 50), (2.5, 6), (3, 6), (3.5, 8), (4, 3), (4.5, 9), (5, 9)]
    out = overlay._clean_counts(raw)
    vals = [c for _, c in out]
    assert vals == sorted(vals)
    assert 50 not in vals and 3 not in vals
    assert vals[-1] == 9


def test_robust_offset_recovers_countdown_start():
    start = 6.13
    samples = []
    for vt in np.arange(start + 0.05, start + 19.9, 0.25):
        remaining = 20 - (vt - start)
        samples.append((vt, int(np.ceil(remaining))))
    est = overlay._robust_offset(samples, 20)
    assert abs(est - start) < 0.15


def test_parse_overlay_strings():
    assert overlay.parse_count("170/360") == 170
    assert overlay.parse_count("9 / 100") == 9
    assert overlay.parse_clock("1:36") == 96
    assert overlay.parse_clock("1:6") is None
    assert overlay.parse_shift("3/6:16") == (3, 16)


def test_tag_corners_square_and_vertical():
    tags = field.load_tags()
    c = field.tag_corners(tags[18])
    edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    assert np.allclose(edges, field.TAG_SIZE, atol=1e-6)
    assert np.allclose(c[:, 1], c[0, 1])  # tag 18 faces -y: corners share y


def test_camera_roundtrip_backproject():
    K = [[1300, 0, 960], [0, 1300, 352], [0, 0, 1]]
    import cv2
    R = cv2.Rodrigues(np.array([1.9, 0.0, 0.0]))[0]
    C = np.array([8.0, -5.0, 3.5])
    t = -R @ C
    cam = field.Camera(K, [0, 0, 0, 0, 0], cv2.Rodrigues(R)[0].ravel().tolist(), t.tolist(), (1920, 704), 0, 0)
    pts = np.array([[4.0, 2.0, 0.0], [12.0, 6.0, 0.0], [8.2, 4.0, 0.0]])
    uv = cam.project(pts)
    back = cam.backproject(uv, z=0.0)
    assert np.allclose(back, pts[:, :2], atol=1e-6)


def _look_at(C, target):
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])  # world -> camera rows


def test_fixed_intrinsics_fallback_recovers_pan_and_drops_misplaced_tag(monkeypatch):
    import cv2
    tags = field.load_tags()
    K, D = [[690.0, 0, 467.5], [0, 690.0, 167.5], [0, 0, 1]], [-0.3, 0, 0, 0, 0]
    C = np.array([0.6, -0.15, 0.52])
    prior = {"K": K, "dist": D, "centre": C.tolist(), "f": 690.0}
    frames, dets, truth = [], {}, []
    for k in range(6):  # an end camera panning across the blue hub; 1-2 tags per frame
        R = _look_at(C, np.array([4.0, 4.0 + 0.1 * k, 1.0]))
        rv, tv = cv2.Rodrigues(R)[0].ravel(), -R @ C
        f = np.zeros((335, 935, 3), np.uint8)
        frames.append(f)

        def proj(i):
            return cv2.projectPoints(field.tag_corners(tags[i]), rv, tv, np.array(K), np.array(D))[0].reshape(4, 2)
        d = {i: proj(i) for i in ((25, 26) if k % 2 else (26,))}
        if k == 3:
            d[3] = proj(18)  # tag 3's id where tag 18 is: must be dropped by the consensus
        dets[id(f)] = d
        truth.append(R)
    monkeypatch.setattr(field, "_detect_masked", lambda img, masks: dets[id(img)])
    diag = {}
    tr = field.calibrate_view_fixed(frames, [float(k) for k in range(6)], tags, (), prior, diag=diag)
    assert tr is not None and len(tr.times) == 6 and tr.rms < 0.05
    assert diag["centre_from"] == "event"
    for k, R in enumerate(truth):
        assert np.allclose(tr.at(float(k)).R, R, atol=1e-5)
        assert np.allclose(tr.at(float(k)).center_world(), C, atol=1e-6)


def test_calib_prior_median_and_exclusions(monkeypatch, tmp_path):
    import cv2
    monkeypatch.setattr(field.config, "CACHE", tmp_path)
    R = _look_at(np.array([0.6, -0.15, 0.5]), np.array([4.0, 4.0, 1.0]))
    rv, tv = cv2.Rodrigues(R)[0].ravel().tolist(), (-R @ np.array([0.6, -0.15, 0.5])).tolist()
    for m, f in (("a", 650.0), ("b", 700.0), ("c", 3000.0)):
        cam = field.CameraTrack([[f, 0, 467.5], [0, f, 167.5], [0, 0, 1]], [-0.3, 0, 0, 0, 0], [0.0], [rv], [tv],
                                (935, 335), 0.6, 4)
        assert field.update_calib_prior("ev", "blue_end", m, cam)
    assert field.load_calib_prior("ev", "blue_end", exclude="a") is None  # needs PRIOR_MIN_MATCHES matches
    bad = field.CameraTrack([[690, 0, 467.5], [0, 690, 167.5], [0, 0, 1]], [0] * 5, [0.0], [rv], [tv], (935, 335), 7.9, 4)
    assert not field.update_calib_prior("ev", "blue_end", "d", bad)  # rms >= MAX_RMS: seen, not added
    rec = field.CameraTrack([[690, 0, 467.5], [0, 690, 167.5], [0, 0, 1]], [0] * 5, [0.0], [rv], [tv], (935, 335), 0.5, 1)
    rec.recovered = "fixed_K"
    assert not field.update_calib_prior("ev", "blue_end", "e", rec)  # recovered cameras never feed the prior
    pr = field.load_calib_prior("ev", "blue_end", (935, 335))
    assert pr["n"] == 3 and pr["f"] == 700.0 and np.allclose(pr["centre"], [0.6, -0.15, 0.5])
    assert field.focal_ok(type("C", (), {"K": [[800.0]]})(), pr) and not field.focal_ok(type("C", (), {"K": [[3000.0]]})(), pr)


def test_split_increments_follows_shots_and_zone():
    a = attribution.TeamSignals(1, shots=np.array([10.0, 10.2, 10.4]), shot_w=np.ones(3),
                                zone_t=np.arange(0, 20, 0.5), zone_v=np.ones(40))
    b = attribution.TeamSignals(2, shots=np.array([]), shot_w=np.array([]),
                                zone_t=np.arange(0, 20, 0.5), zone_v=np.ones(40))
    c = attribution.TeamSignals(3, shots=np.array([10.1]), shot_w=np.ones(1),
                                zone_t=np.arange(0, 20, 0.5), zone_v=np.zeros(40))  # out of zone
    teleop = lambda t: t + 40  # noqa: E731  (zone gating applies in teleop)
    res = attribution.split_increments([(12.0, 10.0)], [a, b, c], teleop)
    tot = {k: sum(v.values()) for k, v in res.items()}
    assert pytest.approx(sum(tot.values())) == 10.0
    assert tot[1] > 8.0 and tot[3] == 0.0
    # in AUTO the zone is not used (AUTO_ZONE_GATE False): the out-of-zone launcher gets its share
    res = attribution.split_increments([(12.0, 10.0)], [a, b, c], lambda t: t)
    tot = {k: sum(v.values()) for k, v in res.items()}
    assert pytest.approx(sum(tot.values())) == 10.0
    assert tot[3] > 1.0 and tot[1] > tot[3]


def test_identity_viterbi_uses_ocr_and_continuity():
    # three continuous segments, one of which (slot 2) is re-acquired as a new segment later
    s0 = Segment(0, "red", 0, 0, 100, points=[(0, 1, 1)])
    s1 = Segment(1, "red", 1, 0, 100, points=[(0, 2, 2)])
    s2 = Segment(2, "red", 2, 0, 50, points=[(0, 3, 3)])
    s3 = Segment(3, "red", 2, 60, 100, points=[(60, 3, 3)])
    ev = {0: np.array([0, 3.0, 0]), 1: np.array([2.0, 0, 0]), 2: np.zeros(3), 3: np.zeros(3)}
    lab = identity.assign([s0, s1, s2, s3], ev)
    assert lab[0] == 1 and lab[1] == 0
    assert lab[2] == 2 and lab[3] == 2  # the only team left


def test_ocr_similarity_prefers_right_team():
    ev = ocr.team_evidence([("5026", 0.9), ("502", 0.6), ("026", 0.5)], [5026, 5419, 9470])
    assert ev[0] > 0 and ev[1] == 0 and ev[2] == 0


def test_field_tracker_keeps_three_slots():
    tr = FieldTracker()
    for i in range(50):
        t = i * 0.1
        obs = [{"alliance": "red", "xy": np.array([2.0 + 0.1 * i, 2.0]), "dets": [], "conf": 0.9},
               {"alliance": "red", "xy": np.array([10.0, 6.0]), "dets": [], "conf": 0.9}]
        tr.step(t, obs)
    segs = tr.finish()
    assert len({s.slot for s in segs}) == 2
    assert all(s.alliance == "red" for s in segs)


def test_vision_count_fit_recovers_linear_rate():
    from autoscout import vision_count as vc
    rng = np.random.default_rng(0)
    nf = len(vc.feature_names())
    samples = []
    w_true = np.zeros(nf)
    w_true[2] = 0.8  # main_exit_n
    for _ in range(5):
        X = rng.uniform(0, 5, (200, nf))
        y = X @ w_true + rng.normal(0, 0.1, 200)
        samples.append((X, y))
    w = vc.fit(samples)
    assert abs(w[2] - 0.8) < 0.05
    assert np.all(w >= 0)


def test_hopper_drain_detects_emptying():
    from autoscout.pipeline import _hopper_drains
    from autoscout.robots import Detection
    dets = []
    for i in range(40):
        t = i * 0.1
        fill = 0.4 if t < 1.5 else max(0.05, 0.4 - 0.3 * (t - 1.5))  # empties from t=1.5 s
        d = Detection(t, "main", (0, 0, 10, 10), 0.9, "red")
        d.fill = fill
        dets.append(d)
    drains = _hopper_drains(dets)
    assert drains, "expected drain events"
    ts = [t for t, _ in drains]
    assert min(ts) > 1.0 and max(ts) < 3.5


def test_period_spread_follows_vision_rate():
    rate_t = np.arange(0, 200, 0.5)
    rate_v = np.where((rate_t > 30) & (rate_t < 35), 5.0, 0.0)  # busy only 30-35 s (video)
    inc = attribution.increments_from_periods({"transition": 40}, lambda mt: mt, rate=(rate_t, rate_v))
    tot = sum(n for _, n in inc)
    busy = sum(n for t, n in inc if 30 <= t <= 35)
    assert abs(tot - 40) < 1e-6 and busy > 0.8 * tot


def test_track_position_interpolates_and_holds():
    from autoscout.pipeline import _track_position
    arr = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [11.0, 3.0, 1.0], [12.0, 3.0, 1.0]])
    assert _track_position(arr, 0.5)[2] is True
    x, y, seen = _track_position(arr, 6.0)  # hidden for 10 s: interpolated
    assert not seen and abs(x - 2.0) < 1e-6
    assert _track_position(arr, 14.0)[2] is False  # held after the last sighting
    assert _track_position(arr, 30.0) is None
    long_gap = np.array([[0.0, 1.0, 1.0], [40.0, 3.0, 1.0]])
    assert _track_position(long_gap, 20.0) is None


def test_hidden_robot_gets_credit_when_visible_ones_are_idle():
    grid = np.arange(0, 20, 0.5)
    seen = attribution.TeamSignals(1, shots=np.array([]), shot_w=np.array([]),
                                   zone_t=grid, zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.ones(len(grid)))
    hidden = attribution.TeamSignals(2, shots=np.array([]), shot_w=np.array([]),
                                     zone_t=grid, zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.zeros(len(grid)))
    old = attribution.VIS_AWARE
    try:
        attribution.VIS_AWARE = True
        res = attribution.split_increments([(12.0, 10.0)], [seen, hidden], lambda t: t)
    finally:
        attribution.VIS_AWARE = old
    tot = {k: sum(v.values()) for k, v in res.items()}
    assert tot[2] > 8.0  # the in-zone robot nobody could see takes most of the credit


def test_identity_absent_state_lets_duplicate_share_a_label():
    # slots 0 and 2 both read team 0 (one robot tracked twice); team 1 is never read
    s0 = Segment(0, "red", 0, 0, 100, points=[(0, 1, 1), (100, 1, 1)])
    s1 = Segment(1, "red", 1, 0, 100, points=[(0, 8, 5), (100, 8, 5)])
    s2 = Segment(2, "red", 2, 0, 100, points=[(0, 3, 1), (100, 3, 1)])
    ev = {0: np.array([3.0, 0, 0]), 1: np.array([0, 0, 3.0]), 2: np.array([2.0, 0, 0])}
    assert identity.assign([s0, s1, s2], ev)[2] == 1  # forced onto the unread team
    lab, absent = identity.assign_absent([s0, s1, s2], ev, absent_cost=[np.inf, 1.0, np.inf])
    assert lab == {0: 0, 1: 2, 2: 0}
    assert absent[1] and absent[1][0][0] == 0 and absent[1][-1][1] >= 100 and not absent[0] and not absent[2]
    # an absent cost above the evidence keeps the 6-permutation answer
    lab, absent = identity.assign_absent([s0, s1, s2], ev, absent_cost=[np.inf, 5.0, np.inf])
    assert lab == identity.assign([s0, s1, s2], ev) and not any(absent.values())
    # never read anywhere, labelled segments carry teammates' reads -> dead-robot evidence
    assert identity.dead_teams([s0, s1, s2], ev, {0: 0, 1: 2, 2: 1}).tolist() == [False, True, False]


def test_absent_or_parked_robot_gets_no_prior_or_hidden_credit():
    grid = np.arange(0, 20, 0.5)
    shooter = attribution.TeamSignals(1, shots=np.array([10.0]), shot_w=np.array([1.0]),
                                      zone_t=grid, zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.ones(len(grid)))
    idle = attribution.TeamSignals(2, zone_t=grid, zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.ones(len(grid)))
    gone = attribution.TeamSignals(3, zone_t=grid, zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.zeros(len(grid)))
    tm = lambda t: t + 40  # noqa: E731  (teleop)
    base = {k: sum(v.values()) for k, v in attribution.split_increments([(12.0, 10.0)], [shooter, idle, gone], tm).items()}
    assert base[3] > base[2] > 0  # hidden robot: HIDDEN credit; visible idle robot: PRIOR
    gone.absent = [(0.0, 20.0)]
    idle.prior_off = True
    res = {k: sum(v.values()) for k, v in attribution.split_increments([(12.0, 10.0)], [shooter, idle, gone], tm).items()}
    assert res[2] == 0.0 and res[3] == 0.0 and pytest.approx(res[1]) == 10.0


def test_learned_credit_model_matches_rule_form_and_opr_prior_keeps_total(tmp_path):
    import json

    grid = np.arange(0, 20, 0.5)
    sig = [attribution.TeamSignals(1, shots=np.array([10.0, 10.2]), shot_w=np.ones(2), zone_t=grid,
                                   zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.ones(len(grid))),
           attribution.TeamSignals(2, shots=np.array([]), shot_w=np.array([]), zone_t=grid,
                                   zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.zeros(len(grid))),
           attribution.TeamSignals(3, shots=np.array([10.1]), shot_w=np.ones(1), zone_t=grid,
                                   zone_v=np.ones(len(grid)), vis_t=grid, vis_v=np.ones(len(grid)),
                                   dist_t=grid, dist_v=np.full(len(grid), 7.5))]
    m = {"prior": attribution.PRIOR, "hidden": attribution.HIDDEN, "b": 1.0, "gz": 1.0, "g_dist": 0.0,
         "dist_center": 3.5, "dist_scale": 2.0, "g_opr": 0.0, "opr_floor": 0.02, "zone_eps": 1e-3}
    p1, p2 = tmp_path / "rule.json", tmp_path / "prior.json"
    p1.write_text(json.dumps(m))
    p2.write_text(json.dumps(dict(m, g_dist=-1.0, g_opr=1.0)))
    inc, tm = [(12.0, 10.0), (14.0, 6.0)], (lambda t: t + 40)
    old = attribution.ATTR_MODEL
    try:
        attribution.ATTR_MODEL = None
        rule = {k: sum(v.values()) for k, v in attribution.split_increments(inc, sig, tm).items()}
        attribution.ATTR_MODEL = str(p1)  # the rule's own parameters: same split
        same = {k: sum(v.values()) for k, v in attribution.split_increments(inc, sig, tm, opr={1: 1, 2: 9, 3: 1}).items()}
        attribution.ATTR_MODEL = str(p2)  # far-from-hub penalty and a strong OPR prior for team 2
        log: list = []
        pri = attribution.split_increments(inc, sig, tm, log=log, opr={1: 10.0, 2: 90.0, 3: 10.0})
    finally:
        attribution.ATTR_MODEL = old
    tot = {k: sum(v.values()) for k, v in pri.items()}
    assert all(pytest.approx(rule[k], rel=1e-3) == same[k] for k in rule)
    assert pytest.approx(sum(tot.values())) == 16.0 and tot[2] > rule[2] and tot[3] < rule[3]
    assert pytest.approx(sum(b for _, _, b in log), abs=0.01) == 16.0


def test_shooting_signals_match_the_dataset_code_and_enter_the_model_weights():
    """autoscout/shooting.py equals data/exp_launch/prod_signals.py (which build_ds.py fits on), and a version-2
    credit model adds g * log1p(window mean) per shooting signal to a robot's log weight."""
    import importlib.util
    from pathlib import Path

    from autoscout import attribution, shooting

    spec = importlib.util.spec_from_file_location(
        "prod_signals", Path(__file__).resolve().parent.parent / "data" / "exp_launch" / "prod_signals.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(0)
    grid = np.arange(0.0, 20.0, 0.5)
    n = 300
    t, view, team = rng.uniform(0, 20, n), rng.integers(0, 3, n), rng.choice([-1, 11, 22, 33], n)
    X = rng.normal(size=(n, 768)).astype(np.float32)
    ridge = {"w": rng.normal(size=769) * 0.01, "mu": np.zeros(768), "sd": np.ones(768)}
    a, b = shooting.dinoc_core(t, view, team, X, ridge, grid), ref.dinoc_core(t, view, team, X, ridge, grid)
    assert a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)
    fill = np.round(rng.uniform(0, 0.5, n), 3)
    a = shooting.drain_core(np.round(t, 3), view, fill, team, grid)
    b = ref.drain_core(np.round(t, 3), view, fill, team, grid)
    assert a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)

    base = {"prior": 0.1, "hidden": 2.0, "b": 0.6, "gz": 0.5, "g_dist": 0.0, "dist_center": 3.5, "dist_scale": 2.0,
            "zone_eps": 1e-3}
    v2 = dict(base, g_dinoc=2.0, g_drain=1.0, shooting={"form": "window"})
    ts = [attribution.TeamSignals(k) for k in (11, 22, 33)]
    ts[0].dinoc_t, ts[0].dinoc_v = grid, np.full(len(grid), 3.0)
    ts[1].drain_t, ts[1].drain_v = grid, np.full(len(grid), 1.0)
    w0, w1 = attribution._model_weights(base, ts, 10.0, False), attribution._model_weights(v2, ts, 10.0, False)
    lr = np.log(w1 / w0)
    assert np.allclose(lr - lr[2], [2.0 * np.log1p(3.0), 1.0 * np.log1p(1.0), 0.0])
