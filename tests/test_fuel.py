import numpy as np

from autoscout import fuel


def _blob(u, v, w=8, h=8):
    return fuel.Blob(u, v, w * h * 0.8, w, h, u - w / 2, v - h / 2)


def test_merge_keeps_motion_blob_for_same_ball_and_adds_new_detections():
    blobs = [_blob(100, 100)]
    dets = np.array([[96, 95, 105, 104, 0.9],  # same ball as the motion blob
                     [300, 50, 310, 60, 0.8],  # missed by motion
                     [500, 50, 510, 60, 0.1]], np.float32)  # below the confidence threshold
    out = fuel.merge_detections(blobs, dets, conf=0.25)
    assert len(out) == 2
    assert out[0].src == "motion" and out[0].u == 100
    assert out[1].src == "det" and abs(out[1].u - 305) < 1e-6


def test_merge_splits_a_stream_merged_into_one_motion_blob():
    blobs = [fuel.Blob(110, 100, 300, 40, 10, 90, 95)]  # one wide blob covering two balls
    dets = np.array([[92, 95, 102, 105, 0.7], [118, 95, 128, 105, 0.7]], np.float32)
    out = fuel.merge_detections(blobs, dets, conf=0.25)
    assert sorted(round(b.u) for b in out) == [97, 123]
    assert all(b.src == "det" for b in out)


def test_view_detections_drops_masked_and_shifts_to_view_pixels():
    valid = np.ones((100, 200), np.uint8)
    valid[:20] = 0
    dets = np.array([[1010, 510, 1020, 520, 0.5],  # inside, valid
                     [1010, 505, 1020, 512, 0.5],  # centre in the masked strip
                     [10, 10, 20, 20, 0.5]], np.float32)  # other view
    out = fuel.view_detections(dets, (1000, 490, 1200, 590), valid)
    assert out.shape == (1, 5)
    assert np.allclose(out[0, :4], [10, 20, 20, 30])


def test_view_blobs_finds_a_moving_yellow_ball():
    H, W = 120, 160
    frames = []
    for x in (40, 60, 80):
        f = np.full((H, W, 3), 60, np.uint8)
        import cv2
        cv2.circle(f, (x, 60), 5, (0, 220, 240), -1)  # BGR yellow
        frames.append(f)
    out = fuel.moving_blobs(*frames)
    assert len(out) == 1
    b = out[0]
    assert abs(b.u - 60) < 1.5 and abs(b.v - 60) < 1.5
    assert b.x0 <= 57 and b.x0 + b.w >= 63  # box covers the ball (+-1 px from open/dilate)


def _stream(n_balls=12, rate=20.0, fps=30.0, speed=600.0, g=900.0, dur=1.2):
    """Balls launched from (500, 400) every 1/rate s, rising at `speed` px/s with gravity g px/s^2
    (spacing per ball ~ speed / rate = 30 px vs ~20 px moved per frame: the aliasing regime)."""
    frames = []
    for i in range(int(dur * fps)):
        t = i / fps
        blobs = []
        for b in range(n_balls):
            tb = t - b / rate
            if 0 <= tb <= 0.6:
                blobs.append(_blob(500 + 150 * tb, 400 - speed * tb + 0.5 * g * tb * tb, 6, 6))
        frames.append((t, blobs))
    return frames


def _smooth_rising(tracks):
    n = 0
    for tr in tracks:
        if len(tr.pts) >= 6:
            vu, vv = tr.velocity0()
            if vv < -300 and tr.fit()["res"] < 2.0:
                n += 1
    return n


def test_ballistic_tracker_follows_each_ball_of_a_stream():
    frames = _stream()
    bt = fuel.BallisticTracker()
    for t, b in frames:
        bt.update(t, b)
    good_bt = _smooth_rising(bt.flush())
    gr = fuel.FlightTracker(gate=45.0 * 2 ** 0.7)
    for t, b in frames:
        gr.update(t, b)
    good_gr = _smooth_rising(gr.flush())
    assert good_bt >= 10  # nearly every one of the 12 balls gets its own smooth rising arc
    assert good_bt > good_gr


def test_split_at_jumps_cuts_a_hop_to_another_ball():
    pts = [(i / 30, 100 + 5 * i, 300 - 20 * i, 30) for i in range(8)]
    pts += [(i / 30, 140 + 5 * i, 330 - 20 * i, 30) for i in range(8, 16)]  # hop: 30 px off the path
    tr = fuel.FlightTrack(1, pts)
    out = fuel.split_at_jumps([tr])
    assert len(out) == 2 and len(out[0].pts) == 8 and len(out[1].pts) == 8


def test_ballistic_tracker_scales_seeding_to_60hz():
    # the same stream sampled at 60 Hz: per-frame steps halve, the tracker must still seed every ball
    frames = _stream(fps=60.0)
    bt = fuel.BallisticTracker()
    for t, b in frames:
        bt.update(t, b)
    assert _smooth_rising(bt.flush()) >= 10
