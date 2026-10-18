import cv2
import numpy as np

from autoscout import field, maincal

L, W = maincal.L, maincal.W


def _look_at(C, target):
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])  # world -> camera rows


def _chezy_like():
    """A high camera beyond the +y sideline looking back across the field (rotation ~180 deg, |rvec| ~ pi)."""
    C = np.array([8.0, 21.5, 6.5])
    K = np.array([[1430.0, 0, 960], [0, 1430.0, 320], [0, 0, 1]])
    return C, K, np.zeros(5)


def test_track_interpolates_across_rvec_branches():
    C, K, D = _chezy_like()
    R0 = _look_at(C, np.array([8.3, 4.0, 0.0]))
    R1 = _look_at(C, np.array([8.4, 4.05, 0.0]))
    r0, r1 = cv2.Rodrigues(R0)[0].ravel(), cv2.Rodrigues(R1)[0].ravel()
    th = np.linalg.norm(r1)
    assert th > 2.5  # near pi: two rvecs per rotation
    r1_alt = r1 * (1 - 2 * np.pi / th)  # the same rotation on the other branch
    assert np.allclose(cv2.Rodrigues(r1_alt)[0], R1, atol=1e-9)
    tr = field.CameraTrack(K.tolist(), D.tolist(), [0.0, 4.0], [r0.tolist(), r1_alt.tolist()],
                           [(-R0 @ C).tolist(), (-R1 @ C).tolist()], (1920, 640), 0.5, 4)
    mid = tr.at(2.0)
    # halfway between two poses 0.3 deg apart: within 0.2 deg of both (blending across branches gave ~180 deg off)
    assert field._rot_angle(mid.rvec, r0) < 0.2 and field._rot_angle(mid.rvec, r1) < 0.2
    assert np.linalg.norm(mid.center_world() - C) < 0.05
    # rvecs far from pi interpolate exactly as before (plain linear blend)
    s0, s1 = [0.1, 1.2, -0.3], [0.12, 1.25, -0.28]
    tr2 = field.CameraTrack(K.tolist(), D.tolist(), [0.0, 4.0], [s0, s1], [[0, 0, 5.0], [0, 0, 5.2]], (1920, 640), 0.5, 4)
    assert np.allclose(tr2.at(1.0).rvec, 0.75 * np.array(s0) + 0.25 * np.array(s1), atol=0, rtol=0)


def _strip_obs(C, K, D, times, noise=0.0, seed=0):
    """Tags 2, 11, 21, 24 (one plane, one height: the Chezy main view) seen by a static camera."""
    tags = field.load_tags()
    rng = np.random.default_rng(seed)
    R = _look_at(C, np.array([8.3, 4.0, 0.0]))
    rv, tv = cv2.Rodrigues(R)[0].ravel(), -R @ C
    obs = []
    for t in times:
        d = {}
        for i in (2, 11, 21, 24):
            p = cv2.projectPoints(field.tag_corners(tags[i]), rv, tv, K, D)[0].reshape(4, 2)
            d[i] = p + rng.normal(0, noise, p.shape)
        obs.append((float(t), d))
    return tags, obs, R


def test_static_fit_shares_centre_and_keeps_rvecs_on_one_branch():
    C, K, D = _chezy_like()
    tags, obs, R = _strip_obs(C, K, D, np.arange(0.0, 40.0, 4.0), noise=0.15)
    tr = maincal.fit_static(obs, tags, (1920, 640), K, D, C + np.array([0.3, 0.5, 0.2]), fix_intrinsics=True)
    assert tr is not None and tr.rms < 0.3
    cents = np.array([tr.at(t).center_world() for t in tr.times])
    assert np.ptp(cents, axis=0).max() < 1e-6  # one centre for every window
    r = np.array(tr.rvecs)
    assert np.abs(np.diff(r, axis=0)).max() < 0.05  # consecutive rvecs on one branch
    # between windows the pose stays put (a static camera): the blend agrees with both neighbouring poses ...
    mid = tr.at(6.0).rvec
    assert field._rot_angle(mid, tr.rvecs[1]) < 0.05 and field._rot_angle(mid, tr.rvecs[2]) < 0.05
    # ... and with the truth to within what a 0.15 px noisy tag strip determines (elevation trades against the
    # centre along the strip's weak direction, so this is a loose bound)
    assert field._rot_angle(mid, cv2.Rodrigues(R)[0].ravel()) < 1.0


def test_guard_drops_window_poses_that_put_boxes_off_the_field():
    C, K, D = _chezy_like()
    R = _look_at(C, np.array([8.3, 4.0, 0.0]))
    good = cv2.Rodrigues(R)[0].ravel()
    bad = cv2.Rodrigues(_look_at(C, np.array([8.3, -6.0, 0.0])))[0].ravel()  # tilted up: floor far behind the field
    times = [0.0, 4.0, 8.0, 12.0]
    rvecs = [good, good, bad, good]
    tr = field.CameraTrack(K.tolist(), D.tolist(), times, [r.tolist() for r in rvecs],
                           [(-cv2.Rodrigues(r)[0] @ C).tolist() for r in rvecs], (1920, 640), 0.5, 4)
    cam = field.Camera(K.tolist(), D.tolist(), good.tolist(), (-R @ C).tolist(), (1920, 640), 0.5, 4)
    pts = np.array([[x, y, 0.0] for x in np.linspace(3, 13, 6) for y in np.linspace(2, 6, 4)])
    uv = cam.project(pts)
    boxes = np.array([(t, u, v) for t in np.arange(-1.0, 13.0, 0.5) for u, v in uv])
    off_before = maincal.off_field(tr, boxes, 0.0).mean()
    tr2, rep = maincal.guard(tr, boxes, 0.0)
    assert rep["dropped"] == 1 and not rep["flag"] and 8.0 not in tr2.times
    assert off_before > 0.1 and maincal.off_field(tr2, boxes, 0.0).mean() == 0.0
    # a track with nothing wrong is returned as is
    tr3, rep3 = maincal.guard(tr2, boxes, 0.0)
    assert tr3 is tr2 and rep3["dropped"] == 0


def test_accept_main_cam_keeps_a_better_candidate_and_leaves_a_good_camera_alone():
    from types import SimpleNamespace

    from autoscout import pipeline, robots
    C, K, D = _chezy_like()
    R = _look_at(C, np.array([8.3, 4.0, 0.0]))
    good = cv2.Rodrigues(R)[0].ravel()
    bad = cv2.Rodrigues(_look_at(C, np.array([8.3, -6.0, 0.0])))[0].ravel()
    times = [0.0, 4.0, 8.0, 12.0]

    def track(rvecs):
        return field.CameraTrack(K.tolist(), D.tolist(), times, [r.tolist() for r in rvecs],
                                 [(-cv2.Rodrigues(r)[0] @ C).tolist() for r in rvecs], (1920, 640), 0.5, 4)
    cam = field.Camera(K.tolist(), D.tolist(), good.tolist(), (-R @ C).tolist(), (1920, 640), 0.5, 4)
    pts = np.array([[x, y, 0.0] for x in np.linspace(3, 13, 6) for y in np.linspace(2, 6, 4)])
    uv = cam.project(pts)  # main view = the top of the frame at Chezy: view pixels = frame pixels
    dets = [robots.Detection(float(t), "main", (u - 20, v - 40, u + 20, v), 0.9, "red")
            for t in np.arange(-1.0, 13.0, 0.5) for u, v in uv]
    md = SimpleNamespace(cams={"main": track([good, good, bad, good])}, prod="chezy", dets=dets)
    rep = pipeline.accept_main_cam(md, [("static", track([good] * 4))])
    assert rep["accepted"] in ("guard", "static") and not rep["flag"]
    assert maincal.off_field(md.cams["main"], pipeline._main_boxes(md), pipeline.ROBOT_HALF_DEPTH).mean() == 0.0
    # a camera with nothing off the field stays as it is
    md2 = SimpleNamespace(cams={"main": track([good] * 4)}, prod="chezy", dets=dets)
    keep = md2.cams["main"]
    rep2 = pipeline.accept_main_cam(md2, [("static", track([good] * 4))])
    assert rep2["accepted"] is None and md2.cams["main"] is keep
