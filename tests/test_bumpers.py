from autoscout import bumpers


def _px_width(scale):
    """floor_width for a camera looking straight down: `scale` metres per pixel."""
    return lambda x0, x1, y: (x1 - x0) * scale


def test_two_faces_of_one_robot_are_grouped():
    front = ((100, 200, 160, 215), 0.9, "red")  # 60 px wide face
    side = ((158, 195, 190, 213), 0.7, "red")  # receding side face touching its corner
    groups = bumpers.group_sides([front, side], _px_width(0.015))  # 90 px = 1.35 m
    assert groups == [[0, 1]]


def test_two_robots_side_by_side_stay_apart():
    a = ((100, 200, 160, 215), 0.9, "blue")
    b = ((162, 200, 222, 215), 0.9, "blue")  # touching, but together 122 px = 1.83 m
    assert sorted(bumpers.group_sides([a, b], _px_width(0.015))) == [[0], [1]]


def test_colours_and_distant_boxes_never_group():
    a = ((100, 200, 160, 215), 0.9, "red")
    b = ((161, 200, 200, 215), 0.9, "blue")
    c = ((400, 200, 450, 215), 0.9, "red")
    assert sorted(bumpers.group_sides([a, b, c], _px_width(0.01))) == [[0], [1], [2]]


def test_duplicate_boxes_merge_and_pixel_fallback_without_camera():
    a = ((100, 200, 160, 215), 0.9, "red")
    dup = ((102, 201, 161, 216), 0.5, "red")
    far_side = ((159, 202, 300, 214), 0.6, "red")  # union 200 px > 9 x 15 px bumper height
    groups = bumpers.group_sides([a, dup, far_side], None)
    assert sorted(groups) == [[0, 1], [2]]


def test_robot_box_without_camera_stands_on_the_bumpers():
    x0, y0, x1, y1 = bumpers.robot_box((100, 200, 160, 215), None, 0, 0, view_top=0)
    assert (x0, x1, y1) == (100, 160, 215)
    assert 215 - y0 >= 1.5 * 15 and y0 >= 0
    # clipped to the view's top edge
    assert bumpers.robot_box((100, 20, 160, 35), None, 0, 0, view_top=10)[1] == 10
