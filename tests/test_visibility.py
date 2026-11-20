"""传送门可见性剔除的 15 个验收测试。

初始状态下 ``portal/visibility.py`` 的 8 个函数全部 raise
NotImplementedError，因此本文件 15 个用例全部失败（15 failed）。
本文件不可修改；修复时只允许改 ``portal/visibility.py``。
"""

import math

import pytest

from portal.visibility import (
    PortalError,
    clip_polygon,
    cull_objects,
    loop_guard,
    merge_frusta,
    portal_frustum,
    room_bounds,
    traverse,
    visibility_batch,
)

EPS = 1e-6


# ---------------------------------------------------------------------------
# 几何辅助（仅测试用，不调用被测函数）
# ---------------------------------------------------------------------------


def _unit(plane):
    a, b, c, d = plane
    length = math.sqrt(a * a + b * b + c * c)
    return [a / length, b / length, c / length, d / length]


def _shift(plane, origin):
    a, b, c, d = plane
    ox, oy, oz = origin
    return [a, b, c, d - (a * ox + b * oy + c * oz)]


def _shift_frustum(frustum, origin):
    return [_shift(plane, origin) for plane in frustum]


def _frustum_neg_z(fov_y_deg=90.0, aspect=1.0, near=0.1, far=50.0):
    """相机在原点、朝 -Z 的 6 个裁剪面（世界坐标）。"""
    t = math.tan(math.radians(fov_y_deg) / 2.0)
    r = t * aspect
    return [
        _unit([-1.0, 0.0, -r, 0.0]),      # 右：x <= -r*z
        _unit([1.0, 0.0, -r, 0.0]),       # 左：x >= r*z
        _unit([0.0, -1.0, -t, 0.0]),      # 上：y <= -t*z
        _unit([0.0, 1.0, -t, 0.0]),       # 下：y >= t*z
        _unit([0.0, 0.0, -1.0, -near]),   # near：z <= -near
        _unit([0.0, 0.0, 1.0, far]),      # far：z >= -far
    ]


def _frustum_pos_z(fov_y_deg=90.0, aspect=1.0, near=0.1, far=50.0):
    """相机在原点、朝 +Z 的 6 个裁剪面（世界坐标）。"""
    t = math.tan(math.radians(fov_y_deg) / 2.0)
    r = t * aspect
    return [
        _unit([-1.0, 0.0, r, 0.0]),       # 右：x <= r*z
        _unit([1.0, 0.0, r, 0.0]),        # 左：x >= -r*z
        _unit([0.0, -1.0, t, 0.0]),       # 上：y <= t*z
        _unit([0.0, 1.0, t, 0.0]),        # 下：y >= -t*z
        _unit([0.0, 0.0, 1.0, -near]),    # near：z >= near
        _unit([0.0, 0.0, -1.0, far]),     # far：z <= far
    ]


def _camera(room=0, position=(0.0, 0.0, 0.0), facing="neg_z",
            fov_y_deg=90.0, max_depth=2):
    if facing == "neg_z":
        forward = (0.0, 0.0, -1.0)
        frustum = _shift_frustum(_frustum_neg_z(fov_y_deg=fov_y_deg), position)
    elif facing == "pos_z":
        forward = (0.0, 0.0, 1.0)
        frustum = _shift_frustum(_frustum_pos_z(fov_y_deg=fov_y_deg), position)
    else:
        raise ValueError("facing 只能是 'neg_z' 或 'pos_z'")
    return {
        "room": room,
        "position": position,
        "forward": forward,
        "up": (0.0, 1.0, 0.0),
        "fov_y_deg": fov_y_deg,
        "aspect": 1.0,
        "frustum": frustum,
        "max_depth": max_depth,
    }


def _room(min_x, min_y, min_z, max_x, max_y, max_z):
    return {"aabb": (min_x, min_y, min_z, max_x, max_y, max_z)}


def _door(src, dst, z, half=1.0):
    """z 平面上的门，法线指向 +Z（from 房间在 +Z 一侧）。"""
    return {
        "from": src,
        "to": dst,
        "quad": [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ],
    }


def _door_reversed(src, dst, z, half=1.0):
    """z 平面上的门，法线指向 -Z（from 房间在 -Z 一侧）。"""
    return {
        "from": src,
        "to": dst,
        "quad": [
            [-half, -half, z],
            [-half, half, z],
            [half, half, z],
            [half, -half, z],
        ],
    }


def _inside(planes, point):
    x, y, z = point
    return all(
        a * x + b * y + c * z + d >= -EPS for a, b, c, d in planes
    )


def _single_room_scene(room_id=0):
    return {
        "rooms": {room_id: _room(-2.0, -2.0, -6.0, 2.0, 2.0, -4.0)},
        "portals": [],
    }


def _cycle_scene():
    """两个房间互为出口的环形传送门场景。"""
    return {
        "rooms": {
            0: _room(-2.0, -2.0, -4.0, 2.0, 2.0, 0.0),
            1: _room(-2.0, -2.0, -10.0, 2.0, 2.0, -4.0),
        },
        "portals": [
            _door(0, 1, -4.0),
            _door_reversed(1, 0, -4.0),
        ],
    }


def _backface_scene():
    """1 -> 0 的门法线朝 +Z；位于 1 号房间的相机会看到门的背面。"""
    return {
        "rooms": {
            1: _room(-2.0, -2.0, -10.0, 2.0, 2.0, -5.0),
            0: _room(1.4, -0.3, -3.0, 1.8, 0.3, -2.0),
        },
        "portals": [_door(1, 0, -5.0)],
    }


def _corridor_scene(depth=4):
    """沿 -Z 串联的走廊：0 -> 1 -> ... -> depth。"""
    rooms = {}
    portals = []
    for i in range(depth + 1):
        rooms[i] = _room(-2.0, -2.0, -4.0 * (i + 1), 2.0, 2.0, -4.0 * i)
        if i < depth:
            portals.append(_door(i, i + 1, -4.0 * (i + 1)))
    return {"rooms": rooms, "portals": portals}


# ---------------------------------------------------------------------------
# 10 个缺陷各一条
# ---------------------------------------------------------------------------


def test_defect_01_portal_frustum_built_from_portal_quad():
    camera = _camera()
    portal = _door(0, 1, -5.0)
    planes = portal_frustum(camera, portal["quad"], 0.1, 50.0)

    assert len(planes) == 6
    assert _inside(planes, (0.0, 0.0, -5.0)) is True
    # 在主相机 90 度视场内、但不在门内，必须被门裁剪掉
    assert _inside(planes, (3.0, 0.0, -5.0)) is False
    assert _inside(planes, (0.0, 0.0, -80.0)) is False


def test_defect_02_clip_polygon_returns_empty_for_degenerate():
    plane = [1.0, 0.0, 0.0, 0.0]  # 保留 x >= 0

    inside = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
    assert len(clip_polygon(inside, plane)) == 4

    outside = [
        [-3.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 2.0, 0.0],
        [-3.0, 2.0, 0.0],
    ]
    assert clip_polygon(outside, plane) == []

    # 刚好贴在裁剪面上，退化成一个零面积线段，必须返回空
    tangent = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [-1.0, 2.0, 0.0],
    ]
    assert clip_polygon(tangent, plane) == []


def test_defect_03_traverse_depth_cap_breaks_cycle():
    scene = _cycle_scene()
    camera = _camera(room=0, position=(0.0, 0.0, -1.0), max_depth=3)

    result = traverse(scene, camera, 3)

    assert isinstance(result, dict)
    assert set(result) == {"visible", "stages"}
    assert result["stages"][0] == camera["frustum"]
    assert len(result["stages"]) <= 4  # 深度 0..3
    assert len(result["visible"]) <= 2


def test_defect_04_merge_frusta_deduplicates():
    camera = _camera()
    portal = _door(0, 1, -5.0)
    child = portal_frustum(camera, portal["quad"], 0.1, 50.0)

    once = merge_frusta([child])
    thrice = merge_frusta([child, child, child])

    assert len(once) == len(child)
    assert thrice == once
    assert len(thrice) == len({tuple(plane) for plane in thrice})


def test_defect_05_room_bounds_missing_raises():
    scene = _single_room_scene()

    assert room_bounds(0, scene) == (-2.0, -2.0, -6.0, 2.0, 2.0, -4.0)

    with pytest.raises(PortalError):
        room_bounds("missing", scene)

    broken = {"rooms": {0: {"aabb": (0.0, 0.0, 0.0)}}, "portals": []}
    with pytest.raises(PortalError):
        room_bounds(0, broken)


def test_defect_06_cull_objects_keeps_straddling_object():
    plane = [[1.0, 0.0, 0.0, -0.5]]  # 保留 x >= 0.5
    objects = [
        {"id": 0, "aabb": (-4.0, -1.0, -1.0, -3.0, 1.0, 1.0)},  # 整体在外
        {"id": 1, "aabb": (-1.0, -1.0, -1.0, 0.6, 1.0, 1.0)},    # 中心在外、跨边界
        {"id": 2, "aabb": (1.0, -1.0, -1.0, 2.0, 1.0, 1.0)},     # 整体在内
    ]

    assert cull_objects(objects, plane) == [1, 2]


def test_defect_07_loop_guard_checks_entire_path():
    assert loop_guard(["A", "B"], "A", 1) is False
    assert loop_guard(["A", "B", "C", "B"], "B", 1) is False
    assert loop_guard(["A", "A"], "A", 2) is False
    assert loop_guard(["A", "B"], "C", 1) is True


def test_defect_08_traverse_rejects_portal_backface():
    scene = _backface_scene()
    camera = _camera(
        room=1,
        position=(0.0, 0.0, -8.0),
        facing="pos_z",
        fov_y_deg=20.0,
        max_depth=2,
    )

    result = traverse(scene, camera, 2)

    # 门背面不可穿越，且 0 号房间本来就在窄视锥外
    assert result["visible"] == [1]
    assert len(result["stages"]) == 1


def test_defect_09_visibility_batch_isolates_scenes():
    scene_a = _cycle_scene()
    camera_a = _camera(room=0, position=(0.0, 0.0, -1.0), max_depth=2)
    scene_b = _single_room_scene(room_id=5)
    camera_b = _camera(room=5, position=(0.0, 0.0, -1.0), max_depth=2)

    batched = visibility_batch([scene_a, scene_b], [camera_a, camera_b])

    assert batched[0] == traverse(scene_a, camera_a, 2)
    assert batched[1] == traverse(scene_b, camera_b, 2)


def test_defect_10_traverse_rejects_zero_max_depth():
    scene = _single_room_scene()

    with pytest.raises(PortalError):
        traverse(scene, _camera(), 0)

    with pytest.raises(PortalError):
        traverse(scene, _camera(), -1)


# ---------------------------------------------------------------------------
# 5 条组合验收
# ---------------------------------------------------------------------------


def test_combination_single_room_equals_main_frustum():
    scene = _single_room_scene()
    camera = _camera()
    room_objects = [
        {"id": room_id, "aabb": room["aabb"]}
        for room_id, room in scene["rooms"].items()
    ]

    expected = cull_objects(room_objects, camera["frustum"])
    result = traverse(scene, camera, 2)

    assert expected == [0]
    assert result["visible"] == expected


def test_combination_batch_rolls_back_on_zero_max_depth():
    scene = _single_room_scene()

    with pytest.raises(PortalError):
        visibility_batch([scene, scene], [_camera(max_depth=2), _camera(max_depth=0)])


def test_combination_empty_scene_returns_empty():
    result = traverse({"rooms": {}, "portals": []}, _camera(room="missing"), 2)
    assert result == {"visible": [], "stages": []}


def test_combination_plane_count_grows_linearly():
    scene = _corridor_scene(depth=4)
    camera = _camera(room=0, position=(0.0, 0.0, -1.0), max_depth=5)

    result = traverse(scene, camera, 5)
    stages = result["stages"]

    assert 1 <= len(stages) <= 6
    for depth, planes in enumerate(stages):
        # 每层只允许最多新增 6 个面，且层内无重复；未去重会指数膨胀
        assert len(planes) <= 6 * (depth + 1)
        assert len(merge_frusta([planes])) == len(planes)


def test_combination_batch_preserves_order():
    scenes = [_single_room_scene(0), _single_room_scene(5), _single_room_scene(9)]
    cameras = [
        _camera(room=0, max_depth=2),
        _camera(room=5, max_depth=2),
        _camera(room=9, max_depth=2),
    ]

    result = visibility_batch(scenes, cameras)

    assert list(result.keys()) == [0, 1, 2]
    for index, (scene, camera) in enumerate(zip(scenes, cameras)):
        assert result[index] == traverse(scene, camera, camera["max_depth"])
