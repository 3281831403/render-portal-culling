"""传送门可见性剔除（portal visibility culling）核心实现。

数据模型
========
场景 ``scene``::

    {
        "rooms": {
            room_id: {"aabb": (min_x, min_y, min_z, max_x, max_y, max_z)},
            ...
        },
        "portals": [
            {
                "from": room_id,   # 传送门所在（观察者一侧）房间
                "to": room_id,     # 穿越后到达的房间
                "quad": [[x, y, z], [x, y, z], [x, y, z], [x, y, z]],
            },
            ...
        ],
    }

- ``room_id`` 为可哈希的整数或字符串；同一房间的 AABB 必须满足各轴
  ``min <= max``。
- ``quad`` 为传送门四角，从 ``from`` 房间一侧观察按逆时针给出，其前向法线
  ``n = (q1 - q0) x (q2 - q0)`` 指向 ``from`` 房间（即指向观察者）。
  只有位于传送门正面的观察者才能穿过：``n . (p - q0) > 1e-6``。

相机 ``camera``::

    {
        "room": room_id,               # 相机所在房间
        "position": (x, y, z),
        "forward": (x, y, z),          # 单位向量
        "up": (x, y, z),               # 单位向量，与 forward 正交
        "fov_y_deg": 60.0,             # 垂直视场角（度）
        "aspect": 16.0 / 9.0,          # 宽高比
        "frustum": [[a, b, c, d], ...],# 主相机裁剪面（6 个）
        "max_depth": 2,                # 可选，visibility_batch 使用的穿越深度
    }

裁剪面约定
==========
平面统一表示为 ``[a, b, c, d]``，内部（保留侧）满足::

    a * x + b * y + c * z + d >= -EPS

法线 ``(a, b, c)`` 归一化为单位长度。所有几何比较使用容差
``EPS = 1e-6``，小于该容差视为共面、重合或退化。
"""

import math
from collections import deque
from collections.abc import Sequence

EPS = 1e-6

# 穿越传送门时为子视锥选择的 near / far。主相机自身的距离范围已由
# camera["frustum"] 表达，子视锥只需要覆盖“透过该门”的开放锥体。
_TRAVERSE_NEAR = 1e-4
_TRAVERSE_FAR = 1e9


class PortalError(Exception):
    """传送门剔除相关的输入 / 场景 / 几何非法。"""


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------


def _is_real(value):
    """真正的数值（``bool`` 是 ``int`` 子类，但不应当作坐标或深度）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value):
    """把数值转成 float；非法时返回 ``None``。"""
    if not _is_real(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value, what="点"):
    """校验三元坐标，返回 ``(x, y, z)`` 浮点元组。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PortalError("%s 必须是三个数值" % what)
    if len(value) != 3:
        raise PortalError("%s 必须是三个数值" % what)
    coords = tuple(_number(component) for component in value)
    if any(component is None for component in coords):
        raise PortalError("%s 必须是有限数值" % what)
    return coords


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize_plane(plane):
    """归一化平面；法线长度过小或分量非法时返回 ``None``。"""
    if isinstance(plane, (str, bytes)) or not isinstance(plane, Sequence):
        return None
    if len(plane) != 4:
        return None
    values = [_number(component) for component in plane]
    if any(component is None for component in values):
        return None
    a, b, c, d = values
    length = math.sqrt(a * a + b * b + c * c)
    if length <= EPS:
        return None
    return [a / length, b / length, c / length, d / length]


def _newell_area(polygon):
    """Newell 法计算多边形面积（三维，取法线与面积大小）。"""
    nx = ny = nz = 0.0
    count = len(polygon)
    for index in range(count):
        x0, y0, z0 = polygon[index]
        x1, y1, z1 = polygon[(index + 1) % count]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _aabb_corners(aabb):
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    return [
        (x, y, z)
        for x in (min_x, max_x)
        for y in (min_y, max_y)
        for z in (min_z, max_z)
    ]


def _quad_normal(quad):
    """按 ``n = (q1 - q0) x (q2 - q0)`` 计算未归一化法线。"""
    q0, q1, q2 = quad[0], quad[1], quad[2]
    return _cross(_sub(q1, q0), _sub(q2, q0))


def _validate_quad(quad):
    """校验传送门四边形，返回四个点；退化时返回 ``None``。"""
    if isinstance(quad, (str, bytes)) or not isinstance(quad, Sequence):
        return None
    if len(quad) < 3:
        return None
    points = []
    vertices = quad[:4] if len(quad) >= 4 else quad
    for vertex in vertices:
        try:
            points.append(_point(vertex, "传送门顶点"))
        except PortalError:
            return None
    if len(points) < 3:
        return None
    area_vector = _quad_normal(points)
    if _length(area_vector) <= EPS:
        return None
    return points


def _validate_rooms(scene):
    """取出并校验 ``scene["rooms"]``，返回房间字典。"""
    if not isinstance(scene, dict):
        raise PortalError("scene 必须是字典")
    rooms = scene.get("rooms")
    if not isinstance(rooms, dict):
        raise PortalError("scene 缺少 rooms")
    return rooms


def _validate_portals(scene):
    portals = scene.get("portals", [])
    if portals is None:
        return []
    if isinstance(portals, (str, bytes)) or not isinstance(portals, Sequence):
        raise PortalError("scene['portals'] 必须是列表")
    return list(portals)


def _validate_camera(camera):
    if not isinstance(camera, dict):
        raise PortalError("camera 必须是字典")
    for key in ("room", "position", "forward", "frustum"):
        if key not in camera:
            raise PortalError("camera 缺少 %s" % key)
    _point(camera["position"], "camera.position")
    _point(camera["forward"], "camera.forward")
    frustum = camera["frustum"]
    if isinstance(frustum, (str, bytes)) or not isinstance(frustum, Sequence):
        raise PortalError("camera.frustum 必须是列表")
    return camera


def _validate_max_depth(max_depth):
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise PortalError("max_depth 必须是整数")
    if max_depth < 1:
        raise PortalError("max_depth 必须 >= 1")
    return max_depth


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def portal_frustum(camera, portal_quad, near, far):
    """由视锥与传送门生成穿过传送门后的裁剪面。

    从 ``camera["position"]`` 向 ``portal_quad`` 的四条边张成 4 个侧面，
    再加上垂直于 ``camera["forward"]`` 的 near、far 两个面，共 6 个面。
    必须按传送门裁剪：禁止直接返回或复用 ``camera["frustum"]``，
    否则隔壁房间会被整间误剔。

    参数：
        camera: 相机字典，至少含 ``position`` 与 ``forward``（单位向量）。
        portal_quad: 传送门四角，4 个 ``[x, y, z]``，从 from 房间一侧看为
            逆时针（法线指向观察者）。
        near, far: 裁剪距离，必须满足 ``0 < near < far``。

    返回：
        ``[[a, b, c, d], ...]``，6 个归一化裁剪面，内部满足
        ``a*x + b*y + c*z + d >= -1e-6``。返回的平面必须随传送门四边形
        变化，并只覆盖"透过该门"可见的锥体。
        四边形退化（顶点不足 3 个、面积 <= 1e-6）或 near/far 非法时返回 ``[]``。
    """
    if not isinstance(camera, dict):
        return []
    eye_value = camera.get("position")
    forward_value = camera.get("forward")
    try:
        eye = _point(eye_value, "camera.position")
        forward = _point(forward_value, "camera.forward")
    except PortalError:
        return []

    near_value = _number(near)
    far_value = _number(far)
    if near_value is None or far_value is None:
        return []
    if not (0.0 < near_value < far_value):
        return []

    quad = _validate_quad(portal_quad)
    if quad is None:
        return []

    planes = []

    def add_plane(normal, point_on_plane):
        length = _length(normal)
        if length <= EPS:
            return
        a = normal[0] / length
        b = normal[1] / length
        c = normal[2] / length
        d = -_dot((a, b, c), point_on_plane)
        planes.append([a, b, c, d])

    # 四个侧面：眼睛与 quad 每条边组成平面，内部为朝向门中心的一侧。
    center = (
        sum(vertex[0] for vertex in quad) / len(quad),
        sum(vertex[1] for vertex in quad) / len(quad),
        sum(vertex[2] for vertex in quad) / len(quad),
    )
    center_direction = _sub(center, eye)
    for index in range(len(quad)):
        vertex = quad[index]
        following = quad[(index + 1) % len(quad)]
        edge = _sub(following, vertex)
        to_start = _sub(vertex, eye)
        # 同时垂直于 edge 与 (vertex - eye)，因此该平面包含眼睛与这条边。
        normal = _cross(edge, to_start)
        if _dot(normal, center_direction) < 0.0:
            normal = (-normal[0], -normal[1], -normal[2])
        add_plane(normal, vertex)

    # near / far：垂直于 camera["forward"]，分别朝向门后与门后远处。
    forward_length = _length(forward)
    if forward_length <= EPS:
        return []
    unit_forward = (
        forward[0] / forward_length,
        forward[1] / forward_length,
        forward[2] / forward_length,
    )
    add_plane(unit_forward, (eye[0] + unit_forward[0] * near_value,
                             eye[1] + unit_forward[1] * near_value,
                             eye[2] + unit_forward[2] * near_value))
    add_plane((-unit_forward[0], -unit_forward[1], -unit_forward[2]),
              (eye[0] + unit_forward[0] * far_value,
               eye[1] + unit_forward[1] * far_value,
               eye[2] + unit_forward[2] * far_value))

    if len(planes) != 6:
        return []
    return planes


def clip_polygon(polygon, plane):
    """Sutherland-Hodgman 单平面裁剪。

    ``polygon`` 为 ``[[x, y, z], ...]``（顶点序）；``plane`` 为
    ``[a, b, c, d]``，内部满足 ``a*x + b*y + c*z + d >= -1e-6``。
    顶点在容差内视为内部；出入边交点按参数线性插值；输出保持原顶点顺序。

    必须处理退化：结果顶点少于 3 个，或按 Newell 法计算的面积 <= 1e-6 时
    返回 ``[]``，绝不能让空 / 退化多边形继续参与后续运算。
    """
    if not isinstance(polygon, (list, tuple)):
        return []
    if len(polygon) < 3:
        return []
    try:
        vertices = [tuple(_point(vertex, "多边形顶点")) for vertex in polygon]
    except PortalError:
        return []
    plane_values = _normalize_plane(plane)
    if plane_values is None:
        return []
    a, b, c, d = plane_values

    def signed(point):
        return a * point[0] + b * point[1] + c * point[2] + d

    output = []
    count = len(vertices)
    for index in range(count):
        current = vertices[index]
        following = vertices[(index + 1) % count]
        current_value = signed(current)
        following_value = signed(following)
        current_inside = current_value >= -EPS
        following_inside = following_value >= -EPS

        if current_inside:
            output.append(current)
            if not following_inside:
                denominator = current_value - following_value
                if abs(denominator) > EPS:
                    t = current_value / denominator
                    output.append((
                        current[0] + t * (following[0] - current[0]),
                        current[1] + t * (following[1] - current[1]),
                        current[2] + t * (following[2] - current[2]),
                    ))
        elif following_inside:
            denominator = current_value - following_value
            if abs(denominator) > EPS:
                t = current_value / denominator
                output.append((
                    current[0] + t * (following[0] - current[0]),
                    current[1] + t * (following[1] - current[1]),
                    current[2] + t * (following[2] - current[2]),
                ))

    if len(output) < 3:
        return []
    if _newell_area(output) <= EPS:
        return []
    return [list(vertex) for vertex in output]


def traverse(scene, camera, max_depth):
    """BFS 穿越传送门，返回 ``{"visible": [...], "stages": [...]}``。

    - 从 ``camera["room"]`` 出发：深度 0 的可见性由 ``camera["frustum"]``
      与房间 AABB 决定，``stages[0]`` 即该主视锥。
    - 逐层穿过位于当前视锥内、且处于正面的传送门，用 ``portal_frustum``
      生成子视锥，再用 ``merge_frusta`` 与父级合并去重。
    - ``max_depth`` 必须为整数且 ``>= 1``，否则抛 ``PortalError``；
      ``max_depth=0`` 或负数不得静默返回空可见集。
    - 只穿正面：``n . (视点 - q0) <= 1e-6`` 的传送门直接跳过，
      禁止从背面穿越（否则会看到墙后）。
    - 环形保护：同一路径上房间出现次数由 ``loop_guard(path, room, 1)``
      限制，同时受 ``max_depth`` 截断；环形传送门不得无限递归。
    - ``visible``：可达房间 id 的升序列表，无重复；``stages[d]``：第 d 层
      合并去重后的裁剪面列表（没有房间的深度不占位）。
    - 空场景（``rooms`` 为空）返回 ``{"visible": [], "stages": []}``。
    """
    _validate_max_depth(max_depth)
    _validate_camera(camera)
    rooms = _validate_rooms(scene)
    portals = _validate_portals(scene)

    if not rooms:
        return {"visible": [], "stages": []}

    start_room = camera["room"]
    if start_room not in rooms:
        raise PortalError("相机所在房间 %r 不存在" % (start_room,))

    # 所有房间的 AABB 先校验，缺失 / 非法时立即失败。
    bounds = {room_id: room_bounds(room_id, scene) for room_id in rooms}

    frustum = [_normalize_plane(plane) for plane in camera["frustum"]]
    frustum = [plane for plane in frustum if plane is not None]

    objects = [{"id": room_id, "aabb": bounds[room_id]} for room_id in rooms]

    visible = set()
    if start_room in cull_objects(objects, frustum):
        visible.add(start_room)

    # stages[0] 就是主相机视锥本身；后续各层才累积传送门生成的子视锥。
    stages = [list(camera["frustum"])]

    # 队列元素：((room_id, 合并后的裁剪面, 路径), 已有深度)
    queue = deque([((start_room, list(frustum), [start_room]), 0)])

    while queue:
        (room_id, room_frustum, path), depth = queue.popleft()
        if depth >= max_depth:
            continue

        for portal in portals:
            if not isinstance(portal, dict):
                raise PortalError("传送门必须是字典")
            if portal.get("from") != room_id:
                continue
            destination = portal.get("to")
            if destination not in rooms:
                raise PortalError("传送门指向的房间 %r 不存在" % (destination,))

            quad = _validate_quad(portal.get("quad"))
            if quad is None:
                # 退化传送门无法裁剪，跳过而不是让它污染后续运算。
                continue

            normal = _quad_normal(quad)
            normal_length = _length(normal)
            if normal_length <= EPS:
                continue
            unit_normal = (normal[0] / normal_length,
                           normal[1] / normal_length,
                           normal[2] / normal_length)
            eye = _point(camera["position"], "camera.position")
            # 只穿正面：观察者必须在门的前侧。
            if _dot(unit_normal, _sub(eye, quad[0])) <= EPS:
                continue

            # 传送门四边形必须落在当前层视锥内，否则透过它看不到任何东西。
            if not _polygon_inside(room_frustum, quad):
                continue
            if not loop_guard(path, destination, 1):
                continue

            child_frustum = portal_frustum(
                camera, [list(vertex) for vertex in quad],
                _TRAVERSE_NEAR, _TRAVERSE_FAR,
            )
            if not child_frustum:
                continue

            merged = merge_frusta([room_frustum, child_frustum])
            new_depth = depth + 1
            new_path = path + [destination]

            if destination not in visible:
                if _aabb_inside(merged, bounds[destination]):
                    visible.add(destination)

            if len(stages) <= new_depth:
                stages.append(list(merged))
            else:
                stages[new_depth] = merge_frusta([stages[new_depth], merged])

            if new_depth < max_depth:
                queue.append(((destination, merged, new_path), new_depth))

    return {"visible": _sorted_rooms(visible), "stages": stages}


def _polygon_inside(frustum, polygon):
    """多边形是否与所有裁剪面内侧有非退化交集（逐面裁剪）。"""
    current = [list(vertex) for vertex in polygon]
    for plane in frustum:
        current = clip_polygon(current, plane)
        if not current:
            return False
    return True


def _aabb_inside(frustum, aabb):
    """AABB 是否可能位于所有裁剪面内侧（8 角点判定，与 cull_objects 一致）。"""
    if not frustum:
        return True
    for plane in frustum:
        a, b, c, d = plane
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        if (
            a * min_x + b * min_y + c * min_z + d < -EPS
            and a * max_x + b * min_y + c * min_z + d < -EPS
            and a * min_x + b * max_y + c * min_z + d < -EPS
            and a * max_x + b * max_y + c * min_z + d < -EPS
            and a * min_x + b * min_y + c * max_z + d < -EPS
            and a * max_x + b * min_y + c * max_z + d < -EPS
            and a * min_x + b * max_y + c * max_z + d < -EPS
            and a * max_x + b * max_y + c * max_z + d < -EPS
        ):
            return False
    return True


def _sorted_rooms(room_ids):
    """房间 id 升序；混合类型无法直接比较时退回稳定字符串序。"""
    try:
        return sorted(room_ids)
    except TypeError:
        return sorted(room_ids, key=lambda value: (type(value).__name__, str(value)))


def merge_frusta(frusta):
    """合并同级裁剪面并去重。

    ``frusta`` 为若干裁剪面列表，例如
    ``[parent_frustum, child_frustum]``。把所有平面展平后按首次出现顺序保留：

    - 法线归一化，长度为 0 的非法面丢弃；
    - 与已保留面满足 ``|a1-a2|, |b1-b2|, |c1-c2|, |d1-d2| <= 1e-6``
      视为同一面，只保留第一个；
    - 近似反向但半空间不同的面不视为重复。

    空输入返回 ``[]``。重复输入不得让面数增长：同一组面重复传入的结果
    必须与只传一次相同，避免面数指数增长。
    """
    if frusta is None:
        return []
    if isinstance(frusta, (str, bytes)) or not isinstance(frusta, Sequence):
        return []

    merged = []
    for frustum in frusta:
        if frustum is None:
            continue
        if isinstance(frustum, (str, bytes)) or not isinstance(frustum, Sequence):
            continue
        for plane in frustum:
            normalized = _normalize_plane(plane)
            if normalized is None:
                continue
            duplicate = False
            for kept in merged:
                if all(
                    abs(normalized[index] - kept[index]) <= EPS
                    for index in range(4)
                ):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(normalized)
    return merged


def room_bounds(room_id, scene):
    """返回房间 AABB ``(min_x, min_y, min_z, max_x, max_y, max_z)``。

    从 ``scene["rooms"][room_id]["aabb"]`` 读取并校验：长度必须为 6、
    六个分量有限、各轴 ``min <= max``，否则抛 ``PortalError``。
    房间不存在时抛 ``PortalError``；禁止用 ``(0, 0, 0)`` 之类的默认值兜底，
    否则缺失房间里的物体会被误剔。
    """
    rooms = _validate_rooms(scene)
    try:
        exists = room_id in rooms
    except TypeError:
        raise PortalError("房间 id 必须可哈希") from None
    if not exists:
        raise PortalError("房间 %r 不存在" % (room_id,))

    room = rooms[room_id]
    if not isinstance(room, dict) or "aabb" not in room:
        raise PortalError("房间 %r 缺少 aabb" % (room_id,))
    aabb = room["aabb"]
    if isinstance(aabb, (str, bytes)) or not isinstance(aabb, Sequence):
        raise PortalError("房间 %r 的 aabb 必须是 6 个数值" % (room_id,))
    if len(aabb) != 6:
        raise PortalError("房间 %r 的 aabb 长度必须是 6" % (room_id,))

    values = tuple(_number(component) for component in aabb)
    if any(component is None for component in values):
        raise PortalError("房间 %r 的 aabb 必须是有限数值" % (room_id,))

    min_x, min_y, min_z, max_x, max_y, max_z = values
    if min_x > max_x or min_y > max_y or min_z > max_z:
        raise PortalError("房间 %r 的 aabb 非法：min > max" % (room_id,))
    return values


def cull_objects(objects, frustum):
    """按裁剪面剔除物体，返回保留物体的 id 列表（保持输入顺序）。

    ``objects``：``[{"id": int, "aabb": (min_x, ..., max_x)}, ...]``。
    ``frustum``：``[[a, b, c, d], ...]``，内部
    ``a*x + b*y + c*z + d >= -1e-6``。

    判定必须使用 AABB 的 8 个角点：只有当物体整体位于某个裁剪面外侧
    （8 个角点在该面的值全部 ``< -1e-6``）时才剔除；与平面相交、贴边
    （距离 <= 1e-6）或跨越边界的物体一律保留，禁止只测中心点。
    ``frustum`` 为空视为没有裁剪面，保留全部；``objects`` 为空返回 ``[]``。
    """
    if objects is None:
        return []
    if isinstance(objects, (str, bytes)) or not isinstance(objects, Sequence):
        raise PortalError("objects 必须是列表")

    planes = []
    if frustum is not None:
        if isinstance(frustum, (str, bytes)) or not isinstance(frustum, Sequence):
            raise PortalError("frustum 必须是列表")
        for plane in frustum:
            normalized = _normalize_plane(plane)
            if normalized is not None:
                planes.append(normalized)

    kept = []
    for item in objects:
        if not isinstance(item, dict) or "id" not in item or "aabb" not in item:
            raise PortalError("物体必须含 id 与 aabb")
        aabb = item["aabb"]
        if isinstance(aabb, (str, bytes)) or not isinstance(aabb, Sequence):
            raise PortalError("物体 aabb 必须是 6 个数值")
        if len(aabb) != 6:
            raise PortalError("物体 aabb 长度必须是 6")
        values = tuple(_number(component) for component in aabb)
        if any(component is None for component in values):
            raise PortalError("物体 aabb 必须是有限数值")
        min_x, min_y, min_z, max_x, max_y, max_z = values

        outside = False
        for a, b, c, d in planes:
            all_outside = True
            for x in (min_x, max_x):
                for y in (min_y, max_y):
                    for z in (min_z, max_z):
                        if a * x + b * y + c * z + d >= -EPS:
                            all_outside = False
                            break
                    if not all_outside:
                        break
                if not all_outside:
                    break
            if all_outside:
                outside = True
                break
        if not outside:
            kept.append(item["id"])
    return kept


def loop_guard(path, room_id, max_visits):
    """环形传送门防死循环：判断能否进入 ``room_id``。

    ``path`` 为当前路径上的房间 id 序列（含起点）；``max_visits`` 为同一
    房间在一条路径上允许出现的最大次数（``>= 1``）。

    返回 ``True`` 当且仅当 ``path.count(room_id) < max_visits``。
    必须统计整条 path：只比较上一步的实现会放过 ``A -> B -> A`` 这类环。
    ``max_visits < 1``、``path`` 非序列或 ``room_id`` 不可哈希时抛
    ``PortalError``。
    """
    if isinstance(max_visits, bool) or not isinstance(max_visits, int):
        raise PortalError("max_visits 必须是整数")
    if max_visits < 1:
        raise PortalError("max_visits 必须 >= 1")
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise PortalError("path 必须是房间 id 序列")
    try:
        hash(room_id)
    except TypeError:
        raise PortalError("room_id 必须可哈希") from None

    visits = 0
    for entry in path:
        try:
            if entry == room_id:
                visits += 1
        except Exception:  # pragma: no cover - 极端自定义类型
            continue
    return visits < max_visits


def visibility_batch(scenes, cameras):
    """批量求解多个场景，任一场景非法时整体回滚。

    ``scenes`` 与 ``cameras`` 一一对应；长度不一致抛 ``PortalError``。
    每个场景独立调用 ``traverse(scene, camera, camera.get("max_depth", 2))``，
    禁止跨场景共享裁剪面缓冲 / 缓存，结果不得互相污染。
    任一场景或 camera 非法（缺 ``rooms``、AABB 非法、``max_depth < 1`` 等）
    必须抛 ``PortalError``，并且不返回部分结果、不修改入参。

    返回 ``{索引: traverse 结果}``，键按输入顺序 ``0..n-1``；
    ``scenes`` 为空返回 ``{}``。
    """
    if isinstance(scenes, (str, bytes)) or not isinstance(scenes, Sequence):
        raise PortalError("scenes 必须是序列")
    if isinstance(cameras, (str, bytes)) or not isinstance(cameras, Sequence):
        raise PortalError("cameras 必须是序列")
    if len(scenes) != len(cameras):
        raise PortalError("scenes 与 cameras 长度不一致")

    results = {}
    for index, (scene, camera) in enumerate(zip(scenes, cameras)):
        if not isinstance(camera, dict):
            raise PortalError("camera 必须是字典")
        max_depth = camera.get("max_depth", 2)
        # 每个场景独立调用，绝不共享裁剪面缓冲或中间结果。
        results[index] = traverse(scene, camera, max_depth)
    return results
