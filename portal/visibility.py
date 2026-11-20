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

EPS = 1e-6


class PortalError(Exception):
    """传送门剔除相关的输入 / 场景 / 几何非法。"""


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
    raise NotImplementedError


def clip_polygon(polygon, plane):
    """Sutherland-Hodgman 单平面裁剪。

    ``polygon`` 为 ``[[x, y, z], ...]``（顶点序）；``plane`` 为
    ``[a, b, c, d]``，内部满足 ``a*x + b*y + c*z + d >= -1e-6``。
    顶点在容差内视为内部；出入边交点按参数线性插值；输出保持原顶点顺序。

    必须处理退化：结果顶点少于 3 个，或按 Newell 法计算的面积 <= 1e-6 时
    返回 ``[]``，绝不能让空 / 退化多边形继续参与后续运算。
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def room_bounds(room_id, scene):
    """返回房间 AABB ``(min_x, min_y, min_z, max_x, max_y, max_z)``。

    从 ``scene["rooms"][room_id]["aabb"]`` 读取并校验：长度必须为 6、
    六个分量有限、各轴 ``min <= max``，否则抛 ``PortalError``。
    房间不存在时抛 ``PortalError``；禁止用 ``(0, 0, 0)`` 之类的默认值兜底，
    否则缺失房间里的物体会被误剔。
    """
    raise NotImplementedError


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
    raise NotImplementedError


def loop_guard(path, room_id, max_visits):
    """环形传送门防死循环：判断能否进入 ``room_id``。

    ``path`` 为当前路径上的房间 id 序列（含起点）；``max_visits`` 为同一
    房间在一条路径上允许出现的最大次数（``>= 1``）。

    返回 ``True`` 当且仅当 ``path.count(room_id) < max_visits``。
    必须统计整条 path：只比较上一步的实现会放过 ``A -> B -> A`` 这类环。
    ``max_visits < 1``、``path`` 非序列或 ``room_id`` 不可哈希时抛
    ``PortalError``。
    """
    raise NotImplementedError


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
    raise NotImplementedError
