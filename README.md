# render-portal-culling

## 题面

室内关卡改用传送门剔除后，隔壁房间要么全黑要么把墙后也画出来：传送门穿越后仍直接复用主相机视锥，不按传送门裁剪，隔壁房间全被剔除；裁剪算法不做退化判断，空多边形参与后续运算；穿越不设深度上限，环形传送门无限递归；合并裁剪面不去重，面数指数增长；房间 AABB 缺失返回 (0,0,0)，物体被误剔；剔除只测中心点，贴边物体闪烁；环形检测只记一步，A→B→A 仍死循环；传送门背面也穿越，看到墙后；批量共用裁剪面缓冲，场景之间串；max_depth=0 不校验，返回空可见集。portal/visibility.py 是核心，tests/test_visibility.py 15个红。要修的是：…

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

当前是初始失败环境：恰好 `15 failed`，且不得出现 collection / import / syntax error。修好 `portal/visibility.py` 后应为 `15 passed`（耗时因机器而异）。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 目录结构

```
render-portal-culling/
├── README.md
├── conftest.py
├── .gitignore
├── portal/
│   ├── __init__.py
│   └── visibility.py
└── tests/
    └── test_visibility.py
```

## 修复约束

- `tests/test_visibility.py` 不可修改，只允许改 `portal/visibility.py`。
- 仅使用 Python 标准库 `math`、`collections` 与 `pytest`；离线、无随机。
- 8 个公开函数签名保持不变，异常统一使用 `PortalError`。
- 所有几何比较容差为 `1e-6`；场景形如 `{"rooms": {id: {"aabb"}}, "portals": [{"from", "to", "quad"}]}`。
