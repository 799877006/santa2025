"""
Standalone dimer constructor + quick visualization.

构造规则（与你需求一致）：
- Tree A 角度 = -90°（向右），Tree B 角度 = +90°（向左）
- 接触条件：A 树冠尖端(index0: (0,0.8)) 接触 B 的第三层最宽点(index1: (-0.35, 0.0))
           B 树冠尖端       接触 A 的第三层最宽点
- 解析位移：d = R(ang_a)*p_tip - R(ang_b)*p_l3，并加 1e-12 余量。

运行：
    python dimer_viz.py            # 弹出 matplotlib 窗口
    python dimer_viz.py --save dimer.png
"""

import argparse
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon

# --- 基础树形（与 sa_runner/dimer_tiling 相同） ---
TREE_POINTS: List[Tuple[float, float]] = [
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
]


def rot_vec(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    x, y = v
    return (x * c - y * s, x * s + y * c)


def build_tree(cx: float, cy: float, deg: float) -> Polygon:
    p = Polygon(TREE_POINTS)
    p = affinity.rotate(p, deg, origin=(0.0, 0.0), use_radians=False)
    p = affinity.translate(p, xoff=cx, yoff=cy)
    return p


def build_dimer():
    ang_a = -90.0
    ang_b = 90.0
    p_tip = (0.0, 0.8)
    # 第三层最宽点（左侧 -0.35, 0.0）
    p_l3 = (-0.35, 0.0)
    a_tip = rot_vec(p_tip, ang_a)
    b_l3 = rot_vec(p_l3, ang_b)
    dx = a_tip[0] - b_l3[0] + 1e-12
    dy = a_tip[1] - b_l3[1]
    A = build_tree(0.0, 0.0, ang_a)
    B = build_tree(dx, dy, ang_b)
    return A, B, dx, dy


def plot_dimer(save_path: str | None = None):
    A, B, dx, dy = build_dimer()
    fig, ax = plt.subplots(figsize=(6, 6))
    for poly, color, label in [(A, "#5b5bd6", "A (-90°)"), (B, "#e07a9b", "B (+90°)")]:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, facecolor=color, edgecolor="k", linewidth=1.5)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.plot(cx, cy, "o", color="darkgreen")
        ax.text(cx, cy, label, fontsize=10, ha="center", va="center")

    # 显示接触点（A tip / B 第三层最宽点）
    ang_a = -90.0
    ang_b = 90.0
    a_tip_world = rot_vec((0.0, 0.8), ang_a)
    b_l3_world = (rot_vec((0.125, 0.5), ang_b)[0] + dx, rot_vec((0.125, 0.5), ang_b)[1] + dy)
    ax.plot([a_tip_world[0]], [a_tip_world[1]], "o", color="red", label="A tip")
    ax.plot([b_l3_world[0]], [b_l3_world[1]], "o", color="blue", label="B layer3 point")

    # AABB 边界可视化
    minx, miny, maxx, maxy = A.bounds
    minx = min(minx, B.bounds[0])
    miny = min(miny, B.bounds[1])
    maxx = max(maxx, B.bounds[2])
    maxy = max(maxy, B.bounds[3])
    ax.add_patch(
        plt.Rectangle(
            (minx, miny), maxx - minx, maxy - miny, fill=False, ls="--", color="crimson", lw=1.2
        )
    )

    ax.set_aspect("equal", "box")
    ax.set_title(f"Dimer (dx={dx:.6f}, dy={dy:.6f})\nA tip ↔ B layer3 endpoint")
    ax.legend(loc="upper right")
    ax.set_xlim(minx - 0.5, maxx + 0.5)
    ax.set_ylim(miny - 0.5, maxy + 0.5)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None, help="保存图片路径，不填则直接显示")
    args = ap.parse_args()
    plot_dimer(args.save)


if __name__ == "__main__":
    main()
