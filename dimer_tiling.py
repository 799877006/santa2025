"""
dimer_tiling.py

目标：用 Shapely 自动计算“2聚体(二聚体)”的最紧密相切位移与晶格向量 (vx, vy)，再进行平铺输出 CSV。

核心约束（默认）：
- 碰撞判定使用 Shapely：允许 touches（接触），禁止有面积的重叠（intersection area < tol）。
- 晶格向量由代码推导，不依赖硬编码 1.075 之类常数。
- 输出坐标最终整体平移，使包络中心位于 (0,0)。

用法：
  python dimer_tiling.py input.csv output.csv

可选参数：
  --area-tol 1e-9
  --dx-tol 1e-6
  --dy-range 1.0          # dy 搜索范围：[-dy_range, +dy_range]
  --dy-step 0.005         # dy 粗搜步长
  --refine-steps 2        # 细化次数
  --max-tries 3           # 边界处理时单体尝试次数
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import concurrent.futures
import os

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree


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


def strip_s(v) -> float:
    s = str(v).strip()
    return float(s[1:] if s.startswith("s") else s)


def fmt_s(v: float) -> str:
    # 与 sa_runner.cpp 的 fmt 类似：固定 18 位
    return "s" + f"{v:.18f}"


def build_tree_poly(cx: float, cy: float, deg: float, base_poly: Polygon) -> Polygon:
    p = affinity.rotate(base_poly, deg, origin=(0.0, 0.0), use_radians=False)
    return affinity.translate(p, xoff=cx, yoff=cy)


def union_bounds(polys: List[Polygon]) -> Tuple[float, float, float, float]:
    if not polys:
        return (0.0, 0.0, 0.0, 0.0)
    minx, miny, maxx, maxy = polys[0].bounds
    for p in polys[1:]:
        x1, y1, x2, y2 = p.bounds
        minx = min(minx, x1)
        miny = min(miny, y1)
        maxx = max(maxx, x2)
        maxy = max(maxy, y2)
    return (minx, miny, maxx, maxy)


def side_len_from_bounds(b: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = b
    return max(maxx - minx, maxy - miny)


def intersection_area(a, b) -> float:
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0
    return float(inter.area)

def rot_vec(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    x, y = v
    return (x * c - y * s, x * s + y * c)


@dataclass
class Dimer:
    """A fixed-orientation dimer: (A at origin, B at (dx,dy))."""

    poly_a: Polygon
    poly_b0: Polygon
    dx: float
    dy: float

    def poly_b(self) -> Polygon:
        return affinity.translate(self.poly_b0, xoff=self.dx, yoff=self.dy)

    def union(self):
        return unary_union([self.poly_a, self.poly_b()])


class DimerConfig:
    def __init__(
        self,
        area_tol: float = 1e-9,
        dx_tol: float = 1e-6,
        dy_range: float = 1.0,
        dy_step: float = 0.005,
        refine_steps: int = 2,
    ):
        self.area_tol = float(area_tol)
        self.dx_tol = float(dx_tol)
        self.dy_range = float(dy_range)
        self.dy_step = float(dy_step)
        self.refine_steps = int(refine_steps)

        self.base_poly = Polygon(TREE_POINTS)

        # 固定 A/B 的角度（按你的要求）
        self.ang_a = -90.0
        self.ang_b = 90.0

        self._poly_a0 = build_tree_poly(0.0, 0.0, self.ang_a, self.base_poly)
        self._poly_b0 = build_tree_poly(0.0, 0.0, self.ang_b, self.base_poly)

    def find_optimal_dimer(self) -> Dimer:
        """
        解析构造“树冠尖端 ↔ 第三层最宽点”首首相接的二聚体：
        - Tree A: angle = -90°（向右）
        - Tree B: angle = +90°（向左）
        接触条件：
          A 的树冠尖端 (index0: (0,0.8)) 对应 B 的第三层最宽点 (index1: (-0.35,0.0))
          B 的树冠尖端 对应 A 的第三层最宽点
        由第一条接触方程即可解出 d = Cb - Ca：
          d = R(ang_a) * p_tip - R(ang_b) * p_l3
        其中 p_tip=(0,0.8), p_l3=(-0.35,0.0), ang_a=-90, ang_b=+90
        """
        p_tip = (0.0, 0.8)
        p_l3 = (-0.35, 0.0)  # 第三层最宽点（左侧 -0.35, 0.0）
        a_tip = rot_vec(p_tip, self.ang_a)
        b_l3 = rot_vec(p_l3, self.ang_b)
        dx = a_tip[0] - b_l3[0]
        dy = a_tip[1] - b_l3[1]
        dx += 1e-12  # 极小余量，避免浮点误差导致面积重叠

        dimer = Dimer(self._poly_a0, self._poly_b0, dx, dy)
        if intersection_area(dimer.poly_a, dimer.poly_b()) > self.area_tol:
            raise RuntimeError(f"Analytic dimer still overlaps. dx={dx}, dy={dy}")
        return dimer

    def find_lattice_vectors(self, dimer: Dimer) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        计算最紧密晶格：
        - vx：把第二个 dimer 放在右侧，沿 x 向左滑到刚好不重叠。
        - vy：把第二个 dimer 放在上方，允许 x 交错 offset_x ∈ {0,0.125,0.25}，沿 y 向下滑到刚好不重叠。
        """
        d0 = dimer.union()
        minx, miny, maxx, maxy = d0.bounds
        w = maxx - minx
        h = maxy - miny

        def ok_shift(dx: float, dy: float) -> bool:
            d1 = affinity.translate(d0, xoff=dx, yoff=dy)
            return intersection_area(d0, d1) <= self.area_tol

        def min_dx(dy: float, hi0: float) -> float:
            lo = 0.0
            hi = hi0
            for _ in range(30):
                if ok_shift(hi, dy):
                    break
                hi *= 1.5
            # 二分找最小 dx
            if ok_shift(lo, dy):
                return lo
            while (hi - lo) > self.dx_tol:
                mid = 0.5 * (lo + hi)
                if ok_shift(mid, dy):
                    hi = mid
                else:
                    lo = mid
            return hi

        # vx: dy=0
        vx_dx = min_dx(dy=0.0, hi0=max(0.5, w))
        vx = (vx_dx + 1e-12, 0.0)

        # vy: dx in {0,0.125,0.25}, minimize dy
        offsets_x = [0.0, 0.125, 0.25]
        best_vy: Tuple[float, float] | None = None

        for ox in offsets_x:
            # 找最小 dy 使得 ok_shift(ox,dy)=True
            lo = 0.0
            hi = max(0.5, h)
            for _ in range(30):
                if ok_shift(ox, hi):
                    break
                hi *= 1.5
                if hi > 100.0:
                    hi = None
                    break
            if hi is None:
                continue

            if ok_shift(ox, lo):
                dy_opt = lo
            else:
                while (hi - lo) > self.dx_tol:
                    mid = 0.5 * (lo + hi)
                    if ok_shift(ox, mid):
                        hi = mid
                    else:
                        lo = mid
                dy_opt = hi

            cand = (ox, dy_opt + 1e-12)
            if (best_vy is None) or (cand[1] < best_vy[1]):
                best_vy = cand

        if best_vy is None:
            raise RuntimeError("Failed to find vy with staggering offsets.")

        vy = best_vy
        return vx, vy


def validate_no_overlaps(polys: List[Polygon], area_tol: float) -> bool:
    if not polys:
        return True
    st = STRtree(polys)
    id_to_i = {id(p): i for i, p in enumerate(polys)}
    for i, p in enumerate(polys):
        cands = st.query(p)
        for cand in cands:
            if hasattr(cand, "geom_type"):
                j = id_to_i.get(id(cand), -1)
                if j <= i:
                    continue
                q = cand
            else:
                j = int(cand)
                if j <= i:
                    continue
                q = polys[j]
            if p.intersects(q):
                a = intersection_area(p, q)
                if a > area_tol:
                    return False
    return True


def _poly_fully_inside(poly: Polygon, bbox_poly: Polygon, area_tol: float) -> bool:
    # 允许 touches：只要 poly 与 bbox 的交面积 ≈ poly.area 即可
    ia = intersection_area(poly, bbox_poly)
    return ia >= (poly.area - area_tol)


def _try_place(
    poly: Polygon,
    placed_polys: List[Polygon],
    bbox_poly: Polygon | None,
    area_tol: float,
) -> bool:
    if bbox_poly is not None:
        if not _poly_fully_inside(poly, bbox_poly, area_tol):
            return False
    for q in placed_polys:
        if poly.intersects(q) and intersection_area(poly, q) > area_tol:
            return False
    return True


def _gen_lattice_anchors(
    bbox_bounds: Tuple[float, float, float, float],
    vx: Tuple[float, float],
    vy: Tuple[float, float],
    origin: Tuple[float, float],
    max_extra: int = 2,
) -> List[Tuple[float, float]]:
    """
    生成覆盖 bbox 的晶格锚点（锚点是 Tree A 的中心）。
    """
    minx, miny, maxx, maxy = bbox_bounds
    W = maxx - minx
    H = maxy - miny
    vx_len = max(1e-9, abs(vx[0]) + abs(vx[1]))
    vy_len = max(1e-9, abs(vy[0]) + abs(vy[1]))
    cols = int(math.ceil(W / vx_len)) + 1 + max_extra
    rows = int(math.ceil(H / vy_len)) + 1 + max_extra
    ox, oy = origin
    anchors: List[Tuple[float, float]] = []
    for r in range(-max_extra, rows + max_extra):
        for c in range(-max_extra, cols + max_extra):
            ax = ox + c * vx[0] + r * vy[0]
            ay = oy + c * vx[1] + r * vy[1]
            anchors.append((ax, ay))
    # 先从左下开始填（更稳定）
    anchors.sort(key=lambda p: (p[1], p[0]))
    return anchors


def solve_tiling(
    df: pd.DataFrame,
    cfg: DimerConfig,
    dimer: Dimer,
    vx: Tuple[float, float],
    vy: Tuple[float, float],
    max_tries: int = 3,
    enforce_input_bbox: bool = True,
    allow_fallback_unconstrained: bool = True,
    base_phi: float = 0.0,  # 额外整体旋转角，用于失败后尝试不同方向的主晶格
    relax_factor: float = 0.0,  # 允许 side_len 放宽比例（用来扩展可用正方形框）
) -> pd.DataFrame:
    ids = df["id"].astype(str).tolist()
    n = len(ids)

    # 原始包络：按多边形 bounds
    base = cfg.base_poly
    polys_in = [build_tree_poly(strip_s(x), strip_s(y), strip_s(d), base) for x, y, d in zip(df["x"], df["y"], df["deg"])]
    b0 = union_bounds(polys_in)
    minx, miny, maxx, maxy = b0
    orig_side = side_len_from_bounds(b0)
    if orig_side <= 0:
        raise RuntimeError("Invalid original bbox side.")
    # 计算放宽后的目标正方形框：以原 bbox 中心为中心，边长 = orig_side*(1+relax_factor)
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    target_side = orig_side * (1.0 + relax_factor)
    half = 0.5 * target_side
    bbox = box(cx - half, cy - half, cx + half, cy + half)
    orig_side = side_len_from_bounds(b0)
    relaxed_side = target_side

    # --- 三阶段贪心：先主二聚体晶格，后换角度二聚体，再随机塞单体 ---
    bbox_poly = bbox if enforce_input_bbox else None
    bbox_bounds = bbox.bounds

    placed_rows: List[Tuple[str, float, float, float]] = []
    placed_polys: List[Polygon] = []
    used = 0

    # 固定角二聚体的基准多边形（已旋转好）
    pa0 = affinity.rotate(dimer.poly_a, base_phi, origin=(0.0, 0.0), use_radians=False)
    pb0 = affinity.rotate(dimer.poly_b0, base_phi, origin=(0.0, 0.0), use_radians=False)
    d0 = rot_vec((dimer.dx, dimer.dy), base_phi)

    def place_dimer_at(ax: float, ay: float, ang_a: float, ang_b: float, dxy: Tuple[float, float]) -> bool:
        nonlocal used
        if used + 1 >= n:
            return False
        dx, dy = dxy

        # A
        poly_a = affinity.rotate(cfg.base_poly, ang_a, origin=(0.0, 0.0), use_radians=False)
        poly_a = affinity.translate(poly_a, xoff=ax, yoff=ay)
        if not _try_place(poly_a, placed_polys, bbox_poly, cfg.area_tol):
            return False

        # B
        bx = ax + dx
        by = ay + dy
        poly_b = affinity.rotate(cfg.base_poly, ang_b, origin=(0.0, 0.0), use_radians=False)
        poly_b = affinity.translate(poly_b, xoff=bx, yoff=by)
        if not _try_place(poly_b, placed_polys, bbox_poly, cfg.area_tol):
            return False

        # commit
        placed_rows.append((ids[used], ax, ay, ang_a))
        placed_polys.append(poly_a)
        placed_rows.append((ids[used + 1], bx, by, ang_b))
        placed_polys.append(poly_b)
        used += 2
        return True

    def place_single_at(ax: float, ay: float, ang: float) -> bool:
        nonlocal used
        if used >= n:
            return False
        poly = affinity.rotate(cfg.base_poly, ang, origin=(0.0, 0.0), use_radians=False)
        poly = affinity.translate(poly, xoff=ax, yoff=ay)
        if not _try_place(poly, placed_polys, bbox_poly, cfg.area_tol):
            return False
        placed_rows.append((ids[used], ax, ay, ang))
        placed_polys.append(poly)
        used += 1
        return True

    # Stage 1: 用主角度(0°)的晶格尽量塞满二聚体（尝试少量原点偏移以提高可放数量）
    origin_base = (float(bbox_bounds[0]), float(bbox_bounds[1]))
    origin_offsets = [0.0]  # 从紧贴左下角开始，不做额外偏移
    best_plan: Tuple[int, Tuple[float, float], List[Tuple[float, float]]] | None = None

    for ox in origin_offsets:
        for oy in origin_offsets:
            org = (origin_base[0] + ox * vx[0] + oy * vy[0], origin_base[1] + ox * vx[1] + oy * vy[1])
            anchors = _gen_lattice_anchors(bbox_bounds, vx, vy, org, max_extra=2)
            # dry-run count only
            tmp_polys: List[Polygon] = []
            cnt = 0
            tused = 0
            for ax, ay in anchors:
                if tused + 1 >= n:
                    break
                # build A/B using pre-rotated polygons for speed (pa0/pb0) + translate
                poly_a = affinity.translate(pa0, xoff=ax, yoff=ay)
                if not _try_place(poly_a, tmp_polys, bbox_poly, cfg.area_tol):
                    continue
                poly_b = affinity.translate(pb0, xoff=ax + d0[0], yoff=ay + d0[1])
                if not _try_place(poly_b, tmp_polys, bbox_poly, cfg.area_tol):
                    continue
                tmp_polys.append(poly_a)
                tmp_polys.append(poly_b)
                cnt += 1
                tused += 2
            if (best_plan is None) or (cnt > best_plan[0]):
                best_plan = (cnt, org, anchors)

    if best_plan is None:
        best_plan = (0, origin_base, _gen_lattice_anchors(bbox_bounds, vx, vy, origin_base, max_extra=2))

    _, _, anchors1 = best_plan
    for ax, ay in anchors1:
        if used + 1 >= n:
            break
        # 使用 pre-rotated pa0/pb0
        poly_a = affinity.translate(pa0, xoff=ax, yoff=ay)
        if not _try_place(poly_a, placed_polys, bbox_poly, cfg.area_tol):
            continue
        poly_b = affinity.translate(pb0, xoff=ax + d0[0], yoff=ay + d0[1])
        if not _try_place(poly_b, placed_polys, bbox_poly, cfg.area_tol):
            continue
        placed_rows.append((ids[used], ax, ay, cfg.ang_a))
        placed_polys.append(poly_a)
        placed_rows.append((ids[used + 1], ax + d0[0], ay + d0[1], cfg.ang_b))
        placed_polys.append(poly_b)
        used += 2

    # Stage 2: 剩余尽量用“换角度”的二聚体塞（phi ∈ {90,60,45,30}，相对 base_phi 再旋转）
    # 这里用严格的旋转：晶格向量与 dimer 位移都做同角旋转（保持几何相切与周期结构）
    phis = [90.0, 60.0, 45.0, 30.0]
    for phi in phis:
        if used + 1 >= n:
            break
        vx_p = rot_vec(vx, phi)
        vy_p = rot_vec(vy, phi)
        d_p = rot_vec(d0, phi)
        ang_a = cfg.ang_a + base_phi + phi
        ang_b = cfg.ang_b + base_phi + phi

        anchors2 = _gen_lattice_anchors(bbox_bounds, vx_p, vy_p, origin_base, max_extra=2)
        for ax, ay in anchors2:
            if used + 1 >= n:
                break
            # 直接构造（旋转后的树）
            if place_dimer_at(ax, ay, ang_a, ang_b, d_p):
                continue

    # Stage 3a: 还剩下的先尝试“回填原解位置”（对 n=1 或 bbox 很小特别重要）
    # 这一步是确定性的：按原 CSV 顺序，把没放进去的树尽量放回原坐标
    if used < n:
        orig_x = [strip_s(v) for v in df["x"].tolist()]
        orig_y = [strip_s(v) for v in df["y"].tolist()]
        orig_deg = [strip_s(v) for v in df["deg"].tolist()]
        i = used
        while i < n:
            if place_single_at(orig_x[i], orig_y[i], orig_deg[i]):
                i = used
                continue
            # 放不回原位就跳过，留给随机阶段
            i += 1

    # Stage 3b: 还剩下的用随机采样在 bbox 内塞单体（角度在 {ang_a, ang_b} 及其旋转集合中取）
    if used < n:
        if bbox_poly is None:
            # 无 bbox 约束时，直接在原点附近扩展采样
            minx2, miny2, maxx2, maxy2 = -100.0, -100.0, 100.0, 100.0
        else:
            minx2, miny2, maxx2, maxy2 = bbox_bounds

        # 候选角度：基准与旋转
        cand_angles: List[float] = [cfg.ang_a, cfg.ang_b]
        for phi in phis:
            cand_angles.extend([cfg.ang_a + phi, cfg.ang_b + phi])

        # 可复现伪随机（用 group_id 的 hash）
        seed = 1469598103934665603
        gid = str(df["id"].iloc[0]).split("_", 1)[0]
        for ch in gid.encode("utf-8"):
            seed ^= ch
            seed *= 1099511628211
            seed &= (1 << 64) - 1
        rnd = seed

        def rng01() -> float:
            nonlocal rnd
            # xorshift64*
            rnd ^= (rnd >> 12) & ((1 << 64) - 1)
            rnd ^= (rnd << 25) & ((1 << 64) - 1)
            rnd ^= (rnd >> 27) & ((1 << 64) - 1)
            val = (rnd * 2685821657736338717) & ((1 << 64) - 1)
            return ((val >> 11) & ((1 << 53) - 1)) / float(1 << 53)

        max_attempts = max(5000, 2000 * (n - used))
        attempts = 0
        while used < n and attempts < max_attempts:
            attempts += 1
            x = minx2 + (maxx2 - minx2) * rng01()
            y = miny2 + (maxy2 - miny2) * rng01()
            ang = cand_angles[int(rng01() * len(cand_angles)) % len(cand_angles)]
            if place_single_at(x, y, ang):
                continue

    # 如果仍然放不完：按开关决定是否允许“扩框随便塞”
    if used < n:
        if allow_fallback_unconstrained:
            # 在更大的范围内随机补全（以放宽后的正方形为中心，额外放大一圈）
            extra = max(relaxed_side, 10.0)
            minx2, miny2, maxx2, maxy2 = cx - extra, cy - extra, cx + extra, cy + extra
            max_attempts = max(30000, 4000 * (n - used))
            attempts = 0
            while used < n and attempts < max_attempts:
                attempts += 1
                x = minx2 + (maxx2 - minx2) * rng01()
                y = miny2 + (maxy2 - miny2) * rng01()
                ang = cand_angles[int(rng01() * len(cand_angles)) % len(cand_angles)]
                if place_single_at(x, y, ang):
                    continue
        if used < n:
            raise RuntimeError(f"Failed to place all trees using staged strategy within bbox: placed={used} total={n}")

    # 整体居中到 (0,0)
    final_polys = []
    for cid, cx, cy, deg in placed_rows:
        final_polys.append(build_tree_poly(cx, cy, deg, cfg.base_poly))
    fb = union_bounds(final_polys)
    fcx = 0.5 * (fb[0] + fb[2])
    fcy = 0.5 * (fb[1] + fb[3])

    out = []
    for cid, cx, cy, deg in placed_rows:
        out.append((cid, cx - fcx, cy - fcy, deg))

    out_df = pd.DataFrame(out, columns=["id", "x_f", "y_f", "deg_f"])
    out_df["x"] = out_df["x_f"].map(fmt_s)
    out_df["y"] = out_df["y_f"].map(fmt_s)
    out_df["deg"] = out_df["deg_f"].map(fmt_s)
    out_df = out_df[["id", "x", "y", "deg"]]

    # 最终校验（可选但默认做一次）
    polys_out = [build_tree_poly(strip_s(x), strip_s(y), strip_s(d), cfg.base_poly) for x, y, d in zip(out_df["x"], out_df["y"], out_df["deg"])]
    if not validate_no_overlaps(polys_out, cfg.area_tol):
        raise RuntimeError("Output has overlaps (area > tol).")

    # side_len 不得超过 relaxed_side
    new_b = union_bounds(polys_out)
    new_side = side_len_from_bounds(new_b)
    if new_side > relaxed_side + 1e-9:
        raise RuntimeError(f"side_len exceeded relaxed limit: new={new_side} limit={relaxed_side}")

    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    # 其余参数固定在代码里，避免每次命令行填写
    # area_tol = 1e-9, dx_tol = 1e-6, dy_range = 1.0, dy_step = 0.01, refine_steps = 1, max_tries = 3
    ap.add_argument("--global-bbox", action="store_true", help="Treat entire CSV as one big bbox (legacy). Default: per group_id bbox.")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if not {"id", "x", "y", "deg"}.issubset(df.columns):
        raise SystemExit("CSV must contain columns: id,x,y,deg")

    cfg = DimerConfig(
        area_tol=1e-9,
        dx_tol=1e-6,
        dy_range=1.0,
        dy_step=0.01,   # 固定为 0.01
        refine_steps=1, # 固定为 1
    )

    # 只计算一次 dimer 与晶格向量（全局通用）
    dimer = cfg.find_optimal_dimer()
    vx, vy = cfg.find_lattice_vectors(dimer)
    print(f"Calculated Lattice: vx={vx}, vy={vy}")

    # 默认不允许越界扩框：必须在输入 bbox 内完成放置
    allow_fallback = False

    if args.global_bbox:
        out_df = solve_tiling(
            
            df,
            cfg,
            dimer=dimer,
            vx=vx,
            vy=vy,
            max_tries=3,
            enforce_input_bbox=True,
            allow_fallback_unconstrained=allow_fallback,
        )
        out_df.to_csv(args.output_csv, index=False)
    else:
        # 默认：按 group_id 分组分别平铺（更符合你之前的评分方式）
        work = df.copy()
        work["group_id"] = work["id"].astype(str).str.split("_", n=1, expand=True)[0]
        work["item_id"] = work["id"].astype(str).str.split("_", n=1, expand=True)[1].astype(int)

        out_parts = []
        groups = list(work.groupby("group_id", sort=False))

        def process_group(item):
            gid, g = item
            g = g.sort_values("item_id").reset_index(drop=True)
            angles_to_try = [0.0, 90.0, 60.0, 45.0, 30.0]
            tree_side = 1.0
            orig_side = side_len_from_bounds(union_bounds([build_tree_poly(strip_s(x), strip_s(y), strip_s(d), cfg.base_poly) for x,y,d in zip(g["x"], g["y"], g["deg"])]))
            max_relax = max(0.0, tree_side / max(1e-9, orig_side))
            relax_list = [0.0]
            step = 0.01
            r = step
            while r < max_relax + 1e-9:
                relax_list.append(r)
                r += step

            for relax_factor in relax_list:
                for base_phi in angles_to_try:
                    try:
                        vx_r = rot_vec(vx, base_phi)
                        vy_r = rot_vec(vy, base_phi)
                        out_g = solve_tiling(
                            g[["id", "x", "y", "deg"]],
                            cfg,
                            dimer=dimer,
                            vx=vx_r,
                            vy=vy_r,
                            max_tries=3,
                            enforce_input_bbox=True,
                            allow_fallback_unconstrained=True,
                            base_phi=base_phi,
                            relax_factor=relax_factor,
                        )
                        if base_phi != 0.0 or relax_factor > 0.0:
                            print(f"[info] gid={gid} succeeded with base_phi={base_phi}, relax={relax_factor}")
                        return out_g
                    except Exception:
                        continue
            raise RuntimeError(f"gid={gid} cannot be placed even after angles+relax up to one-tree side")

        max_workers = max(1, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_group, item) for item in groups]
            for fut in concurrent.futures.as_completed(futures):
                out_parts.append(fut.result())

        out_df = pd.concat(out_parts, axis=0, ignore_index=True)
        out_df.to_csv(args.output_csv, index=False)

    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()


