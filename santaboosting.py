import argparse
from decimal import Decimal, getcontext
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
import time
from typing import Dict, List, Optional
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import affinity
from shapely.strtree import STRtree

# 提高字符串到 Decimal 的精度，避免读写 csv 反复损失
getcontext().prec = 50

# 配置 JAX 使用 64 位精度以提高几何计算稳定性 (可选)
jax.config.update("jax_enable_x64", True)

# ==========================================
# 配置: 指定要优化的单个组 (设为 None 则优化所有组)
# ==========================================
SINGLE_GROUP_ID: Optional[int] = 25 # 例如设置为 8 只优化 group 8

# ==========================================
# 1. 几何定义：15 顶点圣诞树
# ==========================================

def get_tree_constants(scale_factor=1.0):
    """
    根据用户提供的 ChristmasTree 类定义生成 JAX 静态数据
    返回:
        vertices: (15, 2) 多边形顶点 (逆时针)
        samples: (M, 2)   内部采样点 (传感器)
    """
    # 原始参数
    trunk_w = 0.15; trunk_h = 0.2
    base_w = 0.7; base_y = 0.0
    mid_w = 0.4; tier_2_y = 0.25
    top_w = 0.25; tier_1_y = 0.5
    tip_y = 0.8; trunk_bottom_y = -trunk_h

    # 定义顶点 (必须逆时针顺序，以便 winding number 计算正确)
    # 你的原始定义看起来是顺时针或混合的，这里为了 JAX 物理引擎的稳定性，
    # 我整理为标准的逆时针 (CCW) 顺序：从树尖开始向左下走
    coords = [
        (0.0, tip_y),                   # Top Tip
        (-top_w / 4, tier_1_y),         # Top Left Inner
        (-top_w / 2, tier_1_y),         # Top Left Outer
        (-mid_w / 4, tier_2_y),         # Mid Left Inner
        (-mid_w / 2, tier_2_y),         # Mid Left Outer
        (-base_w / 2, base_y),          # Base Left Outer
        (-trunk_w / 2, base_y),         # Trunk Left Top
        (-trunk_w / 2, trunk_bottom_y), # Trunk Left Bottom
        (trunk_w / 2, trunk_bottom_y),  # Trunk Right Bottom
        (trunk_w / 2, base_y),          # Trunk Right Top
        (base_w / 2, base_y),           # Base Right Outer
        (mid_w / 2, tier_2_y),          # Mid Right Outer
        (mid_w / 4, tier_2_y),          # Mid Right Inner
        (top_w / 2, tier_1_y),          # Top Right Outer
        (top_w / 4, tier_1_y),          # Top Right Inner
    ]
    
    vertices = np.array(coords, dtype=np.float64) * scale_factor

    # 生成采样点 (用于检测碰撞的传感器)
    # 使用线段密集采样方案：沿每条边进行 raymarching 式采样
    sample_points = []
    
    # 1. 包含所有顶点（稍微内缩以避免精确边界点）
    for cx, cy in coords:
        # 向中心方向微移以避免精确边界
        sample_points.append((cx * 0.99, cy * 0.99))
    
    # 2. 沿每条边密集采样（每条边 5 个点，避免太密集）
    num_samples_per_edge = 5
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])
        for t in np.linspace(0.1, 0.9, num_samples_per_edge):
            pt = p1 * (1-t) + p2 * t
            # 稍微向内收缩
            sample_points.append(tuple(pt * 0.98))
    
    # 3. 内部骨架点 (Y轴) - 用于检测内部穿透
    spine_ys = np.linspace(trunk_bottom_y + 0.02, tip_y - 0.02, 10)
    for y in spine_ys:
        sample_points.append((0.0, y))
    
    # 4. 左右骨架点 - 形成内部网格
    for y in np.linspace(-0.1, 0.5, 6):
        # 根据 y 位置计算合理的 x 范围
        if y < 0:  # 树干区域
            xs = [-0.04, 0.04]
        elif y < 0.25:  # 底层
            xs = [-0.2, -0.1, 0.1, 0.2]
        elif y < 0.5:  # 中层
            xs = [-0.1, 0.1]
        else:  # 上层
            xs = [-0.05, 0.05]
        for x in xs:
            sample_points.append((x, y))
    
    samples = np.array(sample_points, dtype=np.float64) * scale_factor
    
    return jnp.array(vertices), jnp.array(samples)

# 初始化常量数据
TREE_VERTS, TREE_SAMPLES = get_tree_constants()

# ==========================================
# 2. JAX 物理引擎 (可微解析几何)
# ==========================================

@jit
def transform_points(points, x, y, theta):
    """批量刚体变换: points(M,2) -> transformed(M,2)"""
    c, s = jnp.cos(theta), jnp.sin(theta)
    rot = jnp.array([[c, -s], [s, c]])
    return jnp.dot(points, rot.T) + jnp.array([x, y])

@jit
def point_segment_dist_sq(px, py, v1, v2):
    """计算点 P 到线段 V1-V2 的最小距离平方"""
    x1, y1 = v1
    x2, y2 = v2
    dx, dy = x2 - x1, y2 - y1
    
    # 投影参数 t
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy + 1e-12)
    t = jnp.clip(t, 0.0, 1.0) # 限制在线段内
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return (px - closest_x)**2 + (py - closest_y)**2

@jit
def winding_number_contribution(px, py, v1, v2):
    """计算单条边对 Winding Number 的贡献 (利用 arctan2)"""
    x1, y1 = v1
    x2, y2 = v2
    
    # 防止数值不稳定：当点太接近顶点时，添加小扰动
    eps = 1e-10
    dx1 = x1 - px + eps
    dy1 = y1 - py + eps
    dx2 = x2 - px + eps
    dy2 = y2 - py + eps
    
    # 计算点 P 看线段 V1V2 的张角
    cross = dx1 * dy2 - dy1 * dx2
    dot = dx1 * dx2 + dy1 * dy2
    
    # 使用安全的 arctan2
    return jnp.arctan2(cross, dot + eps)

@jit
def analytical_sdf(px, py, vertices):
    """
    解析法 SDF 核心: 计算点到多边形的有向距离
    """
    # 构造边: v[i] -> v[i+1]
    v_curr = vertices
    v_next = jnp.roll(vertices, -1, axis=0)
    
    # 1. 距离场: 到所有边的最小距离
    dists_sq = vmap(point_segment_dist_sq, in_axes=(None, None, 0, 0))(
        px, py, v_curr, v_next
    )
    min_dist = jnp.sqrt(jnp.min(dists_sq) + 1e-12)
    
    # 2. 符号场: Winding Number
    angles = vmap(winding_number_contribution, in_axes=(None, None, 0, 0))(
        px, py, v_curr, v_next
    )
    total_angle = jnp.sum(angles)
    
    # 平滑符号函数 (Soft Sign): 
    # 内部 total_angle ≈ 2pi, 外部 ≈ 0
    # Sigmoid 让梯度在边界处平滑过渡
    is_inside = jax.nn.sigmoid(10.0 * (jnp.abs(total_angle) - 3.0))
    
    # 内部为负 (-1)，外部为正 (+1)
    sign = 1.0 - 2.0 * is_inside
    return min_dist * sign


@jit
def segment_sdf_query(p1x, p1y, p2x, p2y, vertices, num_samples=10):
    """
    线段 SDF 查询：沿着线段 P1-P2 进行 raymarching 式采样
    返回线段上所有采样点的最小 SDF 值
    
    如果返回值 < 0，说明线段穿过了多边形
    """
    # 生成沿线段的采样点
    t_values = jnp.linspace(0.0, 1.0, num_samples)
    
    def sample_sdf(t):
        px = p1x + t * (p2x - p1x)
        py = p1y + t * (p2y - p1y)
        return analytical_sdf(px, py, vertices)
    
    # 计算所有采样点的 SDF
    sdfs = vmap(sample_sdf)(t_values)
    
    # 返回最小 SDF（如果 < 0，说明线段穿过多边形）
    return jnp.min(sdfs)


@jit
def compute_edge_penetration_energy(edge_p1, edge_p2, vertices, num_samples=8):
    """
    计算一条边穿透多边形的能量
    沿边进行 raymarching 采样，对所有负 SDF 值求平方和
    """
    t_values = jnp.linspace(0.0, 1.0, num_samples)
    
    def sample_energy(t):
        px = edge_p1[0] + t * (edge_p2[0] - edge_p1[0])
        py = edge_p1[1] + t * (edge_p2[1] - edge_p1[1])
        sdf = analytical_sdf(px, py, vertices)
        return jnp.maximum(0.0, -sdf) ** 2
    
    energies = vmap(sample_energy)(t_values)
    return jnp.sum(energies)


# ==========================================
# 3. 能量函数 (Loss Function)
# ==========================================

@jit
def compute_total_loss(params, boosting_weights, box_size, vertices_ref, samples_ref):
    """
    params: (N, 3) -> [x, y, theta]
    boosting_weights: (N, N)
    增强版：除了采样点检测，还包含边线段的 Raymarching SDF 检测
    """
    N = params.shape[0]
    num_verts = vertices_ref.shape[0]
    
    # --- A. 边界约束 (Container Wall) ---
    # 将每棵树的采样点变换到世界坐标
    # world_samples shape: (N, num_samples, 2)
    world_samples = vmap(transform_points, in_axes=(None, 0, 0, 0))(
        samples_ref, params[:, 0], params[:, 1], params[:, 2]
    )
    
    # 将每棵树的顶点变换到世界坐标（用于边检测）
    # world_verts shape: (N, num_verts, 2)
    world_verts = vmap(transform_points, in_axes=(None, 0, 0, 0))(
        vertices_ref, params[:, 0], params[:, 1], params[:, 2]
    )
    
    limit = box_size / 2.0
    # 简单的 Box 惩罚: max(0, |p| - limit)^2
    loss_bound = jnp.sum(jnp.maximum(0.0, jnp.abs(world_samples) - limit)**2)
    
    # --- B. 互斥约束 (Overlap) ---
    
    # 定义计算 "树 i 和 树 j" 重叠能量的函数
    def compute_pair_energy(idx_i, idx_j):
        # JAX 技巧：jnp.where 会评估两个分支。
        # 我们必须确保 idx_j 在计算时不等于 idx_i 才能彻底避开 arctan2(0,0) 的 NaN。
        safe_idx_j = jnp.where(idx_i == idx_j, (idx_i + 1) % N, idx_j)

        # 获取采样点和参考坐标
        pts_i_world = world_samples[idx_i]
        jx, jy, jtheta = params[safe_idx_j]

        # 坐标变换：世界坐标 -> 树 j 的局部坐标
        c, s = jnp.cos(-jtheta), jnp.sin(-jtheta)
        inv_rot = jnp.array([[c, -s], [s, c]])
        
        pts_rel = pts_i_world - jnp.array([jx, jy])
        pts_in_j = jnp.dot(pts_rel, inv_rot)

        # Part 1: 采样点 SDF 穿透能量
        sdfs = vmap(analytical_sdf, in_axes=(0, 0, None))(
            pts_in_j[:, 0], pts_in_j[:, 1], vertices_ref
        )
        sample_energy = jnp.sum(jnp.maximum(0.0, -sdfs)**2)
        
        # Part 2: 边线段 Raymarching SDF 穿透能量
        # 获取树 i 的顶点在世界坐标系下
        verts_i_world = world_verts[idx_i]
        verts_i_next = jnp.roll(verts_i_world, -1, axis=0)  # 下一个顶点
        
        # 将顶点变换到树 j 的局部坐标
        verts_rel = verts_i_world - jnp.array([jx, jy])
        verts_in_j = jnp.dot(verts_rel, inv_rot)
        verts_next_rel = verts_i_next - jnp.array([jx, jy])
        verts_next_in_j = jnp.dot(verts_next_rel, inv_rot)
        
        # 对每条边进行线段 SDF 查询
        def edge_penetration(v1, v2):
            """计算一条边穿透多边形 j 的能量"""
            # 沿边采样 8 个点
            t_values = jnp.linspace(0.1, 0.9, 8)
            def sample_sdf(t):
                px = v1[0] + t * (v2[0] - v1[0])
                py = v1[1] + t * (v2[1] - v1[1])
                sdf = analytical_sdf(px, py, vertices_ref)
                return jnp.maximum(0.0, -sdf) ** 2
            return jnp.sum(vmap(sample_sdf)(t_values))
        
        edge_energy = jnp.sum(vmap(edge_penetration)(verts_in_j, verts_next_in_j))

        # 总能量 = 采样点能量 + 边能量
        total_energy = sample_energy + edge_energy

        # 如果 idx_i == idx_j，则结果归零
        return jnp.where(idx_i == idx_j, 0.0, total_energy)

    # 全矩阵并行计算 (Nested vmap)
    # 提示: 如果 N 很大 (>100)，这里应该用 lax.scan 或只算邻居，但在 N<100 时直接算最快
    indices = jnp.arange(N)
    energy_matrix = vmap(vmap(compute_pair_energy, in_axes=(None, 0)), in_axes=(0, None))(
        indices, indices
    )
    
    # 乘以 Boosting 权重
    # jnp.eye(N) 用于去除对角线 (自己撞自己)
    mask = 1.0 - jnp.eye(N)
    weighted_energy = energy_matrix * boosting_weights * mask
    
    # 总 Loss = 边界惩罚 + 重叠惩罚 + 微弱的向心引力 (防止梯度消失)
    loss_gravity = jnp.sum(params[:, :2]**2) * 0.001
    loss_overlap = jnp.sum(weighted_energy)
    return loss_bound * 10.0 + loss_overlap + loss_gravity

# 数值梯度函数：使用 Shapely 重叠面积作为损失
def compute_numerical_gradient(params_np: np.ndarray, box_size: float, eps: float = 1e-4) -> np.ndarray:
    """
    使用有限差分计算数值梯度
    params_np: (N, 3) numpy array
    返回: (N, 3) 梯度数组
    """
    N = params_np.shape[0]
    grads = np.zeros_like(params_np)
    
    # 计算当前损失
    loss_base = compute_shapely_loss_with_boundary(params_np, box_size)
    
    # 对每个参数计算数值梯度
    for i in range(N):
        for j in range(3):  # x, y, theta
            # 前向差分
            params_forward = params_np.copy()
            params_forward[i, j] += eps
            loss_forward = compute_shapely_loss_with_boundary(params_forward, box_size)
            
            # 数值梯度
            grads[i, j] = (loss_forward - loss_base) / eps
    
    return grads


# 为了兼容性，保留原来的 grad_fn 接口，但使用数值梯度
def grad_fn(params, boosting_weights, box_size, vertices_ref, samples_ref):
    """
    梯度函数接口（兼容原有调用）
    实际使用 Shapely 数值梯度
    """
    params_np = np.asarray(params)
    grads_np = compute_numerical_gradient(params_np, float(box_size))
    return jnp.array(grads_np)


def compute_loss_breakdown(params, boosting_weights, box_size, vertices_ref, samples_ref):
    """
    计算各项 Loss 的分解，用于日志输出（使用 Shapely 重叠面积）
    返回: (total_loss, overlap_loss, boundary_loss, gravity_loss)
    """
    params_np = np.asarray(params)
    
    # 使用 Shapely 计算各项损失
    overlap_area = compute_shapely_overlap_area(params_np)
    loss_overlap = overlap_area * 100.0  # 重叠面积损失（放大100倍以匹配之前的尺度）
    
    # 边界惩罚
    polygons = params_to_shapely_polygons(params_np)
    limit = float(box_size) / 2.0
    boundary_penalty = 0.0
    
    for poly in polygons:
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        if bounds[0] < -limit:
            boundary_penalty += (-limit - bounds[0]) ** 2
        if bounds[1] < -limit:
            boundary_penalty += (-limit - bounds[1]) ** 2
        if bounds[2] > limit:
            boundary_penalty += (bounds[2] - limit) ** 2
        if bounds[3] > limit:
            boundary_penalty += (bounds[3] - limit) ** 2
    
    loss_bound = boundary_penalty * 10.0
    
    # 重力损失
    loss_gravity = float(np.sum(params_np[:, :2]**2) * 0.001)
    
    # 总损失
    total_loss = loss_overlap + loss_bound + loss_gravity
    
    return total_loss, loss_overlap, loss_bound, loss_gravity

# ==========================================
# 3.5 重叠检测函数 (使用 Shapely，与 SA.py 一致)
# ==========================================

# 圣诞树的标准顶点坐标（与 SA.py 一致）
TREE_COORDS_TEMPLATE = [
    (0.0, 0.8),           # 树尖
    (0.125, 0.5),         # top_w/2
    (0.0625, 0.5),        # top_w/4
    (0.2, 0.25),          # mid_w/2
    (0.1, 0.25),          # mid_w/4
    (0.35, 0.0),          # base_w/2
    (0.075, 0.0),         # trunk_w/2
    (0.075, -0.2),        # trunk bottom right
    (-0.075, -0.2),       # trunk bottom left
    (-0.075, 0.0),        # trunk top left
    (-0.35, 0.0),         # -base_w/2
    (-0.1, 0.25),         # -mid_w/4
    (-0.2, 0.25),         # -mid_w/2
    (-0.0625, 0.5),       # -top_w/4
    (-0.125, 0.5),        # -top_w/2
]


def params_to_shapely_polygons(params_np: np.ndarray) -> List[ShapelyPolygon]:
    """
    将 params (N, 3) [x, y, theta_rad] 转换为 Shapely 多边形列表
    """
    polygons = []
    
    # 检查是否有 NaN/Inf
    if np.any(np.isnan(params_np)) or np.any(np.isinf(params_np)):
        raise ValueError(f"params_np contains NaN or Inf: nan_count={np.sum(np.isnan(params_np))}, inf_count={np.sum(np.isinf(params_np))}")
    
    for i in range(params_np.shape[0]):
        x, y, theta_rad = params_np[i]
        # 确保转换为 Python float
        x = float(x)
        y = float(y)
        theta_deg = float(np.rad2deg(theta_rad))
        
        # 检查 NaN/Inf
        if np.isnan(x) or np.isnan(y) or np.isnan(theta_deg):
            raise ValueError(f"NaN detected at i={i}: x={x}, y={y}, theta_deg={theta_deg}")
        if np.isinf(x) or np.isinf(y) or np.isinf(theta_deg):
            raise ValueError(f"Inf detected at i={i}: x={x}, y={y}, theta_deg={theta_deg}")
        
        # 创建原点处的多边形
        poly = ShapelyPolygon(TREE_COORDS_TEMPLATE)
        # 旋转（围绕原点）
        poly = affinity.rotate(poly, theta_deg, origin=(0, 0))
        # 平移到指定位置
        poly = affinity.translate(poly, xoff=x, yoff=y)
        polygons.append(poly)
    
    return polygons


def get_overlapping_pairs_shapely(params_np: np.ndarray) -> List[tuple]:
    """
    使用 Shapely 检测重叠，返回所有重叠的 (i, j) 对列表
    """
    polygons = params_to_shapely_polygons(params_np)
    
    if len(polygons) <= 1:
        return []
    
    strtree = STRtree(polygons)
    overlapping_pairs = set()
    
    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)
        
        for cand in candidates:
            if hasattr(cand, "geom_type"):
                continue  # Shapely 1.x
            else:
                j = int(cand)
                if j <= i:  # 避免重复和自己
                    continue
                other = polygons[j]
            
            # 检查是否真的相交（不只是 bbox 相交）
            if poly.intersects(other) and not poly.touches(other):
                overlapping_pairs.add((i, j))
    
    return list(overlapping_pairs)


def compute_overlap_count_shapely(params_np: np.ndarray) -> int:
    """
    使用 Shapely 的 STRtree 检测重叠（与 SA.py 的 validate_no_overlaps 一致）
    返回重叠的多边形对数
    """
    polygons = params_to_shapely_polygons(params_np)
    
    if len(polygons) <= 1:
        return 0
    
    strtree = STRtree(polygons)
    overlap_count = 0
    
    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)
        
        for cand in candidates:
            # Shapely 2.x 返回索引
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly:
                    continue
            else:
                j = int(cand)
                if j <= i:  # 避免重复计算
                    continue
                other = polygons[j]
            
            # 允许 touches（边/点接触）；若既不分离也不 touches 则判为重叠
            if (not poly.disjoint(other)) and (not poly.touches(other)):
                overlap_count += 1
    
    return overlap_count


def validate_no_overlaps_shapely(params_np: np.ndarray) -> bool:
    """
    使用 Shapely 验证是否无重叠（与 SA.py 完全一致的逻辑）
    返回 True 表示无重叠
    """
    return compute_overlap_count_shapely(params_np) == 0


def compute_shapely_overlap_area(params_np: np.ndarray) -> float:
    """
    计算所有重叠对的总重叠面积（使用 Shapely）
    返回总重叠面积
    """
    polygons = params_to_shapely_polygons(params_np)
    
    if len(polygons) <= 1:
        return 0.0
    
    strtree = STRtree(polygons)
    total_overlap_area = 0.0
    
    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)
        
        for cand in candidates:
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly:
                    continue
            else:
                j = int(cand)
                if j <= i:  # 避免重复计算
                    continue
                other = polygons[j]
            
            # 检查是否真的相交（不只是 bbox 相交）
            if poly.intersects(other) and not poly.touches(other):
                intersection = poly.intersection(other)
                if hasattr(intersection, 'area'):
                    total_overlap_area += intersection.area
    
    return total_overlap_area


def compute_shapely_loss_with_boundary(params_np: np.ndarray, box_size: float) -> float:
    """
    计算基于 Shapely 的总损失：
    - 重叠面积损失
    - 边界惩罚（树超出边界）
    - 重力损失（微弱的向心引力）
    """
    # 1. 重叠面积损失
    overlap_area = compute_shapely_overlap_area(params_np)
    
    # 2. 边界惩罚：检查所有树是否超出边界
    polygons = params_to_shapely_polygons(params_np)
    limit = box_size / 2.0
    boundary_penalty = 0.0
    
    for poly in polygons:
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        # 检查是否超出边界
        if bounds[0] < -limit:  # minx < -limit
            boundary_penalty += (-limit - bounds[0]) ** 2
        if bounds[1] < -limit:  # miny < -limit
            boundary_penalty += (-limit - bounds[1]) ** 2
        if bounds[2] > limit:  # maxx > limit
            boundary_penalty += (bounds[2] - limit) ** 2
        if bounds[3] > limit:  # maxy > limit
            boundary_penalty += (bounds[3] - limit) ** 2
    
    # 3. 重力损失（微弱的向心引力，防止梯度消失）
    gravity_loss = np.sum(params_np[:, :2]**2) * 0.001
    
    # 总损失
    total_loss = overlap_area * 100.0 + boundary_penalty * 10.0 + gravity_loss
    
    return total_loss


# 包装函数：将 JAX 数组转换为 numpy 后调用 Shapely 检测
def compute_overlap_count(params, box_size, vertices_ref, samples_ref) -> int:
    """
    重叠检测入口（兼容旧接口）
    使用 Shapely 进行精确碰撞检测
    """
    params_np = np.array(params)
    return compute_overlap_count_shapely(params_np)


@jit
def compute_overlap_count_sdf(params, box_size, vertices_ref, samples_ref):
    """
    使用 SDF 采样点检测重叠（与优化目标一致）
    返回: 重叠的采样点数量
    """
    N = params.shape[0]
    
    world_samples = vmap(transform_points, in_axes=(None, 0, 0, 0))(
        samples_ref, params[:, 0], params[:, 1], params[:, 2]
    )
    
    def compute_pair_overlap(idx_i, idx_j):
        safe_idx_j = jnp.where(idx_i == idx_j, (idx_i + 1) % N, idx_j)
        
        pts_i_world = world_samples[idx_i]
        jx, jy, jtheta = params[safe_idx_j]
        
        pts_rel = pts_i_world - jnp.array([jx, jy])
        c, s = jnp.cos(-jtheta), jnp.sin(-jtheta)
        inv_rot = jnp.array([[c, -s], [s, c]])
        pts_in_j = jnp.dot(pts_rel, inv_rot)
        
        sdfs = vmap(analytical_sdf, in_axes=(0, 0, None))(
            pts_in_j[:, 0], pts_in_j[:, 1], vertices_ref
        )
        # 使用更宽松的阈值
        overlap_count = jnp.sum(sdfs < -0.001)
        
        return jnp.where(idx_i == idx_j, 0, overlap_count)
    
    indices = jnp.arange(N)
    overlap_matrix = vmap(vmap(compute_pair_overlap, in_axes=(None, 0)), in_axes=(0, None))(
        indices, indices
    )
    
    return jnp.sum(overlap_matrix)


@jit
def check_boundary_violation(params, box_size, samples_ref):
    """检测是否有采样点超出边界"""
    world_samples = vmap(transform_points, in_axes=(None, 0, 0, 0))(
        samples_ref, params[:, 0], params[:, 1], params[:, 2]
    )
    limit = box_size / 2.0
    violations = jnp.sum(jnp.abs(world_samples) > limit)
    return violations

# ==========================================
# 4. 数据读写与工具
# ==========================================


def parse_csv(csv_path: str) -> Dict[str, jnp.ndarray]:
    """
    读取 csv（列: id,x,y,deg），拆分 group，并把角度转为弧度。
    返回: group_id -> params (N,3) [x,y,theta(rad)]
    """
    print(f"加载初始解: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["x", "y", "deg"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip("s")
    df[["group_id", "item_id"]] = df["id"].astype(str).str.split("_", n=2, expand=True)

    dict_params: Dict[str, jnp.ndarray] = {}
    for gid, gdf in df.groupby("group_id"):
        params: List[List[float]] = []
        for row in gdf.itertuples(index=False):
            x = float(row.x)
            y = float(row.y)
            theta_rad = np.deg2rad(float(row.deg))
            params.append([x, y, theta_rad])
        dict_params[gid] = jnp.array(params, dtype=jnp.float64)
    return dict_params


def save_dict_to_csv(dict_of_params: Dict[str, np.ndarray], output_path: str):
    """
    将 group -> params(N,3) 写回 csv，角度转回度数并加 s 前缀。
    """
    rows = []
    for gid in sorted(dict_of_params.keys(), key=lambda x: int(x)):
        params = np.array(dict_of_params[gid])
        for idx, (x, y, theta) in enumerate(params):
            rows.append(
                {
                    "id": f"{gid}_{idx}",
                    "x": f"s{Decimal(str(x))}",
                    "y": f"s{Decimal(str(y))}",
                    "deg": f"s{Decimal(str(np.rad2deg(theta)))}",
                }
            )
    pd.DataFrame(rows, columns=["id", "x", "y", "deg"]).to_csv(output_path, index=False)
    print(f"结果已保存: {output_path}")


# ==========================================
# 5. 求解器类 (Solver)
# ==========================================

class BoostedHydraulicPress:
    def __init__(self, num_trees, initial_size=10.0, initial_params=None, seed=42):
        self.N = num_trees

        if initial_params is not None:
            self.params = jnp.array(initial_params, dtype=jnp.float64)
            if initial_size is None:
                self.box_size = self._estimate_box_size(np.array(initial_params))
            else:
                self.box_size = float(initial_size)
        else:
            self.box_size = initial_size
            rng = np.random.default_rng(seed)
            pos = (rng.random((self.N, 2)) - 0.5) * self.box_size * 0.8
            theta = rng.random((self.N, 1)) * 2 * np.pi
            self.params = jnp.array(np.hstack([pos, theta]))

        # 保存初始边界尺寸作为松弛阶段的上限（留 10% 裕度）
        self.initial_box_size = self.box_size * 1.1

        # Boosting 权重矩阵 (初始全为 1)
        self.weights = jnp.ones((self.N, self.N))

        # 记录最佳结果
        self.best_params = self.params
        self.best_size = self.box_size
        self.loss_history: List[float] = []

    @staticmethod
    def _estimate_box_size(params_np: np.ndarray) -> float:
        """根据初始坐标估算容器尺寸，留出极小裕度。"""
        xs = params_np[:, 0]
        ys = params_np[:, 1]
        # 减小裕度到 0.2，让液压机更快接触到树木
        extent = np.max(np.abs(np.concatenate([xs, ys]))) * 2.0 + 0.2
        return max(extent, 1.0)

    def solve(self, epochs, steps_per_epoch, log_loss=True):
        """
        主优化循环 (压缩-松弛交替)
        
        逻辑:
        1. 检查当前是否有重叠
        2. 如果没有重叠 -> 压缩边界
        3. 如果有重叠 -> 梯度下降松弛
           - 检查重叠 loss 是否收敛（相邻 5 个检查点都下降）
           - 如果无法收敛 -> 扩大边界 0.5%，继续松弛
        """
        lr = 0.03
        compression = 0.99  # 每次压缩比例
        expand_rate = 1.005  # 无法收敛时扩大 0.5%
        check_interval = 50  # 每 50 步检查一次
        convergence_window = 2  # 连续 5 次下降才算收敛
        max_expand_times = 20  # 最大扩大次数（防止无限扩大）
        compact_steps = 100   # 压缩后的收缩步数（让树木向中心移动）

        print(f"开始优化: N={self.N}, 初始尺寸={self.box_size:.2f}, 边界上限={self.initial_box_size:.2f}")

        epoch = 0
        total_steps = 0
        
        while epoch < epochs:
            # 调试：检查 params 是否有 NaN
            params_np = np.asarray(self.params)
            if np.any(np.isnan(params_np)):
                print(f"[DEBUG] params contains NaN at epoch {epoch}!")
                print(f"  NaN count: {np.sum(np.isnan(params_np))}")
                print(f"  First few params: {params_np[:3]}")
                break
                
            # 1. 检查当前是否有重叠 (使用 Shapely 精确检测)
            shapely_overlap = int(compute_overlap_count(
                self.params, self.box_size, TREE_VERTS, TREE_SAMPLES
            ))
            
            if shapely_overlap == 0:
                # 2. 没有重叠 -> 压缩边界
                self.box_size *= compression
                epoch += 1
                
                # 压缩后进行梯度下降，让树木向中心收缩
                for step_i in range(compact_steps):
                    grads = grad_fn(
                        self.params, self.weights, self.box_size, 
                        TREE_VERTS, TREE_SAMPLES
                    )
                    # 梯度裁剪：替换 NaN 为 0，限制梯度大小
                    grads = jnp.nan_to_num(grads, nan=0.0, posinf=1.0, neginf=-1.0)
                    grads = jnp.clip(grads, -10.0, 10.0)
                    self.params = self.params - lr * grads
                
                if log_loss:
                    total_loss, overlap_loss, bound_loss, _ = compute_loss_breakdown(
                        self.params, self.weights, self.box_size, TREE_VERTS, TREE_SAMPLES
                    )
                    # 压缩后再检查一次重叠 (Shapely)
                    new_shapely_overlap = int(compute_overlap_count(
                        self.params, self.box_size, TREE_VERTS, TREE_SAMPLES
                    ))
                    status = "✓" if new_shapely_overlap == 0 else f"✗({new_shapely_overlap})"
                    print(
                        f"Epoch {epoch}/{epochs} | [COMPRESS] Box: {self.box_size:.4f} | "
                        f"Total: {float(total_loss):.6f} | OverlapLoss: {float(overlap_loss):.6f} | "
                        f"Bound: {float(bound_loss):.6f} | Shapely: {status}"
                    )
            else:
                # 3. 有重叠 -> 梯度下降松弛
                if log_loss:
                    print(f"  [RELAX] 检测到 {shapely_overlap} 个重叠，开始松弛...")
                
                # 松弛阶段不限制步数，持续到重叠消除或失败
                max_relax_steps = 10000  # 松弛最大步数
                relaxed = self._relax_until_converge(
                    lr, check_interval, convergence_window, 
                    expand_rate, max_expand_times, max_relax_steps, log_loss
            )
                
                if not relaxed:
                    # 松弛失败，停止优化
                    print(f"[WARN] 松弛失败，停止优化")
                    break
            
            # 更新最佳结果
            self.best_params = self.params
            self.best_size = self.box_size

        print(f"优化完成! 最终尺寸: {self.best_size:.4f}")

    def _relax_until_converge(self, lr, check_interval, convergence_window, 
                               expand_rate, max_expand_times, max_steps, log_progress):
        """
        松弛直到重叠消除或收敛失败
        
        收敛判断: 相邻 convergence_window 次检查，overlap_loss 都在下降
        如果不收敛: 扩大边界 0.5%，继续松弛
        
        返回: True 如果成功消除重叠，False 如果失败
        """
        expand_count = 0
        step = 0
        loss_history = []  # 记录最近的 overlap_loss
        
        while step < max_steps:
            # 梯度下降（使用 Shapely 数值梯度）
            grads = grad_fn(
                self.params, self.weights, self.box_size, 
                TREE_VERTS, TREE_SAMPLES
            )
            # 梯度裁剪：替换 NaN 为 0，限制梯度大小
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=1.0, neginf=-1.0)
            grads = jnp.clip(grads, -10.0, 10.0)
            self.params = self.params - lr * grads
            step += 1
            
            # 定期检查
            if step % check_interval == 0:
                # 使用 Shapely 精确检测并获取重叠对
                params_np = np.asarray(self.params)
                overlapping_pairs = get_overlapping_pairs_shapely(params_np)
                shapely_overlap = len(overlapping_pairs)
                
                # Shapely 重叠已消除
                if shapely_overlap == 0:
                    if log_progress:
                        print(f"    [Step {step}] 重叠已消除! ✓")
                    return True
                
                # 动态更新 Boosting 权重：增加重叠对的权重
                # 这样梯度下降会更强调消除这些真正重叠的对
                weights_np = np.array(self.weights, copy=True)  # 创建可写副本
                boost_factor = 1.2  # 每次检查增加 20%
                for (i, j) in overlapping_pairs:
                    weights_np[i, j] = min(weights_np[i, j] * boost_factor, 10.0)  # 最大权重 10
                    weights_np[j, i] = min(weights_np[j, i] * boost_factor, 10.0)
                self.weights = jnp.array(weights_np)
                
                # 计算当前 overlap loss (用于梯度下降监控)
                _, overlap_loss, _, _ = compute_loss_breakdown(
                self.params, self.weights, self.box_size, TREE_VERTS, TREE_SAMPLES
            )
                overlap_loss = float(overlap_loss)
                loss_history.append(overlap_loss)
                
                if log_progress:
                    max_weight = float(jnp.max(self.weights))
                    print(f"    [Step {step}] Overlaps: {shapely_overlap} | OverlapLoss: {overlap_loss:.6f} | Box: {self.box_size:.4f} | MaxW: {max_weight:.1f}")
                
                # Early Stopping 风格的收敛判断：连续 convergence_window 次变化很小
                if len(loss_history) >= convergence_window:
                    recent = loss_history[-convergence_window:]
                    
                    # 计算最近几轮的变化量
                    changes = [abs(recent[i] - recent[i+1]) for i in range(len(recent)-1)]
                    avg_change = sum(changes) / len(changes) if changes else float('inf')
                    
                    # 相对变化：相对于当前 loss 的比例
                    relative_change = avg_change / (overlap_loss + 1e-10)
                    
                    # 判断是否 "停滞"（变化很小）
                    is_stagnant = avg_change < 0.001 or relative_change < 0.01
                    
                    if is_stagnant:
                        # Loss 停滞，说明当前边界下无法继续优化，需要扩大边界
                        if self.box_size * expand_rate <= self.initial_box_size:
                            self.box_size *= expand_rate
                            expand_count += 1
                            loss_history.clear()  # 重置历史
                            
                            if log_progress:
                                print(f"    [EXPAND #{expand_count}] Loss 停滞 (avg_change={avg_change:.6f})，扩大边界到 {self.box_size:.4f}")
                            
                            if expand_count >= max_expand_times:
                                if log_progress:
                                    print(f"    [FAIL] 已扩大 {max_expand_times} 次，仍无法消除重叠")
                                return False
                        else:
                            if log_progress:
                                print(f"    [FAIL] 已达到边界上限 {self.initial_box_size:.4f}，无法继续扩大")
                            return False
                
                # 更新 Boosting 权重
                    self._apply_boosting()
            
        # 达到最大步数
        overlap_count = int(compute_overlap_count(
            self.params, self.box_size, TREE_VERTS, TREE_SAMPLES
        ))
        return overlap_count == 0

    def _ensure_zero_overlap(self, max_steps=10000, log_progress=True):
        """
        额外松弛阶段：确保重叠完全消除
        如果仍有重叠，会逐步扩大 box_size 并继续松弛
        注意：box_size 不会超过初始边界尺寸
        """
        lr = 0.02
        check_interval = 100
        expand_rate = 1.005  # 每次扩大 0.5%
        max_box_size = self.initial_box_size  # 边界上限为初始尺寸
        
        overlap_count = int(compute_overlap_count(
            self.params, self.box_size, TREE_VERTS, TREE_SAMPLES
        ))
        
        if overlap_count == 0:
            if log_progress:
                print(f"[OK] 已无重叠，无需额外松弛")
            return
        
        if log_progress:
            print(f"\n[RELAXATION] 开始额外松弛阶段，当前重叠点数: {overlap_count}")
            print(f"  边界上限: {max_box_size:.4f} (不超过初始解边界)")
        
        step = 0
        no_progress_count = 0
        last_overlap = overlap_count
        
        while step < max_steps and overlap_count > 0:
            # 计算梯度并更新
            grads = grad_fn(
                self.params, self.weights, self.box_size, 
                TREE_VERTS, TREE_SAMPLES
            )
            self.params = self.params - lr * grads
            
            step += 1
            
            # 定期检查重叠
            if step % check_interval == 0:
                overlap_count = int(compute_overlap_count(
                    self.params, self.box_size, TREE_VERTS, TREE_SAMPLES
            ))
                
                if log_progress:
                    total_loss, overlap_loss, bound_loss, _ = compute_loss_breakdown(
                        self.params, self.weights, self.box_size, TREE_VERTS, TREE_SAMPLES
                    )
                    print(f"  [Step {step}] 重叠点: {overlap_count} | Total: {float(total_loss):.6f} | Overlap: {float(overlap_loss):.6f} | Bound: {float(bound_loss):.6f} | Box: {self.box_size:.4f}")
                
                # 如果重叠没有减少，扩大边界（但不超过初始尺寸）
                if overlap_count >= last_overlap:
                    no_progress_count += 1
                    if no_progress_count >= 3:
                        new_size = self.box_size * expand_rate
                        if new_size <= max_box_size:
                            self.box_size = new_size
                            if log_progress:
                                print(f"  [EXPAND] 扩大边界到 {self.box_size:.4f}")
                        else:
                            if log_progress and self.box_size < max_box_size:
                                self.box_size = max_box_size
                                print(f"  [EXPAND] 已达到边界上限 {max_box_size:.4f}")
                        no_progress_count = 0
                else:
                    no_progress_count = 0
                
                last_overlap = overlap_count
                
                # Boosting
                if step % 500 == 0:
                    self._apply_boosting()
        
        # 更新最佳结果
        self.best_params = self.params
        self.best_size = self.box_size
        
        if overlap_count == 0:
            if log_progress:
                print(f"[SUCCESS] 松弛完成! 重叠已消除, 最终 Box: {self.box_size:.4f}")
        else:
            if log_progress:
                print(f"[WARN] 达到最大步数 {max_steps}，仍有 {overlap_count} 个重叠点")

    def _apply_boosting(self):
        """计算当前重叠残差，增加困难样本的权重"""
        # 利用梯度的模长近似重叠程度
        # 如果某棵树梯度很大，说明它受力很大（重叠严重）
        grads = grad_fn(
            self.params, jnp.ones_like(self.weights), self.box_size,
            TREE_VERTS, TREE_SAMPLES
        )
        grad_norms = jnp.sqrt(jnp.sum(grads**2, axis=1)) # (N,)
        
        # 找到受力最大的前 20% 的树
        threshold = jnp.percentile(grad_norms, 80)
        hard_mask = grad_norms > threshold # (N,) 布尔数组
        
        # 构造权重更新矩阵: 如果 i 和 j 都是困难户，它们之间的斥力加倍
        # 使用外积: (N, 1) * (1, N) -> (N, N)
        boost_factor = jnp.outer(hard_mask, hard_mask) * 0.5 # 增加 0.5 倍权重
        
        # 更新权重
        self.weights = self.weights + boost_factor
        # 限制最大权重防止溢出
        self.weights = jnp.clip(self.weights, 1.0, 50.0)

# ==========================================
# 5. 可视化
# ==========================================

# ==========================================
# 6. 运行主程序
# ==========================================


def process_single_group(gid, params, epochs, steps_per_epoch, initial_size, log_progress=False):
    """
    处理单个组的工作函数 (用于多线程)
    返回: (gid, best_params_np, base_loss, final_loss, best_size, improved, final_overlap)
    """
    solver = BoostedHydraulicPress(
        num_trees=params.shape[0],
        initial_size=initial_size,
        initial_params=params,
    )
    
    base_loss = float(
        compute_total_loss(
            solver.params,
            jnp.ones((solver.N, solver.N)),
            solver.box_size,
            TREE_VERTS,
            TREE_SAMPLES,
        )
    )
    
    base_overlap = int(compute_overlap_count(
        solver.params, solver.box_size, TREE_VERTS, TREE_SAMPLES
    ))
    
    # 在线程中减少日志输出，避免输出混乱
    solver.solve(epochs=epochs, steps_per_epoch=steps_per_epoch, log_loss=log_progress)
    
    final_loss = float(
        compute_total_loss(
            solver.best_params,
            jnp.ones((solver.N, solver.N)),
            solver.best_size,
            TREE_VERTS,
            TREE_SAMPLES,
        )
    )
    
    final_overlap = int(compute_overlap_count(
        solver.best_params, solver.best_size, TREE_VERTS, TREE_SAMPLES
    ))
    
    improved = final_loss < base_loss
    return (gid, np.array(solver.best_params), base_loss, final_loss, solver.best_size, improved, final_overlap)


def main():
    parser = argparse.ArgumentParser(description="Boosting SA with CSV IO")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/Users/zbr/code/santa2025/best_result.csv",
        help="初始解 csv（id,x,y,deg）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/Users/zbr/code/santa2025/output_boost.csv",
        help="输出解 csv",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--steps_per_epoch", type=int, default=2000, help="每轮步骤")
    parser.add_argument(
        "--initial_size",
        type=float,
        default=None,
        help="容器初始边长；缺省则自动从初始解估计",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=None,
        help="只优化前 N 个分组（按编号升序）",
    )
    parser.add_argument(
        "--single_group",
        type=int,
        default=None,
        help="只优化指定的单个分组 ID (覆盖 --max_groups 和 SINGLE_GROUP_ID)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行线程数；缺省使用 CPU 核心数",
    )
    args = parser.parse_args()

    dict_of_params = parse_csv(args.input_csv)

    # 确定要优化的组
    # 优先级: 命令行 --single_group > 全局变量 SINGLE_GROUP_ID > --max_groups
    single_gid = args.single_group if args.single_group is not None else SINGLE_GROUP_ID
    
    if single_gid is not None:
        # 只优化单个组
        # 尝试多种格式匹配 (支持带前导零和不带前导零的 group id)
        gid_candidates = [
            str(single_gid),           # "25"
            f"{single_gid:03d}",       # "025"
            f"{single_gid:02d}",       # "25" (两位)
        ]
        gid_str = None
        for candidate in gid_candidates:
            if candidate in dict_of_params:
                gid_str = candidate
                break
        
        if gid_str is None:
            print(f"[ERROR] 组 {single_gid} 不存在于输入文件中!")
            available = sorted(dict_of_params.keys())
            print(f"可用的组 ID: {available[:10]}... (共 {len(available)} 个)")
            return
        group_ids = [gid_str]
        print(f"=== 单组模式: 只优化组 {gid_str} ===")
    else:
        group_ids = sorted(dict_of_params.keys(), key=lambda x: int(x))
    if args.max_groups is not None:
        group_ids = group_ids[: args.max_groups]

    total_groups = len(group_ids)
    
    # 单组模式使用单线程并显示详细日志
    is_single_mode = single_gid is not None
    
    if is_single_mode:
        # 单组模式：直接运行，显示详细日志
        gid = group_ids[0]
        params = dict_of_params[gid]
        
        overall_start = time.time()
        print(f"组 {gid} 共有 {params.shape[0]} 棵树")
        print(f"epochs={args.epochs}, steps_per_epoch={args.steps_per_epoch}")
        print("-" * 50)
        
        gid, best_params, base_loss, final_loss, best_size, improved, final_overlap = process_single_group(
            gid, params, args.epochs, args.steps_per_epoch, args.initial_size, log_progress=True
        )
        
        results = {gid: best_params}
        # 同时保留其他组的原始数据
        for other_gid in dict_of_params.keys():
            if other_gid not in results:
                results[other_gid] = np.array(dict_of_params[other_gid])
        
        save_dict_to_csv(results, args.output_csv)
        
        print("-" * 50)
        print(f"组 {gid} 优化{'成功' if improved else '未改善'}")
        print(f"  Loss: {base_loss:.6f} -> {final_loss:.6f}")
        print(f"  尺寸: {best_size:.4f}")
        print(f"  最终重叠点数: {final_overlap}")
        print(f"耗时: {time.time() - overall_start:.2f}s")
        
    else:
        # 多组模式：并行处理
        num_workers = args.workers if args.workers else min(os.cpu_count(), total_groups)
    
    overall_start = time.time()
    print(f"即将优化 {total_groups} 个分组，epochs={args.epochs}, steps={args.steps_per_epoch}")
    print(f"使用 {num_workers} 个并行线程")

    results: Dict[str, np.ndarray] = {}
    improved_count = 0
    no_overlap_count = 0
    completed = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_gid = {
            executor.submit(
                process_single_group,
                    gid, dict_of_params[gid], args.epochs, args.steps_per_epoch, 
                    args.initial_size, True
            ): gid
            for gid in group_ids
        }
        
        # 按完成顺序收集结果
        for future in as_completed(future_to_gid):
            gid = future_to_gid[future]
            try:
                gid, best_params, base_loss, final_loss, best_size, improved, final_overlap = future.result()
                results[gid] = best_params
                
                if improved:
                    improved_count += 1
                    if final_overlap == 0:
                        no_overlap_count += 1
                
                completed += 1
                with print_lock:
                    overlap_status = "✓" if final_overlap == 0 else f"✗({final_overlap})"
                    print(
                        f"[{completed}/{total_groups}] 组 {gid} 完成: "
                        f"loss {base_loss:.6f} -> {final_loss:.6f}, "
                        f"尺寸 {best_size:.4f}, 重叠 {overlap_status}"
                    )
            except Exception as e:
                completed += 1
                with print_lock:
                    print(f"[{completed}/{total_groups}] 组 {gid} 失败: {e}")
                # 失败时保留原始参数
                results[gid] = np.array(dict_of_params[gid])

    save_dict_to_csv(results, args.output_csv)

    print(
        f"\n全部完成! 改善分组数: {improved_count}/{total_groups}, "
            f"无重叠分组数: {no_overlap_count}/{total_groups}, "
        f"耗时 {time.time() - overall_start:.2f}s"
    )


if __name__ == "__main__":
    main()