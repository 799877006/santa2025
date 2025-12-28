import math
import random
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import time
import multiprocessing

# --- Configuration copied from SA_GEMINI.py ---
getcontext().prec = 50
scale_factor = Decimal('1e18')

# --- ChristmasTree Class (Adapted) ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0', shrink_factor=Decimal('1.0')):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.shrink_factor = shrink_factor
        self.polygon = self._create_polygon()

    def _create_polygon(self):
        # Tree dimensions (standard 100% scale)
        trunk_w = Decimal('0.15') * self.shrink_factor
        trunk_h = Decimal('0.2') * self.shrink_factor
        base_w = Decimal('0.7') * self.shrink_factor
        base_y = Decimal('0.0') * self.shrink_factor
        mid_w = Decimal('0.4') * self.shrink_factor
        tier_2_y = Decimal('0.25') * self.shrink_factor
        top_w = Decimal('0.25') * self.shrink_factor
        tier_1_y = Decimal('0.5') * self.shrink_factor
        tip_y = Decimal('0.8') * self.shrink_factor
        trunk_bottom_y = -trunk_h * self.shrink_factor

        # Coordinates scaled by scale_factor
        points = [
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ]
        
        # Create polygon
        poly = Polygon(points)
        # Rotate
        rotated = affinity.rotate(poly, float(self.angle), origin=(0, 0))
        # Translate to position (also scaled)
        final_poly = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )
        return final_poly

# --- Helpers ---

def get_strtree_query_result(tree, geom, items_list=None):
    """Compatibility wrapper for Shapely STRtree query."""
    result = tree.query(geom)
    if len(result) == 0: return []
    first_elem = result[0]
    is_index_array = False
    if hasattr(result, 'dtype') and hasattr(result.dtype, 'kind'):
        if result.dtype.kind in 'iu': is_index_array = True
    elif isinstance(first_elem, (int, float)):
         is_index_array = True
    if is_index_array:
        if items_list is None: raise ValueError("items_list required for Shapely 2.0 index mapping")
        return [items_list[i] for i in result]
    return result

def solve_overlaps(trees, max_iter=None):
    """
    Final Optimizer with Wall Compression:
    1. 动态范围：从 0.999 -> 1.0 (微量膨胀)
    2. 墙壁压缩：强力压制边界扩张，防止分数反弹
    """
    # === 关键配置 ===
    # 既然此时 SA 已经是 0.999 了，我们只做最后千分之一的膨胀
    START_SCALE = 0.999  
    TARGET_SCALE = 1.0
    INFLATION_STEPS = 100 # 步数加倍，动作极慢
    FINAL_MAX_ITER = 10000  # 增加最大迭代次数
    
    n = len(trees)
    scale_f = float(scale_factor)
    
    # print(f"   >>> [Phase 1] Micro-Inflation ({START_SCALE} -> {TARGET_SCALE})...")

    state = []
    for i, t in enumerate(trees):
        state.append({
            'idx': i,
            'base_poly': t.polygon, 
            'cx': float(t.center_x) * scale_f,
            'cy': float(t.center_y) * scale_f,
            'angle': float(t.angle),
            'current_poly': None
        })

    # 计算初始包围盒边界（必须保持不变）
    initial_min_x = min(s['cx'] for s in state)
    initial_max_x = max(s['cx'] for s in state)
    initial_min_y = min(s['cy'] for s in state)
    initial_max_y = max(s['cy'] for s in state)
    
    # 计算初始重心
    cx_sum = sum(s['cx'] for s in state)
    cy_sum = sum(s['cy'] for s in state)
    gravity_x = cx_sum / n
    gravity_y = cy_sum / n
    
    def enforce_bounds(s):
        """强制限制单个树的位置，确保不超出初始边界"""
        if s['cx'] > initial_max_x: s['cx'] = initial_max_x
        elif s['cx'] < initial_min_x: s['cx'] = initial_min_x
        if s['cy'] > initial_max_y: s['cy'] = initial_max_y
        elif s['cy'] < initial_min_y: s['cy'] = initial_min_y
    
    def enforce_all_bounds():
        """确保整个包围盒不超出初始边界"""
        current_min_x = min(s['cx'] for s in state)
        current_max_x = max(s['cx'] for s in state)
        current_min_y = min(s['cy'] for s in state)
        current_max_y = max(s['cy'] for s in state)
        
        # 如果包围盒超出，强制收缩
        if current_min_x < initial_min_x:
            shift = initial_min_x - current_min_x
            for s in state:
                s['cx'] += shift
        if current_max_x > initial_max_x:
            shift = initial_max_x - current_max_x
            for s in state:
                s['cx'] += shift
        if current_min_y < initial_min_y:
            shift = initial_min_y - current_min_y
            for s in state:
                s['cy'] += shift
        if current_max_y > initial_max_y:
            shift = initial_max_y - current_max_y
            for s in state:
                s['cy'] += shift

    # Phase 1: 渐进膨胀
    for step in range(INFLATION_STEPS + 1):
        progress = step / INFLATION_STEPS
        current_scale = START_SCALE + (TARGET_SCALE - START_SCALE) * progress
        
        # 动态计算当前的包围盒 (用于施加墙壁压力)
        min_x = min(s['cx'] for s in state)
        max_x = max(s['cx'] for s in state)
        min_y = min(s['cy'] for s in state)
        max_y = max(s['cy'] for s in state)
        width = max_x - min_x
        height = max_y - min_y
        
        sub_iters = 10
        step_size = 1.0 * scale_f * 0.001 
        
        for _ in range(sub_iters):
            # 在子迭代开始前确保边界
            enforce_all_bounds()
            # 更新多边形
            current_polys = []
            for s in state:
                temp_poly = affinity.translate(
                    s['base_poly'], 
                    xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
                    yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
                )
                s['current_poly'] = affinity.scale(temp_poly, xfact=current_scale, yfact=current_scale)
                current_polys.append(s['current_poly'])
            
            tree_index = STRtree(current_polys)
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                me = state[i]
                
                # --- 核心修改：墙壁压缩力 (Wall Compression) ---
                # 这是一个指向中心的力，且距离边界越近，力越大
                # 这种力专门用来抵抗 "膨胀"
                
                # 归一化坐标 (-1 到 1)
                rel_x = (me['cx'] - gravity_x) / (width * 0.5 + 1e-9)
                rel_y = (me['cy'] - gravity_y) / (height * 0.5 + 1e-9)
                
                # 施加反向力 (向内压)，系数需要足够强以抵消斥力
                me['cx'] -= rel_x * 0.001 * scale_f 
                me['cy'] -= rel_y * 0.001 * scale_f
                
                # 边界检查：先检查初始包围盒，再检查全局边界
                enforce_bounds(me)
                limit = 100.0 * scale_f
                if me['cx'] > limit: me['cx'] = limit
                elif me['cx'] < -limit: me['cx'] = -limit
                if me['cy'] > limit: me['cy'] = limit
                elif me['cy'] < -limit: me['cy'] = -limit
                
                # --- 自然力场系统：弹簧-阻尼模型 ---
                candidates = get_strtree_query_result(tree_index, me['current_poly'], state)
                force_x, force_y = 0.0, 0.0
                
                # 估算树的"理想半径"（基于多边形面积）
                my_area = me['current_poly'].area
                ideal_radius = math.sqrt(my_area / math.pi) * 1.2  # 理想间距约为半径的1.2倍
                
                for other in candidates:
                    if other['idx'] == me['idx']: continue
                    
                    # 计算中心距离
                    dx = me['cx'] - other['cx']
                    dy = me['cy'] - other['cy']
                    center_dist = math.sqrt(dx*dx + dy*dy) + 1e-9
                    
                    # 检查是否有重叠
                    has_overlap = False
                    overlap_area = 0.0
                    if me['current_poly'].intersects(other['current_poly']):
                        try:
                            overlap_area = me['current_poly'].intersection(other['current_poly']).area
                            if overlap_area > 1e-9:
                                has_overlap = True
                        except: pass
                    
                    if has_overlap:
                        # 重叠时：强排斥力（与重叠面积成正比）
                        overlap_ratio = overlap_area / (my_area + 1e-9)
                        repulsion_strength = 1.0 + overlap_ratio * 10.0
                        force_x += (dx / center_dist) * repulsion_strength
                        force_y += (dy / center_dist) * repulsion_strength
                    elif center_dist < ideal_radius * 2.0:
                        # 距离较近但未重叠：轻微排斥（形成自然间距）
                        closeness = (ideal_radius * 2.0 - center_dist) / ideal_radius
                        gentle_repulsion = closeness * 0.1  # 轻微推力
                        force_x += (dx / center_dist) * gentle_repulsion
                        force_y += (dy / center_dist) * gentle_repulsion
                    # 距离很远时：不施加力（允许自然分布）
                
                # 应用力（如果有）
                force_magnitude = math.sqrt(force_x**2 + force_y**2)
                if force_magnitude > 1e-9:
                    # 归一化并应用，步长根据力的大小自适应
                    adaptive_step = step_size * min(1.0, force_magnitude / 5.0)
                    me['cx'] += (force_x / force_magnitude) * adaptive_step
                    me['cy'] += (force_y / force_magnitude) * adaptive_step
                    
                    # 边界检查 (移动后立即检查)
                    enforce_bounds(me)
                    limit = 100.0 * scale_f
                    if me['cx'] > limit: me['cx'] = limit
                    elif me['cx'] < -limit: me['cx'] = -limit
                    if me['cy'] > limit: me['cy'] = limit
                    elif me['cy'] < -limit: me['cy'] = -limit
            
            # 每个子迭代结束后确保边界
            enforce_all_bounds()

    # print(f"   >>> [Phase 2] Final Cleanup with Wall Compression...")
    
    # Phase 2: 保持 1.0，强力清理，但继续保留墙壁压力
    final_step_size = 3.0 * scale_f * 0.001  # 增大初始步长
    decay = 0.9995  # 减慢衰减

    for final_iter in range(FINAL_MAX_ITER):
        # ... (同样的更新多边形逻辑) ...
        current_polys = []
        for s in state:
            temp_poly = affinity.translate(
                s['base_poly'], 
                xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
                yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
            )
            s['current_poly'] = temp_poly
            current_polys.append(temp_poly)
            
        tree_index = STRtree(current_polys)
        indices = list(range(n))
        random.shuffle(indices)
        total_overlaps = 0
        
        # 重新计算边界用于压缩
        min_x = min(s['cx'] for s in state); max_x = max(s['cx'] for s in state)
        min_y = min(s['cy'] for s in state); max_y = max(s['cy'] for s in state)
        width = max_x - min_x; height = max_y - min_y
        
        for i in indices:
            me = state[i]
            # 持续施加墙壁压力!
            rel_x = (me['cx'] - gravity_x) / (width * 0.5 + 1e-9)
            rel_y = (me['cy'] - gravity_y) / (height * 0.5 + 1e-9)
            
            # Phase 2 的压力稍微小一点点，允许局部调整，但不能完全放开
            me['cx'] -= rel_x * 0.0002 * scale_f
            me['cy'] -= rel_y * 0.0002 * scale_f
            
            # 边界检查：先检查初始包围盒，再检查全局边界
            enforce_bounds(me)
            limit = 100.0 * scale_f
            if me['cx'] > limit: me['cx'] = limit
            elif me['cx'] < -limit: me['cx'] = -limit
            if me['cy'] > limit: me['cy'] = limit
            elif me['cy'] < -limit: me['cy'] = -limit
            
            candidates = get_strtree_query_result(tree_index, me['current_poly'], state)
            force_x, force_y = 0.0, 0.0
            
            # 估算树的"理想半径"
            my_area = me['current_poly'].area
            ideal_radius = math.sqrt(my_area / math.pi) * 1.2
            
            for other in candidates:
                if other['idx'] == me['idx']: continue
                
                # 计算中心距离
                dx = me['cx'] - other['cx']
                dy = me['cy'] - other['cy']
                center_dist = math.sqrt(dx*dx + dy*dy) + 1e-9
                
                # 检查是否有重叠
                has_overlap = False
                overlap_area = 0.0
                if me['current_poly'].intersects(other['current_poly']):
                    try:
                        overlap_area = me['current_poly'].intersection(other['current_poly']).area
                        if overlap_area > 1e-10:
                            has_overlap = True
                            total_overlaps += 1
                    except: pass
                
                if has_overlap:
                    # 重叠时：强排斥力（与重叠面积成正比）
                    overlap_ratio = overlap_area / (my_area + 1e-9)
                    repulsion_strength = 2.0 + overlap_ratio * 50.0  # Phase 2 更强
                    force_x += (dx / center_dist) * repulsion_strength
                    force_y += (dy / center_dist) * repulsion_strength
                elif center_dist < ideal_radius * 2.0:
                    # 距离较近但未重叠：轻微排斥（形成自然间距）
                    closeness = (ideal_radius * 2.0 - center_dist) / ideal_radius
                    gentle_repulsion = closeness * 0.15  # Phase 2 稍微强一点
                    force_x += (dx / center_dist) * gentle_repulsion
                    force_y += (dy / center_dist) * gentle_repulsion
            
            # 应用力（如果有）
            force_magnitude = math.sqrt(force_x**2 + force_y**2)
            if force_magnitude > 1e-9:
                # 归一化并应用，步长根据力的大小自适应
                adaptive_step = final_step_size * min(1.0, force_magnitude / 10.0)
                me['cx'] += (force_x / force_magnitude) * adaptive_step
                me['cy'] += (force_y / force_magnitude) * adaptive_step
                
                # 边界检查 (移动后立即检查)
                enforce_bounds(me)
                limit = 100.0 * scale_f
                if me['cx'] > limit: me['cx'] = limit
                elif me['cx'] < -limit: me['cx'] = -limit
                if me['cy'] > limit: me['cy'] = limit
                elif me['cy'] < -limit: me['cy'] = -limit
        
        # 每轮迭代结束后，强制确保整个包围盒不超出初始边界
        enforce_all_bounds()

        final_step_size *= decay
        if final_step_size < 1e-6 * scale_f: final_step_size = 1e-6 * scale_f
        
        if total_overlaps == 0:
            # 再次验证确保没有重叠
            current_polys_check = []
            for s in state:
                temp_poly = affinity.translate(
                    s['base_poly'], 
                    xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
                    yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
                )
                current_polys_check.append(temp_poly)
            
            # 严格检查
            all_clean = True
            check_tree = STRtree(current_polys_check)
            for i, p in enumerate(current_polys_check):
                candidates = get_strtree_query_result(check_tree, p, current_polys_check)
                for other in candidates:
                    if other is p: continue
                    if p.intersects(other):
                        try:
                            if p.intersection(other).area > 1e-6:
                                all_clean = False
                                break
                        except: pass
                if not all_clean: break
            
            if all_clean:
                break
        
        # if final_iter % 500 == 0:
        #     print(f"   ... Cleanup iter {final_iter}, Overlaps: {total_overlaps}")

    # Phase 3: 局部重排 - 针对仍有重叠的情况进行激进优化
    # 更新最终多边形用于检查
    final_polys = []
    for s in state:
        temp_poly = affinity.translate(
            s['base_poly'], 
            xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
            yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
        )
        final_polys.append(temp_poly)
    
    # 检查是否仍有重叠
    check_tree = STRtree(final_polys)
    overlapping_pairs = []
    for i, p in enumerate(final_polys):
        candidates = get_strtree_query_result(check_tree, p, final_polys)
        for other_poly in candidates:
            if other_poly is p: continue
            j = final_polys.index(other_poly)
            if p.intersects(other_poly):
                try:
                    inter_area = p.intersection(other_poly).area
                    if inter_area > 1e-6:
                        overlapping_pairs.append((i, j, inter_area))
                except: pass
    
    # 如果有重叠，进行局部重排
    if overlapping_pairs:
        # 找出所有涉及重叠的树索引
        overlapping_indices = set()
        for i, j, _ in overlapping_pairs:
            overlapping_indices.add(i)
            overlapping_indices.add(j)
        
        # 局部重排：对重叠的树进行更激进的优化
        local_iterations = 2000
        local_step = 5.0 * scale_f * 0.001  # 更大的步长
        
        for local_iter in range(local_iterations):
            # 更新多边形
            current_polys_local = []
            for s in state:
                temp_poly = affinity.translate(
                    s['base_poly'], 
                    xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
                    yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
                )
                current_polys_local.append(temp_poly)
            
            local_tree = STRtree(current_polys_local)
            local_overlaps = 0
            
            # 只处理重叠的树
            for idx in overlapping_indices:
                me = state[idx]
                me_poly = current_polys_local[idx]
                
                # 查找所有重叠的邻居
                candidates = get_strtree_query_result(local_tree, me_poly, current_polys_local)
                force_x, force_y = 0.0, 0.0
                max_overlap = 0.0
                
                for other_poly in candidates:
                    if other_poly is me_poly: continue
                    # 通过遍历找到对应的索引（更可靠）
                    other_idx = None
                    for k, poly in enumerate(current_polys_local):
                        if poly is other_poly:
                            other_idx = k
                            break
                    if other_idx is None or other_idx == idx: continue
                    other = state[other_idx]
                    
                    if me_poly.intersects(other_poly):
                        try:
                            overlap_area = me_poly.intersection(other_poly).area
                            if overlap_area > 1e-6:
                                local_overlaps += 1
                                max_overlap = max(max_overlap, overlap_area)
                                
                                # 计算排斥向量
                                dx = me['cx'] - other['cx']
                                dy = me['cy'] - other['cy']
                                dist = math.sqrt(dx*dx + dy*dy) + 1e-9
                                
                                # 非常强的排斥力
                                force_strength = 5.0 + (overlap_area * 200.0)
                                force_x += (dx / dist) * force_strength
                                force_y += (dy / dist) * force_strength
                        except: pass
                
                # 应用移动
                if force_x != 0.0 or force_y != 0.0:
                    force_mag = math.sqrt(force_x**2 + force_y**2)
                    if force_mag > 1e-9:
                        # 使用更大的步长
                        move_dist = local_step * (1.0 + max_overlap * 100.0)
                        me['cx'] += (force_x / force_mag) * move_dist
                        me['cy'] += (force_y / force_mag) * move_dist
                        
                        # 边界检查
                        enforce_bounds(me)
                        limit = 100.0 * scale_f
                        if me['cx'] > limit: me['cx'] = limit
                        elif me['cx'] < -limit: me['cx'] = -limit
                        if me['cy'] > limit: me['cy'] = limit
                        elif me['cy'] < -limit: me['cy'] = -limit
            
            # 确保整体边界
            enforce_all_bounds()
            
            # 如果无重叠，提前退出
            if local_overlaps == 0:
                # 最终验证
                final_check_polys = []
                for s in state:
                    temp_poly = affinity.translate(
                        s['base_poly'], 
                        xoff=s['cx'] - (float(trees[s['idx']].center_x) * scale_f),
                        yoff=s['cy'] - (float(trees[s['idx']].center_y) * scale_f)
                    )
                    final_check_polys.append(temp_poly)
                
                all_clean = True
                final_check_tree = STRtree(final_check_polys)
                for i, p in enumerate(final_check_polys):
                    candidates = get_strtree_query_result(final_check_tree, p, final_check_polys)
                    for other in candidates:
                        if other is p: continue
                        if p.intersects(other):
                            try:
                                if p.intersection(other).area > 1e-6:
                                    all_clean = False
                                    break
                            except: pass
                    if not all_clean: break
                
                if all_clean:
                    break
            
            # 步长衰减
            local_step *= 0.9995
            if local_step < 1e-6 * scale_f:
                local_step = 1e-6 * scale_f

    # 构造结果
    resolved_trees = []
    for s in state:
        orig = trees[s['idx']]
        new_x = Decimal(s['cx']) / scale_factor
        new_y = Decimal(s['cy']) / scale_factor
        resolved_trees.append(ChristmasTree(str(new_x), str(new_y), str(orig.angle)))
        
    return resolved_trees

# --- Parsing and Main Logic ---

def parse_csv(csv_path):
    print(f'Loading csv: {csv_path}')
    result = pd.read_csv(csv_path)
    
    # Clean 's' prefix
    for col in ['x', 'y', 'deg']:
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip('s')

    if 'id' in result.columns:
        result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)
    else:
        result['group_id'] = '0'
        
    dict_of_tree_list = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=str(row.x), center_y=str(row.y), angle=str(row.deg), shrink_factor=Decimal('1.0'))
            for row in group_data.itertuples(index=False)
        ]
        dict_of_tree_list[group_id] = tree_list
    return dict_of_tree_list

def validate_no_overlaps(trees):
    polygons = [t.polygon for t in trees]
    if not polygons:
        return True
    s = STRtree(polygons)
    for i, p in enumerate(polygons):
        candidates = get_strtree_query_result(s, p, polygons)
        for other in candidates:
            if other is p: continue
            if p.intersects(other):
                if p.intersection(other).area > 1e-6:
                    return False
    return True

def save_solution(dict_of_tree_list, output_path):
    data = []
    sorted_keys = sorted(dict_of_tree_list.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for group_id in sorted_keys:
        trees = dict_of_tree_list[group_id]
        for i, tree in enumerate(trees):
            data.append({
                'id': f"{group_id}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}",
            })
    df = pd.DataFrame(data)[['id', 'x', 'y', 'deg']]
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def process_group(args):
    gid, trees = args
    if validate_no_overlaps(trees):
        return gid, trees, "CLEAN", 0.0
        
    start_t = time.time()
    # 动态调整最大迭代次数 - 对于小组也增加迭代
    max_iter_val = 15000 if len(trees) > 50 else 10000
    new_trees = solve_overlaps(trees, max_iter=max_iter_val)
    dur = time.time() - start_t
    
    is_fixed = validate_no_overlaps(new_trees)
    status = "FIXED" if is_fixed else "FAILED"
    return gid, new_trees, status, dur

def main_resolve():
    # Configuration
    INPUT_FILE = "C:\\kaggle\\70.983188115078_shrinked0.999_round1.csv" 
    OUTPUT_FILE = "C:\\kaggle\\70.983188115078_shrinked0.999_round1_resolved.csv"
    
    print("--- Loading Solution ---")
    try:
        all_groups = parse_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File {INPUT_FILE} not found.")
        return

    sorted_groups = sorted(all_groups.items(), key=lambda x: len(x[1]), reverse=True)
    total_groups = len(sorted_groups)
    resolved_groups = {}
    
    num_processes = multiprocessing.cpu_count()
    print(f"Starting on {num_processes} cores...")
    
    tasks = [(gid, trees) for gid, trees in sorted_groups]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_group, tasks)
        finished_count = 0
        for gid, final_trees, status, dur in results:
            finished_count += 1
            resolved_groups[gid] = final_trees
            n_trees = len(final_trees)
            print(f"[{finished_count}/{total_groups}] Group {gid} ({n_trees} trees): {status} in {dur:.2f}s")
            
            if finished_count % 20 == 0:
                 save_solution(resolved_groups, OUTPUT_FILE)

    print("\n--- Saving Final Result ---")
    save_solution(resolved_groups, OUTPUT_FILE)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_resolve()