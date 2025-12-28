import math
import random
import pandas as pd
import numpy as np  # 必须引入 numpy 来修复类型判断
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import time
import multiprocessing
import os

# --- Configuration ---
getcontext().prec = 50
scale_factor = Decimal('1e18')

# --- ChristmasTree Class ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.polygon = self._create_polygon()

    def _create_polygon(self):
        # 始终保持 1.0 的大小，绝对不缩放几何体
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')
        base_w = Decimal('0.7'); base_y = Decimal('0.0')
        mid_w = Decimal('0.4'); tier_2_y = Decimal('0.25')
        top_w = Decimal('0.25'); tier_1_y = Decimal('0.5')
        tip_y = Decimal('0.8'); trunk_bottom_y = -trunk_h
        
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
        poly = Polygon(points)
        rotated = affinity.rotate(poly, float(self.angle), origin=(0, 0))
        final_poly = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )
        return final_poly

# --- Helper Function (FIXED) ---
def get_strtree_query_result(tree, geom, items_list=None):
    """
    修正后的兼容性查询函数。
    使用 numpy 类型检查来避免 TypeError。
    """
    result = tree.query(geom)
    if len(result) == 0: return []

    first = result[0]
    
    # 判定是否为索引模式 (Shapely 2.0 返回 int 或 numpy.int)
    # 使用 np.integer 可以同时覆盖 int32, int64 等所有 numpy 整数类型
    is_index = isinstance(first, (int, np.integer))
    
    if is_index:
        if items_list is None: 
            raise ValueError("items_list required for Shapely 2.0 index mapping")
        # 如果是 numpy array，直接用 result 索引 items_list 可能会有问题，稳妥起见转成 list 索引
        return [items_list[i] for i in result]
    
    # Shapely 1.x 返回对象
    return result

# --- SA Solver for Overlap Minimization ---

def run_squeeze_sa(args):
    group_id, trees, target_score_ratio = args
    scale_f = float(scale_factor)
    n = len(trees)
    
    # 1. 硬压缩 (Hard Compression)
    # print(f"Group {group_id}: Compressing by ratio {target_score_ratio:.5f}...")
    
    state = []
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for i, t in enumerate(trees):
        # 核心：坐标压缩
        cx = float(t.center_x) * scale_f * target_score_ratio
        cy = float(t.center_y) * scale_f * target_score_ratio
        angle = float(t.angle)
        
        # 重新生成多边形 (临时用于检测)
        temp_x = Decimal(cx) / scale_factor
        temp_y = Decimal(cy) / scale_factor
        t_temp = ChristmasTree(str(temp_x), str(temp_y), str(angle))
        
        state.append({
            'idx': i,
            'poly': t_temp.polygon,
            'cx': cx,
            'cy': cy,
            'angle': angle
        })
        
        min_x = min(min_x, cx); max_x = max(max_x, cx)
        min_y = min(min_y, cy); max_y = max(max_y, cy)

    # 稍微放宽一点点牢笼
    bounds_margin = 0.05 * scale_f 
    limit_min_x = min_x - bounds_margin
    limit_max_x = max_x + bounds_margin
    limit_min_y = min_y - bounds_margin
    limit_max_y = max_y + bounds_margin

    # 计算初始重叠
    polys = [s['poly'] for s in state]
    tree_index = STRtree(polys)
    current_overlap_area = 0.0
    
    # 全量检测
    for i in range(n):
        # 使用修复后的 helper
        candidates = get_strtree_query_result(tree_index, state[i]['poly'], list(range(n)))
        for cand_idx in candidates:
            j = int(cand_idx)
            if j > i: 
                if state[i]['poly'].intersects(state[j]['poly']):
                    current_overlap_area += state[i]['poly'].intersection(state[j]['poly']).area

    # === 关键修正：若无重叠，返回压缩后的树 ===
    if current_overlap_area == 0:
        compressed_trees = []
        for s in state:
            nx = Decimal(s['cx']) / scale_factor
            ny = Decimal(s['cy']) / scale_factor
            compressed_trees.append(ChristmasTree(str(nx), str(ny), str(s['angle'])))
        return group_id, compressed_trees, 0.0 

    # SA Loop
    max_iter = 150000 
    T = 1.0
    alpha = 0.99995
    
    best_overlap = current_overlap_area
    best_state_params = [{'cx': s['cx'], 'cy': s['cy'], 'angle': s['angle']} for s in state]
    
    for iteration in range(max_iter):
        idx = random.randint(0, n-1)
        target = state[idx]
        
        old_cx, old_cy, old_angle = target['cx'], target['cy'], target['angle']
        old_poly = target['poly']
        
        move_step = 2.0 * scale_f * 0.001 * T
        angle_step = 2.0 * T 
        
        dx = (random.random() - 0.5) * move_step
        dy = (random.random() - 0.5) * move_step
        da = (random.random() - 0.5) * angle_step
        
        new_cx = old_cx + dx
        new_cy = old_cy + dy
        new_angle = old_angle + da
        
        # 越界检查
        if not (limit_min_x <= new_cx <= limit_max_x and limit_min_y <= new_cy <= limit_max_y):
            continue 
            
        poly_rot = affinity.rotate(old_poly, da, origin=(old_cx, old_cy))
        new_poly = affinity.translate(poly_rot, xoff=dx, yoff=dy)
        
        # Cost Delta
        old_local_overlap = 0.0
        candidates = get_strtree_query_result(tree_index, old_poly, list(range(n)))
        neighbor_indices = []
        for cand in candidates:
            j = int(cand)
            if j != idx:
                neighbor_indices.append(j)
                if old_poly.intersects(state[j]['poly']):
                    old_local_overlap += old_poly.intersection(state[j]['poly']).area
        
        new_local_overlap = 0.0
        for j in neighbor_indices:
             if new_poly.intersects(state[j]['poly']):
                 new_local_overlap += new_poly.intersection(state[j]['poly']).area
                 
        delta = new_local_overlap - old_local_overlap
        
        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-6:
             if random.random() < math.exp(-delta / (scale_f * T * 100)): 
                 accept = True
                 
        if accept:
            target['cx'] = new_cx
            target['cy'] = new_cy
            target['angle'] = new_angle
            target['poly'] = new_poly
            current_overlap_area += delta
            
            if current_overlap_area < best_overlap:
                best_overlap = current_overlap_area
                best_state_params[idx] = {'cx': new_cx, 'cy': new_cy, 'angle': new_angle}
            
            if current_overlap_area <= 1e-9:
                # print(f"Group {group_id}: Converged to 0 overlap at iter {iteration}!")
                break
        
        T *= alpha
        if iteration % 5000 == 0:
            polys = [s['poly'] for s in state]
            tree_index = STRtree(polys)

    # 还原结果
    final_trees = []
    for i, s in enumerate(best_state_params):
        nx = Decimal(s['cx']) / scale_factor
        ny = Decimal(s['cy']) / scale_factor
        final_trees.append(ChristmasTree(str(nx), str(ny), str(s['angle'])))
        
    return group_id, final_trees, best_overlap

# --- Main Pipeline ---

def parse_csv(csv_path):
    print(f'Loading csv: {csv_path}')
    result = pd.read_csv(csv_path)
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
            ChristmasTree(center_x=str(row.x), center_y=str(row.y), angle=str(row.deg))
            for row in group_data.itertuples(index=False)
        ]
        dict_of_tree_list[group_id] = tree_list
    return dict_of_tree_list

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

def main_squeeze():
    # === 使用说明 ===
    # 1. 填入你那个 70.9 分的 CSV (干净、无重叠的那个)
    INPUT_FILE = "C:\\kaggle\\70.983188115078.csv"
    OUTPUT_FILE = "C:\\kaggle\\squeezed_solution.csv"
    
    # 2. 设定目标压缩比
    # 例如：想从 70.9 压到 70.5 -> 70.5 / 70.9 ≈ 0.994
    # 建议步子别太大，先试 0.998
    TARGET_RATIO = 0.998 
    
    print(f"--- Loading {INPUT_FILE} ---")
    try:
        all_groups = parse_csv(INPUT_FILE)
    except:
        print("File not found.")
        return

    tasks = []
    for gid, trees in all_groups.items():
        tasks.append((gid, trees, TARGET_RATIO))
        
    print(f"Starting Squeeze SA on {len(tasks)} groups...")
    
    results_dict = {}
    
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        for gid, trees, final_overlap in pool.imap_unordered(run_squeeze_sa, tasks):
            results_dict[gid] = trees
            status = "CLEAN" if final_overlap <= 1e-9 else f"DIRTY ({final_overlap:.2f})"
            print(f"Group {gid}: Finished. Status: {status}")
            
            if len(results_dict) % 10 == 0:
                save_solution(results_dict, OUTPUT_FILE)

    save_solution(results_dict, OUTPUT_FILE)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_squeeze()