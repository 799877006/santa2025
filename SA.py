import pandas as pd  # 数据读取与写出
from decimal import Decimal, getcontext  # 高精度十进制支持
from shapely import affinity  # 仿射变换：旋转/平移
from shapely.geometry import Polygon  # 多边形对象
from shapely.strtree import STRtree  # 空间索引用于快速碰撞检测
import time  # 计时
import multiprocessing  # 多进程并行
import math  # 数学函数
import random  # 随机扰动
import os  # 文件路径处理
import shutil  # 文件复制

# --- 全局配置 ---
getcontext().prec = 50  # Decimal 运算精度 50 位
scale_factor = Decimal('1e18')  # 坐标缩放倍率，放大避免精度损失

def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = 0  # 总分初始化
    for k, v in dict_of_side_length.items():  # k 为组号，v 为该组边长
        score += v ** 2 / Decimal(k)  # 评分公式：边长平方 / 组号
    return score  # 返回总分

# --- 核心树形类定义 ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)  # 树中心 x
        self.center_y = Decimal(center_y)  # 树中心 y
        self.angle = Decimal(angle)  # 旋转角度（度）
        self.polygon = self._create_polygon()  # 构造多边形

    def _create_polygon(self):
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')  # 树干宽、高
        base_w = Decimal('0.7'); base_y = Decimal('0.0')     # 底层宽、y
        mid_w = Decimal('0.4'); tier_2_y = Decimal('0.25')   # 中层宽、y
        top_w = Decimal('0.25'); tier_1_y = Decimal('0.5')   # 上层宽、y
        tip_y = Decimal('0.8'); trunk_bottom_y = -trunk_h    # 树尖 y、树干底 y

        # 以原点为中心的树形轮廓，所有坐标都放大 scale_factor
        initial_polygon = Polygon([
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
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))  # 围绕原点旋转
        return affinity.translate(  # 平移到指定中心
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )

    def clone(self) -> "ChristmasTree":
        new_tree = ChristmasTree.__new__(ChristmasTree)  # 不调用 __init__，直接复制字段
        new_tree.center_x = self.center_x
        new_tree.center_y = self.center_y
        new_tree.angle = self.angle
        new_tree.polygon = self.polygon
        return new_tree

# --- 快速辅助函数 ---

def get_tree_list_side_length_fast(polygons) -> float:
    """基于 bbox 的树群边长（float 精度，速度快）。"""
    if not polygons:
        return 0.0  # 空列表返回 0
    minx, miny, maxx, maxy = polygons[0].bounds  # 先取第一个多边形的 bbox
    for p in polygons[1:]:  # 遍历更新全局最小/最大
        b = p.bounds
        if b[0] < minx: minx = b[0]
        if b[1] < miny: miny = b[1]
        if b[2] > maxx: maxx = b[2]
        if b[3] > maxy: maxy = b[3]
    return max(maxx - minx, maxy - miny) / float(scale_factor)  # 边长除以缩放系数


def validate_no_overlaps(polygons):
    """最终安全检查：使用 STRtree 判断是否存在面积重叠（触碰允许）。"""
    if not polygons:
        return True  # 空列表视为无重叠

    strtree = STRtree(polygons)  # 构建空间索引

    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)  # 查询可能碰撞的候选

        for cand in candidates:
            # Shapely 1.8 返回几何对象；2.x 可能返回索引
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly:
                    continue  # 跳过自身
            else:
                j = int(cand)
                if j == i:
                    continue
                other = polygons[j]

            # 允许 touches（边/点接触）；若既不分离也不 touches 则判为重叠
            if (not poly.disjoint(other)) and (not poly.touches(other)):
                return False

    return True  # 全部通过则无重叠


def parse_csv(csv_path):
    print(f'Loading csv: {csv_path}')  # 提示加载
    result = pd.read_csv(csv_path)  # 读取 csv
    for col in ['x', 'y', 'deg']:  # 去掉前缀 s
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip('s')

    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)  # 拆分组号/序号

    dict_of_tree_list = {}  # 组 -> 树列表
    for group_id, group_data in result.groupby('group_id'):
        # 使用 itertuples 更快
        tree_list = [
            ChristmasTree(center_x=str(row.x), center_y=str(row.y), angle=str(row.deg))
            for row in group_data.itertuples(index=False)
        ]
        dict_of_tree_list[group_id] = tree_list
    return dict_of_tree_list  # 返回字典


def save_dict_to_csv(dict_of_tree_list, output_path):
    # print(f"Saving solution to {output_path}...")  # 如需调试可打开
    data = []  # 汇总输出行
    sorted_keys = sorted(dict_of_tree_list.keys(), key=lambda x: int(x))  # 按组号排序
    for group_id in sorted_keys:
        trees = dict_of_tree_list[group_id]
        for i, tree in enumerate(trees):
            data.append({
                'id': f"{group_id}_{i}",         # 组号_序号
                'x': f"s{tree.center_x}",        # 写回时再加 s 前缀
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}",
            })
    df = pd.DataFrame(data)[['id', 'x', 'y', 'deg']]  # 固定列顺序
    df.to_csv(output_path, index=False)  # 保存
    print("Save complete.")  # 完成提示


# --- 模拟退火 worker ---

def run_simulated_annealing(args):
    group_id, initial_trees, max_iterations, t_start, t_end = args  # 解包参数
    n_trees = len(initial_trees)  # 树的数量

    # 判断规模：小规模提升迭代/温度，重力权重更高
    is_small_n = n_trees <= 50

    if is_small_n:
        effective_max_iter = max_iterations * 3  # 小规模多迭代
        effective_t_start = t_start * 2.0        # 初温更高
        gravity_weight = 1e-4                    # 重力权重更大
    else:
        effective_max_iter = max_iterations
        effective_t_start = t_start
        gravity_weight = 1e-6

    # 初始化状态列表：多边形 + 放大后的中心坐标 + 角度
    state = []
    for t in initial_trees:
        cx_float = float(t.center_x) * float(scale_factor)
        cy_float = float(t.center_y) * float(scale_factor)
        state.append({
            'poly': t.polygon,  # shapely 多边形
            'cx': cx_float,     # 中心 x（放大）
            'cy': cy_float,     # 中心 y（放大）
            'angle': float(t.angle),  # 角度（度）
        })

    current_polys = [s['poly'] for s in state]       # 当前多边形列表
    current_bounds = [p.bounds for p in current_polys]  # 对应 bbox

    scale_f = float(scale_factor)  # 缩放系数（float）
    inv_scale_f = 1.0 / scale_f    # 1/scale
    inv_scale_f2 = 1.0 / (scale_f * scale_f)  # 1/scale^2

    # 计算给定 bounds 列表的整体包围盒
    def _envelope_from_bounds(bounds_list):
        if not bounds_list:
            return (0.0, 0.0, 0.0, 0.0)
        minx, miny, maxx, maxy = bounds_list[0]
        for b in bounds_list[1:]:
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        return (minx, miny, maxx, maxy)

    # 仅替换 bounds_list[replace_i] 后计算包围盒
    def _envelope_from_bounds_replace(bounds_list, replace_i: int, replace_bounds):
        """替换一个 bbox 后的包围盒计算，不修改原列表。"""
        if not bounds_list:
            return (0.0, 0.0, 0.0, 0.0)
        b0 = replace_bounds if replace_i == 0 else bounds_list[0]
        minx, miny, maxx, maxy = b0
        for i, b in enumerate(bounds_list[1:], start=1):
            if i == replace_i:
                b = replace_bounds
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        return (minx, miny, maxx, maxy)

    def _side_len_from_env(env):
        minx, miny, maxx, maxy = env
        return max(maxx - minx, maxy - miny) * inv_scale_f  # 还原缩放

    # 初始化包围盒与“重力”距离平方和
    env = _envelope_from_bounds(current_bounds)
    dist_sum = 0.0
    for s in state:
        dist_sum += s['cx'] * s['cx'] + s['cy'] * s['cy']

    def energy_from(env_local, dist_sum_local):
        side_len = _side_len_from_env(env_local)  # 边长
        normalized_dist = (dist_sum_local * inv_scale_f2) / max(1, n_trees)  # 平均距离平方
        return side_len + gravity_weight * normalized_dist, side_len  # 返回能量与边长

    current_energy, current_side_len = energy_from(env, dist_sum)  # 当前能量/边长

    # 记录当前最优状态参数（只存坐标和角度）
    best_state_params = [{'cx': s['cx'], 'cy': s['cy'], 'angle': s['angle']} for s in state]
    best_real_score = current_side_len  # 最优实际边长

    T = effective_t_start  # 当前温度
    cooling_rate = math.pow(t_end / effective_t_start, 1.0 / effective_max_iter)  # 冷却率
    

    for i in range(effective_max_iter):  # 主循环
        progress = i / effective_max_iter  # 进度 [0,1]

        if is_small_n:
            move_scale = max(0.005, 3.0 * (1 - progress))  # 小规模步长随进度缩小
            rotate_scale = max(0.001, 5.0 * (1 - progress))
        else:
            move_scale = max(0.001, 1.0 * (T / effective_t_start))  # 大规模步长随温度缩小
            rotate_scale = max(0.002, 5.0 * (T / effective_t_start))

        idx = random.randint(0, n_trees - 1)  # 随机挑选一棵树
        target = state[idx]

        orig_poly = target['poly']  # 原多边形
        orig_bounds = current_bounds[idx]  # 原 bbox
        orig_cx, orig_cy, orig_angle = target['cx'], target['cy'], target['angle']

        # 随机平移与旋转扰动
        dx = (random.random() - 0.5) * scale_f * 0.1 * move_scale
        dy = (random.random() - 0.5) * scale_f * 0.1 * move_scale
        d_angle = (random.random() - 0.5) * rotate_scale

        rotated_poly = affinity.rotate(orig_poly, d_angle, origin=(orig_cx, orig_cy))  # 围绕自身中心旋转
        new_poly = affinity.translate(rotated_poly, xoff=dx, yoff=dy)  # 平移
        new_bounds = new_poly.bounds  # 新 bbox
        minx, miny, maxx, maxy = new_bounds

        new_cx = orig_cx + dx  # 新中心 x
        new_cy = orig_cy + dy  # 新中心 y
        new_angle = orig_angle + d_angle  # 新角度

        # --- 碰撞检测：先 bbox 过滤，再用 disjoint/touches ---
        collision = False
        for k in range(n_trees):
            if k == idx:
                continue  # 跳过自身
            ox1, oy1, ox2, oy2 = current_bounds[k]
            if maxx < ox1 or minx > ox2 or maxy < oy1 or miny > oy2:
                continue  # bbox 不相交则跳过
            other = current_polys[k]
            # touches 允许，若既不分离也不 touches 则判为重叠
            if (not new_poly.disjoint(other)) and (not new_poly.touches(other)):
                collision = True
                break

        if collision:
            T *= cooling_rate  # 碰撞则降温继续
            continue

        # 更新距离平方和（重力项）
        old_d = orig_cx * orig_cx + orig_cy * orig_cy
        new_d = new_cx * new_cx + new_cy * new_cy
        cand_dist_sum = dist_sum - old_d + new_d

        # 包围盒增量更新：若原极值被破坏则重算，否则直接扩张
        env_minx, env_miny, env_maxx, env_maxy = env
        need_recompute = (
            (orig_bounds[0] == env_minx and new_bounds[0] > env_minx) or
            (orig_bounds[1] == env_miny and new_bounds[1] > env_miny) or
            (orig_bounds[2] == env_maxx and new_bounds[2] < env_maxx) or
            (orig_bounds[3] == env_maxy and new_bounds[3] < env_maxy)
        )
        if need_recompute:
            cand_env = _envelope_from_bounds_replace(current_bounds, idx, new_bounds)  # 破坏极值需重算
        else:
            cand_env = (
                min(env_minx, new_bounds[0]),
                min(env_miny, new_bounds[1]),
                max(env_maxx, new_bounds[2]),
                max(env_maxy, new_bounds[3]),
            )  # 未破坏极值直接更新

        new_energy, new_real_score = energy_from(cand_env, cand_dist_sum)  # 新能量/边长
        delta = new_energy - current_energy  # 能量差

        # Metropolis 接受准则
        accept = False
        if delta < 0:
            accept = True  # 更优直接接受
        else:
            if T > 1e-10:
                prob = math.exp(-delta * 1000 / T)  # 按温度计算接受概率
                accept = random.random() < prob

        if accept:
            # 提交新状态
            current_polys[idx] = new_poly
            current_bounds[idx] = new_bounds
            target['poly'] = new_poly
            target['cx'] = new_cx
            target['cy'] = new_cy
            target['angle'] = new_angle

            current_energy = new_energy
            env = cand_env
            dist_sum = cand_dist_sum

            # 记录最好边长
            if new_real_score < best_real_score:
                best_real_score = new_real_score
                for k in range(n_trees):
                    best_state_params[k]['cx'] = state[k]['cx']
                    best_state_params[k]['cy'] = state[k]['cy']
                    best_state_params[k]['angle'] = state[k]['angle']

        T *= cooling_rate  # 冷却

    # 构造最终树对象并做一次严格碰撞校验
    final_trees = []
    final_polys_check = []
    for p in best_state_params:
        cx_dec = Decimal(p['cx']) / scale_factor  # 还原缩放
        cy_dec = Decimal(p['cy']) / scale_factor
        angle_dec = Decimal(p['angle'])
        new_t = ChristmasTree(str(cx_dec), str(cy_dec), str(angle_dec))
        final_trees.append(new_t)
        final_polys_check.append(new_t.polygon)

    if not validate_no_overlaps(final_polys_check):  # 若仍有重叠，退回原解
        orig_score = get_tree_list_side_length_fast([t.polygon for t in initial_trees])
        return group_id, initial_trees, orig_score

    return group_id, final_trees, best_real_score  # 返回组号、优化后树列表、最优得分

# --- 主流程 ---
def main():
    INPUT_CSV = "C:\\kaggle\\finally (5)_round1_spyrrow_final.csv"  # 初始输入
    OUTPUT_CSV = "C:\\kaggle\\finally (5)_round1_spyrrow_final2.csv"        # 输出
    LOOP_ROUNDS = 1  # 手动指定循环次数：每轮用上一轮输出作为下一轮输入

    current_input_csv = INPUT_CSV  # 当前输入文件

    for round_idx in range(1, LOOP_ROUNDS + 1):
        print("\n" + "#" * 60)
        print(f"ROUND {round_idx}/{LOOP_ROUNDS}")
        print(f"Input : {current_input_csv}")
        print(f"Output: {OUTPUT_CSV}")
        print("#" * 60 + "\n")

        try:
            dict_of_tree_list = parse_csv(current_input_csv)  # 读入解
        except FileNotFoundError:
            print(f"Error: Could not find {current_input_csv}.")
            return

        # 组按编号倒序
        groups_to_optimize = sorted(
            dict_of_tree_list.keys(), key=lambda x: int(x), reverse=True
        )

        MAX_ITER = 800000  # 单组基础迭代数
        T_START = 1.0      # 初始温度
        T_END = 0.003      # 终止温度

        KAGGLE_TIME_LIMIT_SEC = 11.5 * 3600  # 时间预算
        SAVE_EVERY_N_GROUPS = 20             # 每 N 组自动保存

        tasks = []
        for gid in groups_to_optimize:
            tasks.append((gid, dict_of_tree_list[gid], MAX_ITER, T_START, T_END))  # 构造任务

        num_processes = multiprocessing.cpu_count()  # CPU 核数
        print(f"Starting SA on {len(tasks)} groups using {num_processes} cores...")
        print(f"Time Limit: {KAGGLE_TIME_LIMIT_SEC / 3600:.2f} hours")
        print("Press Ctrl+C to stop early and save progress.")

        start_time = time.time()  # 计时
        improved_count = 0        # 改善组计数
        total_tasks = len(tasks)  # 总任务数
        finished_tasks = 0        # 已完成任务数

        pool = multiprocessing.Pool(processes=num_processes)  # 进程池

        try:
            results_iter = pool.imap_unordered(run_simulated_annealing, tasks, chunksize=1)  # 无序迭代结果

            for result in results_iter:
                group_id, optimized_trees, score = result  # 解包结果
                finished_tasks += 1

                orig_polys = [t.polygon for t in dict_of_tree_list[group_id]]  # 原多边形
                orig_score = get_tree_list_side_length_fast(orig_polys)        # 原边长

                status_msg = ""  # 状态提示
                if score < orig_score:
                    diff = orig_score - score
                    if diff > 1e-12:  # 有效改进
                        status_msg = f" -> Improved! (-{diff:.6f})"
                        dict_of_tree_list[group_id] = optimized_trees  # 更新解
                        improved_count += 1

                elapsed_time = time.time() - start_time  # 已耗时
                if elapsed_time > KAGGLE_TIME_LIMIT_SEC:
                    print(
                        f"\n[WARNING] Time limit approach ({elapsed_time / 3600:.2f}h). "
                        "Stopping early to save data safely."
                    )
                    pool.terminate()
                    break  # 超时提前结束

                if finished_tasks % SAVE_EVERY_N_GROUPS == 0:
                    print(
                        f"   >>> Auto-saving checkpoint at "
                        f"{finished_tasks}/{total_tasks}..."
                    )
                    save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)  # 自动保存

                print(
                    f"[{finished_tasks}/{total_tasks}] "
                    f"G:{group_id} {orig_score:.5f}->{score:.5f} {status_msg}"
                )

            pool.close()  # 正常关闭池
            pool.join()
            print(f"\nOptimization finished normally in {time.time() - start_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\n!!! Caught Ctrl+C (KeyboardInterrupt) !!!")
            print("Terminating workers and saving current progress...")
            pool.terminate()
            pool.join()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            pool.terminate()
            pool.join()
        finally:
            # --- 最终评分计算 ---
            print("\n" + "=" * 60)
            print("FINAL SCORE CALCULATION")
            print("=" * 60)
            
            dict_of_side_length = {}  # 组 -> 边长
            for group_id in dict_of_tree_list.keys():
                trees = dict_of_tree_list[group_id]
                polys = [t.polygon for t in trees]
                side_length = get_tree_list_side_length_fast(polys)  # 计算边长
                dict_of_side_length[group_id] = Decimal(str(side_length))  # 用 Decimal 保存
            
            total_score = get_total_score(dict_of_side_length)  # 总分
            
            print(f"Total Groups: {len(dict_of_side_length)}")
            print(f"Total Score: {total_score}")
            print(f"Groups Improved: {improved_count}")
            print("=" * 60)
            
            print(f"Final Save. Total Improved: {improved_count}")
            save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)  # 最终保存

            # 每轮都把 finally.csv 备份一份，文件名后缀带上循环回数
            base, ext = os.path.splitext(OUTPUT_CSV)
            round_csv = f"{base}_round{round_idx}{ext}"
            try:
                shutil.copyfile(OUTPUT_CSV, round_csv)
                print(f"Round Save: {round_csv}")
            except Exception as e:
                print(f"[WARNING] Could not backup round csv: {e}")

        # 下一轮用本轮的输出作为输入（文件名不变）
        current_input_csv = OUTPUT_CSV


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows 兼容
    main()