#!/usr/bin/env python3  # 指定解释器
import argparse  # 命令行参数解析
import math
import multiprocessing
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import spyrrow


# --- Geometry helpers (native coordinates) ---
def get_tree_coords(scale: float = 1.0) -> List[Tuple[float, float]]:
    """Return the base Christmas-tree coordinates at the origin."""
    trunk_w, trunk_h = 0.15 * scale, 0.2 * scale
    base_w, base_y = 0.7 * scale, 0.0 * scale
    mid_w, tier_2_y = 0.4 * scale, 0.25 * scale
    top_w, tier_1_y = 0.25 * scale, 0.5 * scale
    tip_y = 0.8 * scale
    trunk_bottom_y = -trunk_h

    # Ensure coords are floats for spyrrow
    return [
        (0.0, tip_y),
        (top_w / 2, tier_1_y),
        (top_w / 4, tier_1_y),
        (mid_w / 2, tier_2_y),
        (mid_w / 4, tier_2_y),
        (base_w / 2, base_y),
        (trunk_w / 2, base_y),
        (trunk_w / 2, trunk_bottom_y),
        (-trunk_w / 2, trunk_bottom_y),
        (-trunk_w / 2, base_y),
        (-base_w / 2, base_y),
        (-mid_w / 4, tier_2_y),
        (-mid_w / 2, tier_2_y),
        (-top_w / 4, tier_1_y),
        (-top_w / 2, tier_1_y),
        (0.0, tip_y),
    ]


def fast_bbox(coords: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Calculate (minx, miny, maxx, maxy) for a list of coordinates."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_side(minx: float, miny: float, maxx: float, maxy: float) -> float:
    return max(maxx - minx, maxy - miny)


# --- CSV helpers ---
def load_submission(csv_path: str) -> pd.DataFrame:  # 读取提交 CSV
    df = pd.read_csv(csv_path)  # 读取文件
    for col in ["x", "y", "deg"]:  # 确保坐标为浮点
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip("s")
        df[col] = df[col].astype(float)
    df[["group_id", "item_id"]] = df["id"].str.split("_", n=1, expand=True)  # 拆分 id
    df["group_id"] = df["group_id"].astype(str)  # 组为字符串
    df["item_id"] = df["item_id"].astype(int)  # 项目为整数
    return df.sort_values(["group_id", "item_id"]).reset_index(drop=True)  # 排序并重置索引


def format_val(v: float, digits: int = 9) -> str:  # 输出格式化
    txt = f"{v:.{digits}f}".rstrip("0").rstrip(".")  # 去尾零和点
    return f"s{txt if txt else '0'}"  # 前缀 s


# --- Spyrrow conversion ---
def coords_to_item(name: str, coords: List[Tuple[float, float]], allowed_orientations: List[float]) -> spyrrow.Item:
    # spyrrow.Item signature: (id, shape, demand, allowed_orientations)
    return spyrrow.Item(name, coords, demand=1, allowed_orientations=allowed_orientations)


# --- Binary search solver ---
@dataclass
class SolveResult:
    side: float
    solution: spyrrow.StripPackingSolution


def can_fit(
    side: float,
    group_name: str,
    items: List[spyrrow.Item],
    time_limit: float,
    bbox_height: float,
    seed: int,
    num_workers: int,
) -> Tuple[bool, spyrrow.StripPackingSolution | None]:
    # Quick reject: if item bbox exceeds strip height, skip solver
    if bbox_height > side:
        return False, None
    try:
        instance = spyrrow.StripPackingInstance(name=group_name, strip_height=side, items=items)
        config = spyrrow.StripPackingConfig(
            early_termination=True,  # Allow early exit if fit found
            total_computation_time=int(time_limit),
            seed=seed,
            num_workers=num_workers,
        )
        solution = instance.solve(config)
        return solution.width <= side, solution
    except Exception as exc:
        print(f"[WARN] Solver failed at side={side}: {exc}", file=sys.stderr)
        return False, None


def binary_search_side(
    group_name: str,
    items: List[spyrrow.Item],
    low: float,
    high: float,
    iterations: int,
    time_limit: float,
    bbox_height: float,
    seed: int,
    num_workers: int,
    max_retries: int = 3,
    grow_factor: float = 1.5,
) -> SolveResult:
    attempt = 0
    while attempt <= max_retries:
        best_solution = None
        best_side = high
        for i in range(iterations):
            mid = 0.5 * (low + high)
            # Use smaller time limit for checks if desired, but here we use full time_limit
            ok, sol = can_fit(mid, group_name, items, time_limit, bbox_height, seed, num_workers)
            # Logging handled in outer loop or verbose mode
            if ok and sol:
                best_solution = sol
                best_side = mid
                high = mid
            else:
                low = mid
        if best_solution is not None:
            return SolveResult(side=best_side, solution=best_solution)
        
        low = high
        high = high * grow_factor
        attempt += 1
        print(f"[WARN] {group_name}: expanding search to high={high:.6f} (attempt {attempt}/{max_retries})")
        
    raise RuntimeError(f"No feasible solution found for group {group_name}")


# --- Utilities ---
def compute_group_bbox_side_fast(
    coords: List[Tuple[float, float]], 
    placements: List[Tuple[float, float, float]]
) -> float:
    """
    Compute bbox side for a set of placed items accurately.
    Instead of a loose radius approximation, we calculate the exact
    transformed coordinates for all vertices.
    """
    if not placements:
        return 0.0
    
    # Calculate global min/max
    g_minx, g_miny, g_maxx, g_maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    # Pre-compute sin/cos for allowed orientations if discrete (optimization optional but good)
    # Since placements can have arbitrary angles, we compute on the fly.
    
    # Optimization: If coords list is large, we could compute the Convex Hull first, 
    # but for ~16 points it's faster to just iterate all.
    
    for px, py, pdeg in placements:
        rad = math.radians(pdeg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        for x, y in coords:
            # Rotate then translate
            rx = x * cos_a - y * sin_a + px
            ry = x * sin_a + y * cos_a + py
            
            if rx < g_minx: g_minx = rx
            if rx > g_maxx: g_maxx = rx
            if ry < g_miny: g_miny = ry
            if ry > g_maxy: g_maxy = ry
            
    if g_minx > g_maxx: # Should not happen unless coords empty
        return 0.0
        
    return max(g_maxx - g_minx, g_maxy - g_miny)


def allowed_orientations(step: int) -> List[float]:
    vals = list(range(0, 360, step))
    if 360 not in vals:
        vals.append(360)
    return vals


def solution_to_rows(
    group_id: str,  # 组名
    placed_items,  # 放置结果
    digits: int,  # 小数位
) -> List[Dict[str, str]]:
    rows = []  # 输出行
    # Assume placed_items order matches item creation order
    for idx, pi in enumerate(placed_items):  # 遍历放置项
        tx, ty = pi.translation  # 平移
        rows.append(
            {
                "id": f"{group_id}_{idx}",  # id
                "x": format_val(tx, digits),  # x
                "y": format_val(ty, digits),  # y
                "deg": format_val(pi.rotation, digits),  # 角度
            }
        )
    return rows  # 返回


# --- Main routine ---
def solve_groups(
    df: pd.DataFrame,
    output_csv: str,
    angle_step: int,
    time_limit: float,
    iterations: int,
    num_workers: int,
    seed: int,
    scale: float,
    digits: int,
):
    all_rows: List[Dict[str, str]] = []
    base_coords = get_tree_coords(scale)
    # Simple polygon area using shoelace formula
    x = [p[0] for p in base_coords]
    y = [p[1] for p in base_coords]
    base_area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(base_coords) - 1)) + x[-1] * y[0] - x[0] * y[-1])
    
    base_minx, base_miny, base_maxx, base_maxy = fast_bbox(base_coords)
    base_bbox = bbox_side(base_minx, base_miny, base_maxx, base_maxy)
    orient_list = allowed_orientations(angle_step)

    # Sort groups by size descending for better logging
    groups = list(df.groupby("group_id"))
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    
    total_groups = len(groups)
    
    for idx, (group_id, gdf) in enumerate(groups, 1):
        item_count = len(gdf)
        group_items = [
            coords_to_item(f"{group_id}_{i}", base_coords, orient_list) for i in range(item_count)
        ]

        min_side = max(base_bbox, math.sqrt(base_area * item_count))

        # Upper bound from current placement
        placements = [(row.x, row.y, row.deg) for row in gdf.itertuples(index=False)]
        current_side = compute_group_bbox_side_fast(base_coords, placements) if placements else min_side * 2
        
        # heuristic generous upper bound
        grid_upper = base_bbox * max(1, math.ceil(math.sqrt(item_count))) * 2.0
        upper = max(current_side * 1.1, min_side * 2.0, grid_upper)

        # print(f"[INFO] Group {group_id}: n={item_count}, lower={min_side:.6f}, upper={upper:.6f}")
        start = time.time()
        result = binary_search_side(
            group_id,
            group_items,
            low=min_side,
            high=upper,
            iterations=iterations,
            time_limit=time_limit,
            bbox_height=base_bbox,
            seed=seed,
            num_workers=num_workers,
        )
        elapsed = time.time() - start
        
        # Logging similar to SA.py
        # [{current}/{total}] G:{group_id} {old_score:.5f}->{new_score:.5f} {status}
        old_score = current_side
        new_score = result.side
        status_msg = ""
        if new_score < old_score:
            diff = old_score - new_score
            if diff > 1e-9:
                 status_msg = f"-> Improved! (-{diff:.6f})"
        
        print(f"[{idx}/{total_groups}] G:{group_id} {old_score:.5f}->{new_score:.5f} {status_msg} (t={elapsed:.2f}s)")
        
        all_rows.extend(solution_to_rows(group_id, result.solution.placed_items, digits))

    pd.DataFrame(all_rows)[["id", "x", "y", "deg"]].to_csv(output_csv, index=False)
    print(f"[INFO] Saved solution to {output_csv}")


def parse_args() -> argparse.Namespace:  # 解析参数
    parser = argparse.ArgumentParser(description="Solve Santa 2025 with spyrrow + binary search for square.")  # 描述
    parser.add_argument("--input", default="submission.csv", help="Input CSV with id,x,y,deg (existing placement).")  # 输入
    parser.add_argument("--output", default="submission_spyrrow.csv", help="Output CSV path.")  # 输出
    parser.add_argument("--angle-step", type=int, default=5, help="Orientation granularity in degrees.")  # 角度步长
    parser.add_argument("--time-limit", type=float, default=30.0, help="Solver time budget per check (seconds).")  # 时间限制
    parser.add_argument("--iterations", type=int, default=12, help="Binary search iterations per group.")  # 二分次数
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Threads for spyrrow (num_workers). Defaults to all logical CPUs.",
    )  # 线程数
    parser.add_argument("--seed", type=int, default=0, help="Random seed for spyrrow.")  # 随机种子
    parser.add_argument("--scale", type=float, default=1.0, help="Global scale factor for geometry (keep 1.0).")  # 尺度
    parser.add_argument("--digits", type=int, default=9, help="Decimal digits in output formatting.")  # 小数位
    return parser.parse_args()  # 返回参数


def main():  # 主入口
    args = parse_args()  # 解析参数
    df = load_submission(args.input)  # 读取输入
    solve_groups(
        df=df,
        output_csv=args.output,
        angle_step=args.angle_step,
        time_limit=args.time_limit,
        iterations=args.iterations,
        num_workers=args.num_workers,
        seed=args.seed,
        scale=args.scale,
        digits=args.digits,
    )  # 执行求解


if __name__ == "__main__":  # 仅脚本执行时运行
    main()  # 调用主函数

# python solve_spyrrow.py --input "finally (5)_round1.csv" --output "optimized_result.csv"