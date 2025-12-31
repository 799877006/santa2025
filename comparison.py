"""
比较两个 submission CSV，找出不一致的分组，绘制两份解的树形与包围框，并计算得分。

输入输出路径在代码内配置，不依赖命令行。
"""

import math
import pathlib
from typing import Dict, List, Tuple

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


TREE_POINTS = [
    (0.0, 0.8),
    (0.25 / 2, 0.5),
    (0.25 / 4, 0.5),
    (0.4 / 2, 0.25),
    (0.4 / 4, 0.25),
    (0.7 / 2, 0.0),
    (0.15 / 2, 0.0),
    (0.15 / 2, -0.2),
    (-0.15 / 2, -0.2),
    (-0.15 / 2, 0.0),
    (-0.7 / 2, 0.0),
    (-0.4 / 4, 0.25),
    (-0.4 / 2, 0.25),
    (-0.25 / 4, 0.5),
    (-0.25 / 2, 0.5),
]
BASE_POLY = Polygon(TREE_POINTS)


def strip_s(val: str) -> float:
    s = str(val).strip()
    if s.startswith("s"):
        s = s[1:]
    return float(s)


def load_submission(path: pathlib.Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    for col in ["x", "y", "deg"]:
        df[col] = df[col].map(strip_s)
    df[["group_id", "item_id"]] = df["id"].astype(str).str.split("_", n=1, expand=True)
    df["item_id_int"] = df["item_id"].astype(int)
    groups: Dict[str, pd.DataFrame] = {}
    for gid, g in df.groupby("group_id", sort=False):
        groups[str(gid)] = g.sort_values("item_id_int").reset_index(drop=True)
    return groups


def build_polys(group_df: pd.DataFrame) -> List[Polygon]:
    polys = []
    for row in group_df.itertuples(index=False):
        poly = affinity.rotate(BASE_POLY, float(row.deg), origin=(0.0, 0.0), use_radians=False)
        poly = affinity.translate(poly, xoff=float(row.x), yoff=float(row.y))
        polys.append(poly)
    return polys


def side_and_score(polys: List[Polygon]) -> Tuple[float, float]:
    if not polys:
        return 0.0, math.inf
    minx, miny, maxx, maxy = polys[0].bounds
    for p in polys[1:]:
        x1, y1, x2, y2 = p.bounds
        minx = min(minx, x1)
        miny = min(miny, y1)
        maxx = max(maxx, x2)
        maxy = max(maxy, y2)
    side = max(maxx - minx, maxy - miny)
    score = (side * side) / max(1, len(polys))
    return side, score


def max_overlap_area(polys: List[Polygon]) -> float:
    m = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            inter = polys[i].intersection(polys[j])
            if not inter.is_empty and not polys[i].touches(polys[j]):
                m = max(m, float(inter.area))
    return m


def group_diff_items(g1: pd.DataFrame, g2: pd.DataFrame, tol: float):
    ids = sorted(set(g1["id"]) | set(g2["id"]))
    diffs = []
    for tid in ids:
        row1 = g1[g1["id"] == tid]
        row2 = g2[g2["id"] == tid]
        if row1.empty or row2.empty:
            diffs.append((tid, "missing"))
            continue
        r1 = row1.iloc[0]
        r2 = row2.iloc[0]
        dx = abs(r1.x - r2.x)
        dy = abs(r1.y - r2.y)
        dd = abs(r1.deg - r2.deg)
        if dx > tol or dy > tol or dd > tol:
            diffs.append((tid, f"dx={dx:.3e}, dy={dy:.3e}, ddeg={dd:.3e}"))
    return diffs


def plot_group(gid: str, g1: pd.DataFrame, g2: pd.DataFrame, out_path: pathlib.Path, tol: float):
    polys1 = build_polys(g1)
    polys2 = build_polys(g2)
    side1, score1 = side_and_score(polys1)
    side2, score2 = side_and_score(polys2)
    ov1 = max_overlap_area(polys1)
    ov2 = max_overlap_area(polys2)
    diffs = group_diff_items(g1, g2, tol)
    diff_ids = {d[0] for d in diffs}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, polys, title, ov, df, color in [
        (axes[0], polys1, f"A | side={side1:.4f} score={score1:.4f} ov={ov1:.2e}", ov1, g1, "tab:red"),
        (axes[1], polys2, f"B | side={side2:.4f} score={score2:.4f} ov={ov2:.2e}", ov2, g2, "tab:blue"),
    ]:
        minx, miny, maxx, maxy = polys[0].bounds
        for poly, row in zip(polys, df.itertuples(index=False)):
            x, y = poly.exterior.xy
            lw = 1.5 if row.id in diff_ids else 0.8
            ec = "red" if row.id in diff_ids else "black"
            ax.plot(x, y, lw=lw, color=ec)
            bx1, by1, bx2, by2 = poly.bounds
            minx = min(minx, bx1); miny = min(miny, by1)
            maxx = max(maxx, bx2); maxy = max(maxy, by2)
            # 标注编号（质心处）
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, row.id, fontsize=6, ha="center", va="center", color=ec)
        # 画总包络
        ax.add_patch(
            plt.Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, ls="--", color="red", lw=1.0)
        )
        ax.set_aspect("equal", "box")
        ax.set_title(title)
    fig.suptitle(f"Group {gid} | diff items: {len(diff_ids)}", fontsize=12)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # 在此直接配置文件路径和参数
    file_a = pathlib.Path(r"C:\kaggle\submission (2).csv")
    file_b = pathlib.Path(r"C:\kaggle\submission (6)_sa.csv")
    out_dir = pathlib.Path("comparison_out")
    max_groups = 20
    tol = 1e-9
    score_tol = 1e-4

    out_dir.mkdir(parents=True, exist_ok=True)

    ga = load_submission(file_a)
    gb = load_submission(file_b)

    all_gids = sorted(set(ga.keys()) | set(gb.keys()), key=lambda s: int(s))
    diff_gids = []
    for gid in all_gids:
        if gid not in ga or gid not in gb:
            # 缺组无法比较，跳过
            continue
        polys_a = build_polys(ga[gid])
        polys_b = build_polys(gb[gid])
        _, score_a = side_and_score(polys_a)
        _, score_b = side_and_score(polys_b)
        if abs(score_a - score_b) > score_tol:
            diff_gids.append(gid)

    print(f"发现 score 差异分组: {len(diff_gids)} 个")
    if not diff_gids:
        return

    for idx, gid in enumerate(diff_gids[:max_groups], 1):
        if gid not in ga:
            print(f"[missing] A 无分组 {gid}")
            continue
        if gid not in gb:
            print(f"[missing] B 无分组 {gid}")
            continue
        polys_a = build_polys(ga[gid])
        polys_b = build_polys(gb[gid])
        _, score_a = side_and_score(polys_a)
        _, score_b = side_and_score(polys_b)
        diffs = group_diff_items(ga[gid], gb[gid], tol)
        out_path = out_dir / f"group_{gid}.png"
        plot_group(gid, ga[gid], gb[gid], out_path, tol)
        print(f"[{idx}/{len(diff_gids)}] gid={gid} score_a={score_a:.9f} score_b={score_b:.9f} diff_items={len(diffs)} -> {out_path}")
        for did, info in diffs:
            print(f"    id={did} {info}")


if __name__ == "__main__":
    main()

