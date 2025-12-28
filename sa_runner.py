import time
import shutil
from decimal import Decimal, getcontext
from typing import List, Dict

import numpy as np
import pandas as pd


import pandas as pd
from decimal import Decimal, getcontext
from typing import Dict, List
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

import subprocess

INPUT_FILE = "/kaggle/input/santa2025-optimal-claus/submission.csv"
OUTPUT_FILE = "/kaggle/working/submission.csv"
SA_FILE = "/kaggle/working/submission_sa.csv"

shutil.copy(INPUT_FILE, SA_FILE)
shutil.copy(INPUT_FILE, OUTPUT_FILE)

getcontext().prec = 50
SCALE_FACTOR = Decimal("1e18")
SCALE_F = float(SCALE_FACTOR)
INV_SCALE_F = 1.0 / SCALE_F
INV_SCALE_F2 = 1.0 / (SCALE_F * SCALE_F)


def _strip_s(v) -> str:
    s = str(v).strip()
    return s[1:] if s.startswith("s") else s

def _make_base_poly_scaled() -> Polygon:
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w = Decimal("0.7")
    base_y = Decimal("0.0")
    mid_w = Decimal("0.4")
    tier_2_y = Decimal("0.25")
    top_w = Decimal("0.25")
    tier_1_y = Decimal("0.5")
    tip_y = Decimal("0.8")
    trunk_bottom_y = -trunk_h

    sf = SCALE_FACTOR
    return Polygon([
        (Decimal("0.0") * sf, tip_y * sf),
        (top_w / Decimal("2") * sf, tier_1_y * sf),
        (top_w / Decimal("4") * sf, tier_1_y * sf),
        (mid_w / Decimal("2") * sf, tier_2_y * sf),
        (mid_w / Decimal("4") * sf, tier_2_y * sf),
        (base_w / Decimal("2") * sf, base_y * sf),
        (trunk_w / Decimal("2") * sf, base_y * sf),
        (trunk_w / Decimal("2") * sf, trunk_bottom_y * sf),
        (-(trunk_w / Decimal("2")) * sf, trunk_bottom_y * sf),
        (-(trunk_w / Decimal("2")) * sf, base_y * sf),
        (-(base_w / Decimal("2")) * sf, base_y * sf),
        (-(mid_w / Decimal("4")) * sf, tier_2_y * sf),
        (-(mid_w / Decimal("2")) * sf, tier_2_y * sf),
        (-(top_w / Decimal("4")) * sf, tier_1_y * sf),
        (-(top_w / Decimal("2")) * sf, tier_1_y * sf),
    ])

BASE_POLY0 = _make_base_poly_scaled()

def build_polygon_scaled(cx: Decimal, cy: Decimal, ang: Decimal) -> Polygon:
    rotated = affinity.rotate(BASE_POLY0, float(ang), origin=(0, 0))
    return affinity.translate(rotated, xoff=float(cx * SCALE_FACTOR), yoff=float(cy * SCALE_FACTOR))

def side_length_from_polygons(polys: List[Polygon]) -> float:
    if not polys:
        return 0.0
    minx, miny, maxx, maxy = polys[0].bounds
    for p in polys[1:]:
        x1, y1, x2, y2 = p.bounds
        if x1 < minx: minx = x1
        if y1 < miny: miny = y1
        if x2 > maxx: maxx = x2
        if y2 > maxy: maxy = y2
    return max(maxx - minx, maxy - miny) * INV_SCALE_F

def validate_no_overlaps_strict(polys: List[Polygon]) -> bool:
    if not polys:
        return True
    st = STRtree(polys)
    id_to_i = {id(p): i for i, p in enumerate(polys)}  # 1.8 geometry返り対応

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

            # ジャッジ同等：intersects & not touches
            if p.intersects(q) and (not p.touches(q)):
                return False
    return True

def read_csv_as_groups(path: str) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    for col in ["x", "y", "deg"]:
        df[col] = df[col].map(_strip_s)

    df[["group_id", "item_id"]] = df["id"].astype(str).str.split("_", n=1, expand=True)
    df["item_id_int"] = df["item_id"].astype(int)

    groups = {}
    for gid, g in df.groupby("group_id", sort=False):
        groups[str(gid)] = g.sort_values("item_id_int").reset_index(drop=True)
    return groups

def build_polys_from_group_df(g: pd.DataFrame) -> List[Polygon]:
    polys = []
    for row in g.itertuples(index=False):
        cx = Decimal(str(row.x))
        cy = Decimal(str(row.y))
        ang = Decimal(str(row.deg))
        polys.append(build_polygon_scaled(cx, cy, ang))
    return polys

def apply_sa_results(
        base_csv="./submission.csv",
        cand_csv="./summission_sa.csv",
        out_csv="./submission.csv",
        n_min=5,
        n_max=200,
    ) -> int:
    base = read_csv_as_groups(base_csv)
    cand = read_csv_as_groups(cand_csv)

    improved = 0
    checked = 0

    for gid, base_g in base.items():
        if gid not in cand:
            continue
        cand_g = cand[gid]

        if len(cand_g) != len(base_g):
            print(f"[py] gid={gid} skip: different n (base={len(base_g)} cand={len(cand_g)})")
            continue
        if set(cand_g["id"]) != set(base_g["id"]):
            print(f"[py] gid={gid} skip: id set mismatch")
            continue

        n = len(base_g)
        if n < n_min or n > n_max:
            continue

        checked += 1

        base_polys = build_polys_from_group_df(base_g)
        cand_polys = build_polys_from_group_df(cand_g)

        base_side = side_length_from_polygons(base_polys)
        cand_side = side_length_from_polygons(cand_polys)

        ok = validate_no_overlaps_strict(cand_polys)
        better = cand_side < base_side - 1e-12

        print(f"[py] gid={gid} n={n} base={base_side:.9f} cand={cand_side:.9f} ok={ok} better={better}")

        if ok and better:
            base[gid] = cand_g
            improved += 1

    rows = []
    for gid in sorted(base.keys(), key=lambda s: int(s)):
        g = base[gid]
        for row in g.itertuples(index=False):
            rows.append({
                "id": row.id,
                "x": "s" + str(row.x),
                "y": "s" + str(row.y),
                "deg": "s" + str(row.deg),
            })

    out_df = pd.DataFrame(rows, columns=["id", "x", "y", "deg"])
    out_df.to_csv(out_csv, index=False)
    print(f"[py] checked={checked} improved={improved} -> wrote {out_csv}")

    return improved


    TREE_POLYGON_POINTS = [
    (0.0,   0.8),
    (0.25/2, 0.5),
    (0.25/4, 0.5),
    (0.4/2,  0.25),
    (0.4/4,  0.25),
    (0.7/2,  0.0),
    (0.15/2, 0.0),
    (0.15/2, -0.2),
    (-0.15/2, -0.2),
    (-0.15/2, 0.0),
    (-0.7/2,  0.0),
    (-0.4/4,  0.25),
    (-0.4/2,  0.25),
    (-0.25/4, 0.5),
    (-0.25/2, 0.5),
]
BASE_TREE_POLY_SCALED = Polygon([(x * SCALE_F, y * SCALE_F) for x, y in TREE_POLYGON_POINTS])

def _build_tree_polygon_scaled(x: float, y: float, deg: float) -> Polygon:
    rot = affinity.rotate(BASE_TREE_POLY_SCALED, deg, origin=(0.0, 0.0), use_radians=False)
    return affinity.translate(rot, xoff=x * SCALE_F, yoff=y * SCALE_F)

def _parse_s_column_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^s", "", regex=True)
    return pd.to_numeric(s, errors="raise")

def _compute_case_metrics_fast_xydeg(x: pd.Series, y: pd.Series, deg: pd.Series):
    n = len(x)
    if n == 0:
        return 0.0, float("inf"), False, 0.0

    polys = [_build_tree_polygon_scaled(float(xi), float(yi), float(di)) for xi, yi, di in zip(x, y, deg)]

    minx, miny, maxx, maxy = polys[0].bounds
    for p in polys[1:]:
        x1, y1, x2, y2 = p.bounds
        if x1 < minx: minx = x1
        if y1 < miny: miny = y1
        if x2 > maxx: maxx = x2
        if y2 > maxy: maxy = y2

    side = max(maxx - minx, maxy - miny) * INV_SCALE_F
    score = (side * side) / n

    has_overlap = False
    max_overlap_area = 0.0

    st = STRtree(polys)
    id_to_i = {id(p): i for i, p in enumerate(polys)}

    for i, pi in enumerate(polys):
        cands = st.query(pi)
        for cand in cands:
            if hasattr(cand, "geom_type"):
                j = id_to_i.get(id(cand), -1)
                if j <= i:
                    continue
                pj = cand
            else:
                j = int(cand)
                if j <= i:
                    continue
                pj = polys[j]

            if pi.intersects(pj) and (not pi.touches(pj)):
                has_overlap = True
                inter = pi.intersection(pj)
                if not inter.is_empty:
                    area = float(inter.area) * INV_SCALE_F2
                    if area > max_overlap_area:
                        max_overlap_area = area

    return float(side), float(score), bool(has_overlap), float(max_overlap_area)

def evaluate_submission_all_precise_fast(submission_path: str):
    df = pd.read_csv(submission_path)

    df["case_id"] = df["id"].astype(str).str.split("_", n=1, expand=True)[0]

    df["x_f"] = _parse_s_column_to_float(df["x"])
    df["y_f"] = _parse_s_column_to_float(df["y"])
    df["deg_f"] = _parse_s_column_to_float(df["deg"])

    results = []
    for cid, g in df.groupby("case_id", sort=False):
        side, score, has_ov, max_ov = _compute_case_metrics_fast_xydeg(g["x_f"], g["y_f"], g["deg_f"])
        results.append({
            "case_id": cid,
            "n_trees": len(g),
            "side": side,
            "score": score,
            "has_overlap": has_ov,
            "max_overlap_area": max_ov,
        })

    per_case_df = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    no_overlap = per_case_df[~per_case_df["has_overlap"]]
    total_score = float(no_overlap["score"].sum()) if len(no_overlap) > 0 else float("inf")

    print(f"Total score (no-overlap only) = {total_score:.12f}")
    print(f"Overlapped cases = {int(per_case_df['has_overlap'].sum())} / {len(per_case_df)}")

    return per_case_df, total_score



n_min = 5
n_max = 200
verbose = "0"

for i in range(5):
    seed = str(i)
    subprocess.run(
        ["./sa_runner", SA_FILE, SA_FILE,
        "200000", "0", "0", seed, verbose, str(n_min), str(n_max)],
        check=True
    )
    apply_sa_results(OUTPUT_FILE, SA_FILE, OUTPUT_FILE, n_min=n_min, n_max=n_max)

per_case, total = evaluate_submission_all_precise_fast(OUTPUT_FILE)
