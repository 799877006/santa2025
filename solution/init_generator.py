# -*- coding: utf-8 -*-
"""
SANTA2025 Initial Solution Generator
Square-optimized dimer packing without strict collision checking
Allows minor touching/overlap which SA will resolve
"""

import pandas as pd
from decimal import Decimal
import math
import sys

# Import custom classes
from clusters import ChristmasTree, scale_factor


def create_standard_dimer_trees(base_x, base_y, dimer_rotation=0):
    """
    Create two trees forming a standard dimer at given position

    Args:
        base_x, base_y: Base position for tree_a
        dimer_rotation: Rotation angle for the dimer (degrees)

    Returns:
        (tree_a, tree_b): Tuple of two ChristmasTree objects
    """
    # Standard dimer: tree_a at (0,0,0), tree_b at (0.35,0.8,180)
    angle_rad = math.radians(dimer_rotation)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Tree A at base position
    tree_a_x = base_x
    tree_a_y = base_y
    tree_a_angle = dimer_rotation

    # Tree B: offset by (0.35, 0.8) rotated by dimer_rotation
    offset_x, offset_y = 0.35, 0.8
    rotated_offset_x = offset_x * cos_a - offset_y * sin_a
    rotated_offset_y = offset_x * sin_a + offset_y * cos_a

    tree_b_x = base_x + rotated_offset_x
    tree_b_y = base_y + rotated_offset_y
    tree_b_angle = dimer_rotation + 180

    tree_a = ChristmasTree(str(tree_a_x), str(tree_a_y), str(tree_a_angle))
    tree_b = ChristmasTree(str(tree_b_x), str(tree_b_y), str(tree_b_angle))

    return tree_a, tree_b


def generate_dimer_packing(n_trees):
    """Generate square-optimized dimer packing"""
    print(f"Generating {n_trees} trees with square-optimized packing...")

    n_dimers = n_trees // 2
    n_single = n_trees % 2

    # Dimer spacing (tight packing)
    dimer_width = 0.85
    dimer_height = 0.95

    # Find optimal grid for square shape
    best_n_cols, best_n_rows = 1, n_dimers
    best_ratio = float('inf')

    for n_cols in range(1, min(n_dimers + 1, 50)):
        n_rows = int(math.ceil(n_dimers / n_cols))
        width = n_cols * dimer_width
        height = n_rows * dimer_height
        ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')

        if ratio < best_ratio:
            best_ratio = ratio
            best_n_cols, best_n_rows = n_cols, n_rows

        if ratio < 1.05:
            break

    n_cols, n_rows = best_n_cols, best_n_rows

    print(f"  Dimers: {n_dimers}, Single: {n_single}")
    print(f"  Grid: {n_cols} × {n_rows}")
    print(f"  Est. bbox: {n_cols * dimer_width:.2f} × {n_rows * dimer_height:.2f}")

    trees = []

    # Place dimers in grid (all same orientation for simplicity)
    for row in range(n_rows):
        for col in range(n_cols):
            if len(trees) >= n_trees - n_single:
                break

            # Position centered around origin
            x = (col - n_cols / 2 + 0.5) * dimer_width
            y = (row - n_rows / 2 + 0.5) * dimer_height

            # Create dimer trees (all 0° rotation)
            tree_a, tree_b = create_standard_dimer_trees(x, y, dimer_rotation=0)
            trees.extend([tree_a, tree_b])

    # Add single tree if needed
    if n_single > 0 and len(trees) < n_trees:
        print(f"  Adding single tree...")
        single_tree = ChristmasTree('0', '0', '0')
        trees.append(single_tree)

    # Calculate bbox and score
    all_polygons = [t.polygon for t in trees]
    minx = min(p.bounds[0] for p in all_polygons) / float(scale_factor)
    miny = min(p.bounds[1] for p in all_polygons) / float(scale_factor)
    maxx = max(p.bounds[2] for p in all_polygons) / float(scale_factor)
    maxy = max(p.bounds[3] for p in all_polygons) / float(scale_factor)

    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    score = (side ** 2) / n_trees

    print(f"  Actual bbox: {width:.3f} × {height:.3f}")
    print(f"  Side: {side:.3f}, Score: {score:.6f}")
    print(f"  Generated: {len(trees)} trees")

    return trees[:n_trees]


def trees_to_dataframe(trees, group_id):
    """Convert tree list to DataFrame"""
    data = []
    for idx, tree in enumerate(trees):
        data.append({
            'id': f"{group_id}_{idx}",
            'x': f"s{tree.center_x}",
            'y': f"s{tree.center_y}",
            'deg': f"s{tree.angle}"
        })
    return pd.DataFrame(data)


def generate_initial_solution(n_list, output_csv):
    """Generate complete initial solution"""
    print("=" * 60)
    print("SANTA2025 Initial Solution Generator")
    print("=" * 60)

    all_data = []

    for n in n_list:
        group_id = f"{n:03d}"
        print(f"\nGroup {group_id} ({n} trees)...")
        trees = generate_dimer_packing(n)
        df = trees_to_dataframe(trees, group_id)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Saved: {output_csv}")
    print(f"Total rows: {len(final_df)}")
    print("="*60)

    return final_df


def update_specific_groups(input_csv, output_csv, n_list):
    """Update specific groups"""
    print("=" * 60)
    print("Update Specific Groups")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"Reading: {input_csv} ({len(df)} rows)")

    df['group_id'] = df['id'].str.split('_').str[0]

    for n in n_list:
        group_id = f"{n:03d}"
        print(f"\nRegenerating group {group_id}...")
        df = df[df['group_id'] != group_id]
        trees = generate_dimer_packing(n)
        new_df = trees_to_dataframe(trees, group_id)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.drop(columns=['group_id'])
    df['sort_key'] = df['id'].str.split('_').str[0].astype(int) * 1000 + \
                     df['id'].str.split('_').str[1].astype(int)
    df = df.sort_values('sort_key').drop(columns=['sort_key'])

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv} ({len(df)} rows)")

    return df


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='SANTA2025 Initial Solution Generator')
    parser.add_argument('--mode', choices=['generate', 'update'], default='generate')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--groups', type=str, required=True)

    args = parser.parse_args()

    if args.groups == 'all':
        n_list = list(range(1, 201))
    elif '-' in args.groups:
        start, end = map(int, args.groups.split('-'))
        n_list = list(range(start, end + 1))
    else:
        n_list = [int(x.strip()) for x in args.groups.split(',')]

    print(f"Target: {n_list[:5]}{'...' if len(n_list) > 5 else ''} ({len(n_list)} groups)\n")

    if args.mode == 'generate':
        generate_initial_solution(n_list, args.output)
    else:
        if not args.input:
            print("Error: update mode requires --input")
            sys.exit(1)
        update_specific_groups(args.input, args.output, n_list)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Running with default parameters...\n")
        output_path = '/Users/zbr/code/santa2025/solution/initial_solution.csv'
        n_list = list(range(1, 11))
        generate_initial_solution(n_list, output_path)
    else:
        main()
