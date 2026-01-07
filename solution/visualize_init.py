"""Visualize initial solution quality"""
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
from clusters import ChristmasTree, scale_factor

def calculate_score(csv_path):
    """Calculate total score from CSV"""
    df = pd.read_csv(csv_path)
    df['group_id'] = df['id'].str.split('_').str[0]

    scores = []
    group_info = []

    for group_id, group_data in df.groupby('group_id'):
        n = len(group_data)
        trees = []
        for _, row in group_data.iterrows():
            x = str(row.x).strip('s')
            y = str(row.y).strip('s')
            deg = str(row.deg).strip('s')
            tree = ChristmasTree(x, y, deg)
            trees.append(tree)

        all_polygons = [t.polygon for t in trees]
        minx = min(p.bounds[0] for p in all_polygons) / float(scale_factor)
        miny = min(p.bounds[1] for p in all_polygons) / float(scale_factor)
        maxx = max(p.bounds[2] for p in all_polygons) / float(scale_factor)
        maxy = max(p.bounds[3] for p in all_polygons) / float(scale_factor)

        width = maxx - minx
        height = maxy - miny
        side = max(width, height)
        score = (side ** 2) / n
        scores.append(score)

        aspect_ratio = max(width, height) / min(width, height)
        group_info.append({
            'group': int(group_id),
            'n': n,
            'width': width,
            'height': height,
            'side': side,
            'score': score,
            'aspect_ratio': aspect_ratio
        })

    return sum(scores), group_info

total_score, info = calculate_score('/Users/zbr/code/santa2025/solution/collision_free_full.csv')
print(f"TOTAL SCORE: {total_score:.6f}")

avg_aspect = sum(g['aspect_ratio'] for g in info) / len(info)
perfect = sum(1 for g in info if g['aspect_ratio'] < 1.05)
good = sum(1 for g in info if g['aspect_ratio'] < 1.1)

print(f"Average aspect ratio: {avg_aspect:.3f}")
print(f"Perfect (<1.05): {perfect}/200 ({perfect/2:.1f}%)")
print(f"Good (<1.1): {good}/200 ({good/2:.1f}%)")
