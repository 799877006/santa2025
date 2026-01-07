"""
测试和可视化 Dimer 类
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from decimal import Decimal
import math
# 导入自定义类
from clusters import ChristmasTree, Dimer, scale_factor


# 固定的二聚体构型
DIMER_CONFIG = {
    'tree1': ('0', '0', '0'),
    'tree2': ('0.35', '0.8', '180')
}


def create_standard_dimer():
    """创建标准构型的二聚体"""
    tree1 = ChristmasTree(*DIMER_CONFIG['tree1'])
    tree2 = ChristmasTree(*DIMER_CONFIG['tree2'])
    return Dimer(tree1, tree2)


def plot_polygon(ax, polygon, color='blue', alpha=0.5, edgecolor='black', linewidth=1):
    """绘制 shapely Polygon"""
    # 获取外部坐标（需要除以 scale_factor 转回逻辑坐标）
    coords = list(polygon.exterior.coords)
    coords_scaled = [(x / float(scale_factor), y / float(scale_factor)) for x, y in coords]

    patch = MplPolygon(coords_scaled, facecolor=color, alpha=alpha,
                      edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(patch)


def plot_tree(ax, tree, color='green', label=None):
    """绘制单棵树"""
    plot_polygon(ax, tree.polygon, color=color, alpha=0.6)

    # 标记树的中心点
    cx = float(tree.center_x)
    cy = float(tree.center_y)
    ax.plot(cx, cy, 'ro', markersize=5)

    # 标记树的角度方向（箭头）
    angle_rad = float(tree.angle) * math.pi / 180.0
    arrow_length = 0.15
    dx = arrow_length * math.sin(angle_rad)
    dy = arrow_length * math.cos(angle_rad)
    ax.arrow(cx, cy, dx, dy, head_width=0.05, head_length=0.05,
            fc='red', ec='red', linewidth=2)

    if label:
        ax.text(cx, cy - 0.1, label, fontsize=8, ha='center')


def plot_dimer(ax, dimer, color1='lightblue', color2='lightcoral', label=None):
    """绘制二聚体"""
    plot_tree(ax, dimer.tree_a, color=color1)
    plot_tree(ax, dimer.tree_b, color=color2)

    # 标记二聚体中心
    cx = float(dimer.center_x)
    cy = float(dimer.center_y)
    ax.plot(cx, cy, 'y*', markersize=15, markeredgecolor='black', markeredgewidth=1)

    # 连线显示两棵树
    ax.plot([float(dimer.tree_a.center_x), float(dimer.tree_b.center_x)],
           [float(dimer.tree_a.center_y), float(dimer.tree_b.center_y)],
           'k--', alpha=0.3, linewidth=1)

    if label:
        ax.text(cx, cy + 0.15, label, fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))


def test_basic_dimer():
    """测试1：基本二聚体创建和显示"""
    print("=== 测试1：基本二聚体 ===")

    _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 创建标准二聚体
    dimer = create_standard_dimer()

    # 绘制
    plot_dimer(ax, dimer, label="Dimer")

    # 设置坐标轴
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test 1: Basic Dimer')

    # 显示信息
    info_text = f"Dimer Center: ({float(dimer.center_x):.3f}, {float(dimer.center_y):.3f})\n"
    info_text += f"Tree A: ({float(dimer.tree_a.center_x):.3f}, {float(dimer.tree_a.center_y):.3f})\n"
    info_text += f"Tree B: ({float(dimer.tree_b.center_x):.3f}, {float(dimer.tree_b.center_y):.3f})"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test1_basic_dimer.png', dpi=150)
    print("保存: test1_basic_dimer.png")
    plt.close()


def test_rotation():
    """测试2：旋转功能"""
    print("\n=== 测试2：旋转功能 ===")

    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 旋转角度列表
    angles = [0, 30, 60, 90, 120, 180]

    for idx, angle in enumerate(angles):
        ax = axes[idx]

        # 创建标准二聚体
        dimer = create_standard_dimer()

        # 计算初始距离
        dx = float(dimer.tree_b.center_x - dimer.tree_a.center_x)
        dy = float(dimer.tree_b.center_y - dimer.tree_a.center_y)
        initial_dist = math.sqrt(dx*dx + dy*dy)

        # 旋转
        if angle > 0:
            dimer.rotate(Decimal(str(angle)))

        # 计算旋转后距离
        dx = float(dimer.tree_b.center_x - dimer.tree_a.center_x)
        dy = float(dimer.tree_b.center_y - dimer.tree_a.center_y)
        current_dist = math.sqrt(dx*dx + dy*dy)

        # 绘制
        plot_dimer(ax, dimer, label=f"{angle}°")

        # 设置坐标轴
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Rotation: {angle}°')

        # 显示距离信息（验证刚性）
        dist_error = abs(current_dist - initial_dist)
        info = f"Distance: {current_dist:.6f}\nError: {dist_error:.10f}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round',
                        facecolor='lightgreen' if dist_error < 1e-6 else 'pink',
                        alpha=0.7))

        print(f"  角度 {angle}°: 距离={current_dist:.8f}, 误差={dist_error:.10f}")

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test2_rotation.png', dpi=150)
    print("保存: test2_rotation.png")
    plt.close()


def test_translation():
    """测试3：平移功能"""
    print("\n=== 测试3：平移功能 ===")

    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 创建多个平移后的二聚体
    translations = [
        (0, 0, 'Original'),
        (1, 0, 'Right'),
        (0, 1, 'Up'),
        (-1, 0, 'Left'),
        (0, -1, 'Down'),
        (1, 1, 'Diagonal')
    ]

    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan']

    for idx, (dx, dy, label) in enumerate(translations):
        # 创建标准二聚体
        dimer = create_standard_dimer()

        # 平移
        if dx != 0 or dy != 0:
            dimer.translate(Decimal(str(dx)), Decimal(str(dy)))

        # 使用不同颜色绘制
        color = colors[idx % len(colors)]
        plot_tree(ax, dimer.tree_a, color=color)
        plot_tree(ax, dimer.tree_b, color=color)

        # 标记
        cx = float(dimer.center_x)
        cy = float(dimer.center_y)
        ax.plot(cx, cy, 'o', markersize=8, color='red')
        ax.text(cx + 0.1, cy + 0.1, label, fontsize=9)

    ax.set_xlim(-2, 2.5)
    ax.set_ylim(-2, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test 3: Translation')

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test3_translation.png', dpi=150)
    print("保存: test3_translation.png")
    plt.close()


def test_dimers_in_rectangle():
    """测试4：在矩形中放置多个二聚体"""
    print("\n=== 测试4：矩形中的多个二聚体 ===")

    _, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 定义矩形边界
    rect_width = 5
    rect_height = 5

    # 创建多个二聚体（网格排列）
    dimers = []
    spacing = 1.2

    for i in range(4):
        for j in range(4):
            x = -2 + i * spacing
            y = -2 + j * spacing

            # 创建标准二聚体
            dimer = create_standard_dimer()

            # 平移到网格位置
            dimer.translate(Decimal(str(x)), Decimal(str(y)))

            # 随机旋转一些二聚体
            rotation_angle = (i * 23 + j * 37) % 360
            dimer.rotate(Decimal(str(rotation_angle)))

            dimers.append(dimer)

    # 绘制所有二聚体
    for idx, dimer in enumerate(dimers):
        # 交替颜色
        color1 = 'lightblue' if idx % 2 == 0 else 'lightgreen'
        color2 = 'lightcoral' if idx % 2 == 0 else 'lightyellow'
        plot_tree(ax, dimer.tree_a, color=color1)
        plot_tree(ax, dimer.tree_b, color=color2)

        # 标记二聚体中心
        cx = float(dimer.center_x)
        cy = float(dimer.center_y)
        ax.plot(cx, cy, 'k.', markersize=3)

    # 绘制矩形边界
    rect = patches.Rectangle((-rect_width/2, -rect_height/2), rect_width, rect_height,
                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Test 4: {len(dimers)} Dimers in Rectangle')

    # 添加图例
    ax.text(0.02, 0.98, f"Total Dimers: {len(dimers)}\nTotal Trees: {len(dimers)*2}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test4_rectangle.png', dpi=150)
    print("保存: test4_rectangle.png")
    plt.close()


def test_dense_packing():
    """测试5：密集平铺排列"""
    print("\n=== 测试5：密集平铺排列 ===")

    _, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 定义矩形边界
    rect_width = 4
    rect_height = 4

    # 创建密集排列的二聚体
    dimers = []

    # 不同的排列策略
    # 策略1：水平排列（0°）
    for i in range(3):
        for j in range(2):
            dimer = create_standard_dimer()
            x = -1.5 + i * 0.8
            y = 1.2 + j * 1.0
            dimer.translate(Decimal(str(x)), Decimal(str(y)))
            dimer.rotate(Decimal('0'))
            dimers.append(('horizontal', dimer))

    # 策略2：垂直排列（90°）
    for i in range(2):
        for j in range(3):
            dimer = create_standard_dimer()
            x = -1.8 + i * 1.0
            y = -1.5 + j * 0.8
            dimer.translate(Decimal(str(x)), Decimal(str(y)))
            dimer.rotate(Decimal('90'))
            dimers.append(('vertical', dimer))

    # 策略3：对角排列（45°）
    for i in range(3):
        dimer = create_standard_dimer()
        x = 0.8 + i * 0.8
        y = -1.5 + i * 0.8
        dimer.translate(Decimal(str(x)), Decimal(str(y)))
        dimer.rotate(Decimal('45'))
        dimers.append(('diagonal', dimer))

    # 策略4：混合角度填充空隙
    extra_positions = [
        (1.2, 1.0, 30),
        (0.5, 0.8, 120),
        (1.5, 0.2, 60),
    ]
    for x, y, angle in extra_positions:
        dimer = create_standard_dimer()
        dimer.translate(Decimal(str(x)), Decimal(str(y)))
        dimer.rotate(Decimal(str(angle)))
        dimers.append(('fill', dimer))

    # 绘制所有二聚体
    strategy_colors = {
        'horizontal': ('lightblue', 'lightcyan'),
        'vertical': ('lightcoral', 'lightpink'),
        'diagonal': ('lightgreen', 'palegreen'),
        'fill': ('lightyellow', 'wheat')
    }

    for strategy, dimer in dimers:
        color1, color2 = strategy_colors[strategy]
        plot_tree(ax, dimer.tree_a, color=color1)
        plot_tree(ax, dimer.tree_b, color=color2)

        # 标记二聚体中心（小点）
        cx = float(dimer.center_x)
        cy = float(dimer.center_y)
        ax.plot(cx, cy, 'k.', markersize=2)

    # 绘制矩形边界
    rect = patches.Rectangle((-rect_width/2, -rect_height/2), rect_width, rect_height,
                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # 计算总边界框
    all_polygons = []
    for _, dimer in dimers:
        all_polygons.extend(dimer.get_polygons())

    if all_polygons:
        minx, miny, maxx, maxy = all_polygons[0].bounds
        for poly in all_polygons[1:]:
            b = poly.bounds
            minx = min(minx, b[0])
            miny = min(miny, b[1])
            maxx = max(maxx, b[2])
            maxy = max(maxy, b[3])

        # 转换回逻辑坐标
        minx /= float(scale_factor)
        miny /= float(scale_factor)
        maxx /= float(scale_factor)
        maxy /= float(scale_factor)

        # 绘制实际边界框
        actual_width = maxx - minx
        actual_height = maxy - miny
        actual_side = max(actual_width, actual_height)

        bbox_rect = patches.Rectangle((minx, miny), actual_width, actual_height,
                                     linewidth=2, edgecolor='blue', facecolor='none', linestyle='-')
        ax.add_patch(bbox_rect)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Test 5: Dense Packing - {len(dimers)} Dimers')

    # 添加图例和统计
    legend_text = f"Total Dimers: {len(dimers)}\n"
    legend_text += f"Total Trees: {len(dimers)*2}\n"
    if all_polygons:
        legend_text += f"Bounding Box: {actual_side:.3f}\n"
        legend_text += f"Density: {len(dimers)*2/actual_side:.2f} trees/unit"

    # 策略图例
    strategy_labels = {
        'horizontal': 'Horizontal (0°)',
        'vertical': 'Vertical (90°)',
        'diagonal': 'Diagonal (45°)',
        'fill': 'Fill angles'
    }

    legend_lines = []
    for strategy, label in strategy_labels.items():
        count = sum(1 for s, _ in dimers if s == strategy)
        legend_text += f"\n{label}: {count}"

    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test5_dense_packing.png', dpi=150)
    print("保存: test5_dense_packing.png")
    print(f"  总二聚体数: {len(dimers)}")
    print(f"  总树数: {len(dimers)*2}")
    if all_polygons:
        print(f"  边界框边长: {actual_side:.3f}")
        print(f"  密度: {len(dimers)*2/actual_side:.2f} trees/unit")
    plt.close()


def test_rigid_structure():
    """测试6：验证刚性结构（距离不变）"""
    print("\n=== 测试6：验证刚性结构 ===")

    # 创建标准二聚体
    dimer = create_standard_dimer()

    # 计算初始距离
    dx = float(dimer.tree_b.center_x - dimer.tree_a.center_x)
    dy = float(dimer.tree_b.center_y - dimer.tree_a.center_y)
    initial_dist = math.sqrt(dx*dx + dy*dy)

    print(f"  初始距离: {initial_dist:.10f}")

    # 连续旋转和平移
    operations = [
        ('rotate', 30),
        ('translate', (0.5, 0.3)),
        ('rotate', 45),
        ('translate', (-0.2, 0.7)),
        ('rotate', 90),
        ('translate', (1.0, -0.5)),
        ('rotate', 120),
    ]

    distances = [initial_dist]

    for op_type, param in operations:
        if op_type == 'rotate':
            dimer.rotate(Decimal(str(param)))
        else:
            dimer.translate(Decimal(str(param[0])), Decimal(str(param[1])))

        dx = float(dimer.tree_b.center_x - dimer.tree_a.center_x)
        dy = float(dimer.tree_b.center_y - dimer.tree_a.center_y)
        current_dist = math.sqrt(dx*dx + dy*dy)
        distances.append(current_dist)

        error = abs(current_dist - initial_dist)
        status = "✓" if error < 1e-6 else "✗"
        print(f"  {status} {op_type:10s} {str(param):20s} 距离: {current_dist:.10f}, 误差: {error:.12f}")

    # 绘制距离变化图
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(distances, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=initial_dist, color='r', linestyle='--', label='Initial Distance')
    ax.set_xlabel('Operation Index')
    ax.set_ylabel('Distance between Trees')
    ax.set_title('Test 5: Rigid Structure Verification')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 添加操作标签
    for i, (op_type, param) in enumerate(operations, 1):
        label = f"{op_type[:3]}({param})"
        ax.text(i, distances[i], label, fontsize=8, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/Users/zbr/code/santa2025/solution/test6_rigid.png', dpi=150)
    print("保存: test6_rigid.png")
    plt.close()

    # 计算统计
    errors = [abs(d - initial_dist) for d in distances]
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)

    print(f"\n  统计:")
    print(f"    最大误差: {max_error:.12f}")
    print(f"    平均误差: {avg_error:.12f}")
    print(f"    刚性验证: {'通过 ✓' if max_error < 1e-6 else '失败 ✗'}")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Dimer 类测试和可视化")
    print("=" * 60)

    test_basic_dimer()
    test_rotation()
    test_translation()
    test_dimers_in_rectangle()
    test_dense_packing()
    test_rigid_structure()

    print("\n" + "=" * 60)
    print("所有测试完成！请查看生成的图片文件。")
    print("=" * 60)


if __name__ == '__main__':
    main()
