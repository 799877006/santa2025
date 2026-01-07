# SANTA2025 初始解生成器使用指南

## 功能说明

`init_generator.py` 使用二聚体密铺策略生成SANTA2025竞赛的初始解。

### 核心特性

1. **二聚体密铺**：自动使用标准二聚体构型 (0,0,0°) + (0.35,0.8,180°) 进行网格排列
2. **自动优化布局**：交替旋转角度（0°/90°）以提高密度
3. **灵活生成**：支持生成全新解或更新现有解中的特定组

## 使用方法

### 1. 生成示例解（默认模式）

直接运行，生成1-10组的初始解：

```bash
python init_generator.py
```

输出：`initial_solution.csv` (包含1-10组，共55棵树)

### 2. 生成所有组（1-200）

```bash
python init_generator.py --mode generate --output full_solution.csv --groups all
```

### 3. 生成特定范围的组

```bash
# 生成第1-50组
python init_generator.py --mode generate --output solution_1_50.csv --groups 1-50

# 生成第100-150组
python init_generator.py --mode generate --output solution_100_150.csv --groups 100-150
```

### 4. 生成指定的几个组

```bash
# 生成第1,5,10,20组
python init_generator.py --mode generate --output solution_custom.csv --groups 1,5,10,20
```

### 5. 更新现有解中的特定组

```bash
# 读取best_result_sa_round20.csv，只重新生成第1-10组，其他组保持不变
python init_generator.py \
  --mode update \
  --input best_result_sa_round20.csv \
  --output updated_solution.csv \
  --groups 1-10
```

### 6. 更新特定组（保持其他组不变）

```bash
# 只重新生成第50,100,150组
python init_generator.py \
  --mode update \
  --input current_solution.csv \
  --output updated_solution.csv \
  --groups 50,100,150
```

## 命令行参数

| 参数 | 必需 | 说明 | 示例 |
|------|-----|------|------|
| `--mode` | 否 | 模式：`generate`(生成新解) 或 `update`(更新现有解) | `--mode generate` |
| `--input` | 条件 | 输入CSV文件（`update`模式必需） | `--input best_result.csv` |
| `--output` | 是 | 输出CSV文件路径 | `--output solution.csv` |
| `--groups` | 是 | 要生成/更新的组号 | `--groups all` 或 `1-200` 或 `1,5,10` |

## CSV格式

生成的CSV文件格式与竞赛要求一致：

```csv
id,x,y,deg
001_0,s0.0,s0.0,s0
002_0,s-0.04,s-0.04,s0
002_1,s0.31,s0.76,s180
...
```

- `id`: 组号_树编号 (格式: `NNN_M`)
- `x`, `y`, `deg`: 坐标和角度，前缀`s`

## 算法说明

### 二聚体构型

使用固定的标准二聚体：
- **树A**: `(0, 0, 0°)` - 正常朝上
- **树B**: `(0.35, 0.8, 180°)` - 倒置

两树形成紧密的"头对头"排列，距离约0.873单位。

### 密铺策略

1. **网格排列**：按0.9单位间距放置二聚体
2. **交替旋转**：棋盘式交替0°/90°旋转，提高密度
3. **自动调整**：根据树数量自动计算矩形大小
4. **单树处理**：奇数树时，最后一棵单独放置在原点

### 边界框估算

- 每个二聚体估算占用 1.0 单位²
- 矩形边长 = √(总面积) × 1.5（留50%余量）

## 示例工作流

### 场景1：从零开始生成完整解

```bash
# 步骤1：生成所有组的初始解
python init_generator.py --mode generate --output init_all.csv --groups all

# 步骤2：使用SA优化（假设已有SA脚本）
python SA.py --input init_all.csv --output sa_result.csv

# 步骤3：如果某些组效果不好，重新生成
python init_generator.py --mode update --input sa_result.csv --output sa_result_updated.csv --groups 1-20
```

### 场景2：更新现有解中的特定组

```bash
# 假设已有best_result_sa_round20.csv，想重新优化前30组
python init_generator.py \
  --mode update \
  --input /Users/zbr/code/santa2025/solution/best_result_sa_round20.csv \
  --output /Users/zbr/code/santa2025/solution/best_result_reinit.csv \
  --groups 1-30
```

## 注意事项

1. **更新模式安全**：`update`模式会保留其他组的数据，只替换指定组
2. **ID排序**：输出CSV自动按组号和树编号排序
3. **数量保证**：确保每组生成正确数量的树
4. **奇数树**：奇数组最后一棵树单独放置在原点
5. **性能**：生成200组约需10-30秒

## 输出示例

运行后会显示进度：

```
============================================================
SANTA2025 Initial Solution Generator
============================================================

Processing group 001 (1 trees)...
Generating 1 trees with dimer packing...
  Dimers: 0
  Single trees: 1
  Rectangle size: 0.000
  Adding 1 single tree...
  Generated: 1 trees

Processing group 002 (2 trees)...
Generating 2 trees with dimer packing...
  Dimers: 1
  Single trees: 0
  Rectangle size: 1.500
  Generated: 2 trees

...

Saved: initial_solution.csv
Total rows: 20100
```

## 故障排查

### 问题：生成的树数量不对

**原因**：矩形边界太小
**解决**：代码已自动调整，确保生成足够的树

### 问题：CSV格式不对

**原因**：可能是编码问题
**解决**：确保使用UTF-8编码，文件开头有 `# -*- coding: utf-8 -*-`

### 问题：update模式失败

**原因**：输入文件路径错误或格式不匹配
**解决**：
1. 检查文件路径是否正确
2. 确认输入CSV有`id,x,y,deg`列
3. 确认ID格式为`NNN_M`

## 代码集成

如果要在Python脚本中使用：

```python
from init_generator import generate_initial_solution, update_specific_groups

# 生成1-50组
df = generate_initial_solution(
    n_list=list(range(1, 51)),
    output_csv='solution.csv'
)

# 更新特定组
df = update_specific_groups(
    input_csv='existing.csv',
    output_csv='updated.csv',
    n_list=[10, 20, 30]
)
```

## 后续优化建议

生成初始解后，建议使用SA (Simulated Annealing) 进行优化：

```bash
python SA.py --input initial_solution.csv --output optimized_solution.csv
```

## 联系与反馈

如有问题或建议，请查看源代码注释或修改参数进行调试。
