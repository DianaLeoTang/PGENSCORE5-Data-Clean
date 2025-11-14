# Python 快速开始指南

## 最简单的使用方法

### 1. 安装依赖

```bash
# 最小安装（只需要读取 .dta 文件）
pip install pandas numpy

# 或者安装所有依赖
pip install -r requirements.txt
```

### 2. 读取数据（3行代码）

```python
import pandas as pd

# 读取数据
df = pd.read_stata("built/stata/PGENSCOREA_R.dta")

# 完成！现在可以使用 df 进行分析了
```

### 3. 运行示例脚本

```bash
# 快速开始（最简单）
python quick_start.py

# 完整示例（包含数据探索和分析）
python read_hrs_data.py
```

## 文件说明

- **`quick_start.py`** - 最简单的示例，3行代码读取数据
- **`read_hrs_data.py`** - 完整的数据读取和探索脚本
- **`Python使用指南.md`** - 详细的使用文档
- **`requirements.txt`** - Python 依赖包列表

## 数据文件位置

- **Stata 格式（推荐）**: `built/stata/PGENSCOREA_R.dta` (非洲祖先)
- **Stata 格式**: `built/stata/PGENSCOREE_R.dta` (欧洲祖先)
- **Stata 格式**: `built/stata/PGENSCOREH_R.dta` (西班牙裔祖先)

## 快速示例

### 示例 1: 读取数据并查看基本信息

```python
import pandas as pd

# 读取数据
df = pd.read_stata("built/stata/PGENSCOREA_R.dta")

# 查看基本信息
print(df.shape)  # 数据维度
print(df.head())  # 前5行
print(df.columns.tolist())  # 所有列名
```

### 示例 2: 选择特定变量

```python
# 选择标识变量和几个 PGS
selected_cols = ['HHID', 'PN', 'A5_BMI_GIANT15', 'A5_T2D_DIAGRAM12', 'A5_GENCOG_CHARGE15']
df_subset = df[selected_cols]

print(df_subset.head())
```

### 示例 3: 描述性统计

```python
# 选择所有 PGS 变量（排除标识变量和主成分）
pgs_cols = [col for col in df.columns 
            if col not in ['HHID', 'PN', 'VERSION'] 
            and not col.startswith('PC')]

# 描述性统计
print(df[pgs_cols[:10]].describe())
```

### 示例 4: 相关性分析

```python
# 选择几个 PGS
pgs_of_interest = ['A5_BMI_GIANT15', 'A5_HEIGHT_GIANT14', 'A5_T2D_DIAGRAM12']

# 计算相关性
correlation = df[pgs_of_interest].corr()
print(correlation)
```

### 示例 5: 使用主成分控制混杂因素

```python
from sklearn.linear_model import LinearRegression

# 选择主成分（必须使用完整的 PC1-5 或 PC1-10）
pc_cols = [col for col in df.columns if 'PC1_5' in col]

# 准备数据
X = df[['A5_BMI_GIANT15'] + pc_cols]  # PGS + 主成分
y = df['A5_HEIGHT_GIANT14']  # 结局变量

# 删除缺失值
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

# 拟合模型
model = LinearRegression()
model.fit(X_clean, y_clean)

print(f"R² = {model.score(X_clean, y_clean):.4f}")
print(f"BMI PGS 系数 = {model.coef_[0]:.4f}")
```

## 常见问题

**Q: 我应该使用哪个数据文件？**

A: 根据你的研究需求：
- 如果研究非洲裔人群，使用 `PGENSCOREA_R.dta`
- 如果研究欧洲裔人群，使用 `PGENSCOREE_R.dta`
- 如果研究西班牙裔人群，使用 `PGENSCOREH_R.dta`
- 如果需要合并分析，可以分别读取后合并

**Q: 如何找到特定疾病或性状的 PGS？**

```python
# 使用关键词搜索
keyword = 'DIABETES'  # 或 'T2D', 'COGNITION', 'DEPRESSION' 等
matching = [col for col in df.columns if keyword.upper() in col.upper()]
print(matching)
```

**Q: 数据已经标准化了吗？**

A: 是的，所有 PGS 变量都已标准化（均值=0，标准差=1），可以直接用于分析。

**Q: 如何使用主成分？**

A: 必须同时使用所有5个PC（PC1-5）或所有10个PC（PC1-10），不能单独使用某个主成分。

```python
# 正确：使用所有 PC1-5
pc_cols = [col for col in df.columns if 'PC1_5' in col]

# 错误：只使用一个 PC
# pc_col = 'PC1_5A'  # 不要这样做！
```

## 更多信息

详细的使用说明请参考：
- **`Python使用指南.md`** - 完整的使用文档
- **`项目说明.md`** - 数据集总体说明

## 技术支持

如有问题，可以：
1. 查看 `Python使用指南.md` 中的常见问题部分
2. 查看 pandas 官方文档: https://pandas.pydata.org/docs/
3. 联系 HRS: hrsquestions@umich.edu

