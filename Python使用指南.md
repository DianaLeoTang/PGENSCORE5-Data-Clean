# HRS 多基因评分数据集 Python 使用指南

本指南介绍如何使用 Python 读取和分析 HRS 多基因评分数据集，无需使用 SAS、SPSS 或 Stata 等统计软件。

## 目录

1. [环境准备](#环境准备)
2. [读取数据文件](#读取数据文件)
3. [数据预处理](#数据预处理)
4. [基本分析示例](#基本分析示例)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装必要的 Python 库

```bash
# 基础数据处理
pip install pandas numpy

# 读取 Stata 文件（.dta）- pandas 内置支持
# pandas 可以直接读取 .dta 文件，无需额外安装

# 读取 SPSS 文件（.sav）- 需要 pyreadstat
pip install pyreadstat

# 读取 SAS 文件（.sas7bdat）- 需要 pyreadstat
pip install pyreadstat

# 统计分析（可选）
pip install scipy scikit-learn statsmodels

# 可视化（可选）
pip install matplotlib seaborn
```

### 2. 导入必要的库

```python
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 如果需要读取 SPSS 或 SAS 文件
try:
    import pyreadstat
except ImportError:
    print("请安装 pyreadstat: pip install pyreadstat")
```

---

## 读取数据文件

### 方法 1: 读取 Stata 格式文件（推荐，最简单）

Stata 格式（`.dta`）是**最推荐**的方式，因为：
- pandas 原生支持，无需额外库
- 读取速度快
- 数据类型自动识别
- 变量标签和值标签保留完整

```python
import pandas as pd

# 设置数据路径
data_dir = Path("built/stata")

# 读取非洲祖先数据
df_aa = pd.read_stata(data_dir / "PGENSCOREA_R.dta")

# 读取欧洲祖先数据
df_ea = pd.read_stata(data_dir / "PGENSCOREE_R.dta")

# 读取西班牙裔祖先数据
df_ha = pd.read_stata(data_dir / "PGENSCOREH_R.dta")

# 查看数据基本信息
print(f"非洲祖先数据形状: {df_aa.shape}")
print(f"欧洲祖先数据形状: {df_ea.shape}")
print(f"西班牙裔祖先数据形状: {df_ha.shape}")

# 查看前几行
print(df_aa.head())

# 查看列名
print(df_aa.columns.tolist())
```

### 方法 2: 读取 SPSS 格式文件（.sav）

```python
import pyreadstat

# 读取 SPSS 文件
df_aa, meta = pyreadstat.read_sav("built/spss/PGENSCOREA_R.sav")

# meta 包含元数据信息（变量标签、值标签等）
print("变量标签:", meta.column_labels)
print("变量名:", meta.column_names)

# 查看数据
print(df_aa.head())
```

### 方法 3: 读取 SAS 格式文件（.sas7bdat）

```python
import pyreadstat

# 读取 SAS 文件
df_aa, meta = pyreadstat.read_sas7bdat("built/sas/PGENSCOREA_R.sas7bdat")

# 查看数据
print(df_aa.head())
print("变量标签:", meta.column_labels)
```

### 方法 4: 读取原始固定宽度文件（.da）

如果上述格式都无法使用，可以直接读取原始固定宽度文件：

```python
import pandas as pd

# 定义列宽和列名（以非洲祖先数据为例）
# 根据 .dct 文件定义列的位置和宽度
colspecs = [
    (0, 6),    # HHID: 位置 0-5 (6个字符)
    (6, 9),    # PN: 位置 6-8 (3个字符)
    (9, 21),   # PC1_5A: 位置 9-20 (12个字符，包含小数点)
    (21, 33),  # PC1_5B
    (33, 45),  # PC1_5C
    # ... 继续定义所有列
]

colnames = [
    'HHID', 'PN', 'PC1_5A', 'PC1_5B', 'PC1_5C',
    # ... 继续添加所有列名
]

# 读取固定宽度文件
df_aa = pd.read_fwf(
    "polys/data/PGENSCOREA_R.da",
    colspecs=colspecs,
    names=colnames,
    na_values=['', ' ']
)

# 将数值列转换为浮点数
numeric_cols = df_aa.columns[2:]  # 从第3列开始都是数值
df_aa[numeric_cols] = df_aa[numeric_cols].apply(pd.to_numeric, errors='coerce')
```

**注意**: 手动定义所有列比较繁琐。建议使用 `.dta` 格式，这是最简单的方法。

---

## 数据预处理

### 1. 合并多个祖先群体的数据

```python
# 为每个数据集添加祖先标识
df_aa['ancestry'] = 'African'
df_ea['ancestry'] = 'European'
df_ha['ancestry'] = 'Hispanic'

# 合并数据
df_all = pd.concat([df_aa, df_ea, df_ha], ignore_index=True)

print(f"合并后数据形状: {df_all.shape}")
print(f"祖先分布:\n{df_all['ancestry'].value_counts()}")
```

### 2. 检查缺失值

```python
# 检查缺失值
missing = df_aa.isnull().sum()
missing_pct = (missing / len(df_aa)) * 100

# 显示缺失值超过0%的列
missing_info = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percentage': missing_pct
})
missing_info = missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("缺失值信息:")
print(missing_info)
```

### 3. 选择主成分变量

```python
# 选择前5个主成分（PC1-5）
pc_cols_1_5 = [col for col in df_aa.columns if 'PC1_5' in col]
print("PC1-5 变量:", pc_cols_1_5)

# 选择后5个主成分（PC6-10）
pc_cols_6_10 = [col for col in df_aa.columns if 'PC6_10' in col]
print("PC6-10 变量:", pc_cols_6_10)

# 选择所有主成分
pc_cols_all = pc_cols_1_5 + pc_cols_6_10
```

### 4. 选择特定类别的 PGS 变量

```python
# 选择所有 PGS 变量（排除标识变量和主成分）
id_cols = ['HHID', 'PN', 'VERSION']
pgs_cols = [col for col in df_aa.columns 
             if col not in id_cols and not col.startswith('PC')]

print(f"共有 {len(pgs_cols)} 个 PGS 变量")

# 按类别选择 PGS
# 例如：选择所有与糖尿病相关的 PGS
t2d_cols = [col for col in pgs_cols if 'T2D' in col or 'DIABETES' in col.upper()]
print("糖尿病相关 PGS:", t2d_cols)

# 选择所有与认知相关的 PGS
cog_cols = [col for col in pgs_cols if 'COG' in col or 'COGNITION' in col.upper()]
print("认知相关 PGS:", cog_cols)

# 选择所有与精神健康相关的 PGS
mental_cols = [col for col in pgs_cols 
               if any(x in col for x in ['MDD', 'SCZ', 'BIP', 'ADHD', 'AUTISM', 'OCD', 'PTSD', 'ANX'])]
print("精神健康相关 PGS:", mental_cols)
```

### 5. 数据标准化检查

```python
# 检查 PGS 是否已标准化（均值应该接近0，标准差应该接近1）
pgs_sample = df_aa[pgs_cols[:10]]  # 选择前10个PGS作为示例

stats = pd.DataFrame({
    'Mean': pgs_sample.mean(),
    'Std': pgs_sample.std(),
    'Min': pgs_sample.min(),
    'Max': pgs_sample.max()
})

print("PGS 统计摘要（前10个变量）:")
print(stats)
```

---

## 基本分析示例

### 示例 1: 描述性统计

```python
# 选择感兴趣的 PGS 变量
pgs_of_interest = [
    'A5_BMI_GIANT15',
    'A5_HEIGHT_GIANT14',
    'A5_T2D_DIAGRAM12',
    'A5_GENCOG_CHARGE15',
    'A5_MDD_PGC13'
]

# 描述性统计
desc_stats = df_aa[pgs_of_interest].describe()
print(desc_stats)
```

### 示例 2: 相关性分析

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 计算 PGS 之间的相关性
correlation_matrix = df_aa[pgs_of_interest].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('PGS 相关性热力图')
plt.tight_layout()
plt.show()
```

### 示例 3: 使用主成分控制混杂因素

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 假设我们要分析 BMI PGS 对某个结局的影响
# 这里用另一个 PGS 作为示例结局变量

# 准备数据
X = df_aa[['A5_BMI_GIANT15'] + pc_cols_1_5]  # 自变量：BMI PGS + 前5个主成分
y = df_aa['A5_HEIGHT_GIANT14']  # 因变量：身高 PGS（示例）

# 删除缺失值
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

# 拟合线性回归模型
model = LinearRegression()
model.fit(X_clean, y_clean)

# 查看结果
print(f"R² = {model.score(X_clean, y_clean):.4f}")
print(f"BMI PGS 系数 = {model.coef_[0]:.4f}")
print(f"截距 = {model.intercept_:.4f}")
```

### 示例 4: 按祖先群体分组分析

```python
# 如果合并了多个祖先群体的数据
# 可以按祖先群体分组分析

# 按祖先群体计算均值
grouped_stats = df_all.groupby('ancestry')[pgs_of_interest].mean()
print("按祖先群体的 PGS 均值:")
print(grouped_stats)

# 按祖先群体计算标准差
grouped_std = df_all.groupby('ancestry')[pgs_of_interest].std()
print("\n按祖先群体的 PGS 标准差:")
print(grouped_std)
```

### 示例 5: 保存处理后的数据

```python
# 保存为 CSV 格式（便于后续使用）
df_aa.to_csv('PGENSCOREA_R_processed.csv', index=False)

# 保存为 Parquet 格式（更高效，保留数据类型）
df_aa.to_parquet('PGENSCOREA_R_processed.parquet', index=False)

# 保存为 Excel 格式（便于查看）
df_aa.to_excel('PGENSCOREA_R_processed.xlsx', index=False)
```

---

## 完整示例脚本

```python
"""
HRS 多基因评分数据集读取和分析示例
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 1. 读取数据 ====================
print("=" * 50)
print("步骤 1: 读取数据")
print("=" * 50)

data_dir = Path("built/stata")

# 读取三个祖先群体的数据
df_aa = pd.read_stata(data_dir / "PGENSCOREA_R.dta")
df_ea = pd.read_stata(data_dir / "PGENSCOREE_R.dta")
df_ha = pd.read_stata(data_dir / "PGENSCOREH_R.dta")

print(f"✓ 非洲祖先数据: {df_aa.shape[0]} 行, {df_aa.shape[1]} 列")
print(f"✓ 欧洲祖先数据: {df_ea.shape[0]} 行, {df_ea.shape[1]} 列")
print(f"✓ 西班牙裔祖先数据: {df_ha.shape[0]} 行, {df_ha.shape[1]} 列")

# ==================== 2. 数据探索 ====================
print("\n" + "=" * 50)
print("步骤 2: 数据探索")
print("=" * 50)

# 查看列名
print(f"\n总列数: {len(df_aa.columns)}")
print(f"\n前10个变量: {df_aa.columns[:10].tolist()}")

# 识别 PGS 变量
id_cols = ['HHID', 'PN', 'VERSION']
pc_cols = [col for col in df_aa.columns if col.startswith('PC')]
pgs_cols = [col for col in df_aa.columns 
            if col not in id_cols and not col.startswith('PC')]

print(f"\n标识变量: {len(id_cols)} 个")
print(f"主成分变量: {len(pc_cols)} 个")
print(f"PGS 变量: {len(pgs_cols)} 个")

# ==================== 3. 数据质量检查 ====================
print("\n" + "=" * 50)
print("步骤 3: 数据质量检查")
print("=" * 50)

# 检查缺失值
missing = df_aa[pgs_cols].isnull().sum()
missing_vars = missing[missing > 0]

if len(missing_vars) > 0:
    print(f"\n有缺失值的变量 ({len(missing_vars)} 个):")
    for var, count in missing_vars.items():
        pct = (count / len(df_aa)) * 100
        print(f"  {var}: {count} ({pct:.1f}%)")
else:
    print("\n✓ 所有 PGS 变量均无缺失值")

# 检查数据标准化
sample_pgs = pgs_cols[:5]
stats = df_aa[sample_pgs].agg(['mean', 'std', 'min', 'max'])
print(f"\n前5个 PGS 的统计摘要:")
print(stats.round(4))

# ==================== 4. 选择感兴趣的变量 ====================
print("\n" + "=" * 50)
print("步骤 4: 选择感兴趣的变量")
print("=" * 50)

# 选择特定类别的 PGS
categories = {
    '代谢': [col for col in pgs_cols if any(x in col for x in ['BMI', 'HEIGHT', 'WC', 'WHR'])],
    '糖尿病': [col for col in pgs_cols if 'T2D' in col],
    '认知': [col for col in pgs_cols if 'COG' in col or 'EDU' in col],
    '精神健康': [col for col in pgs_cols if any(x in col for x in ['MDD', 'SCZ', 'BIP', 'ADHD'])],
    '心血管': [col for col in pgs_cols if any(x in col for x in ['CAD', 'MI', 'BP', 'HTN'])],
}

for category, vars_list in categories.items():
    print(f"{category}: {len(vars_list)} 个变量")
    if len(vars_list) > 0:
        print(f"  示例: {vars_list[0]}")

# ==================== 5. 基本分析 ====================
print("\n" + "=" * 50)
print("步骤 5: 基本分析")
print("=" * 50)

# 选择几个代表性的 PGS
selected_pgs = [
    'A5_BMI_GIANT15',
    'A5_T2D_DIAGRAM12',
    'A5_GENCOG_CHARGE15',
]

# 描述性统计
desc = df_aa[selected_pgs].describe()
print("\n描述性统计:")
print(desc.round(4))

# 相关性
corr = df_aa[selected_pgs].corr()
print("\n相关性矩阵:")
print(corr.round(4))

print("\n" + "=" * 50)
print("分析完成！")
print("=" * 50)
```

---

## 常见问题

### Q1: 如何读取特定祖先群体的数据？

```python
# 根据你的研究需求选择对应的数据集
# 非洲祖先
df = pd.read_stata("built/stata/PGENSCOREA_R.dta")

# 欧洲祖先
df = pd.read_stata("built/stata/PGENSCOREE_R.dta")

# 西班牙裔祖先
df = pd.read_stata("built/stata/PGENSCOREH_R.dta")
```

### Q2: 如何找到特定疾病或性状的 PGS？

```python
# 方法1: 使用关键词搜索
keyword = 'DIABETES'  # 或 'T2D', 'COGNITION', 'DEPRESSION' 等
matching_cols = [col for col in df_aa.columns if keyword.upper() in col.upper()]
print(matching_cols)

# 方法2: 使用正则表达式
import re
pattern = re.compile(r'T2D|DIABETES', re.IGNORECASE)
matching_cols = [col for col in df_aa.columns if pattern.search(col)]
print(matching_cols)
```

### Q3: 如何使用主成分控制混杂因素？

```python
# 必须同时使用所有5个或所有10个主成分
# 不能单独使用某个主成分

# 使用前5个主成分
pc_cols = [f'PC1_5{i}' for i in ['A', 'B', 'C', 'D', 'E']]

# 在回归分析中包含这些主成分
# 例如：
# model = LinearRegression()
# X = df[['your_pgs'] + pc_cols]
# model.fit(X, y)
```

### Q4: 数据已经标准化了吗？

是的，所有 PGS 变量都已标准化（均值为0，标准差为1）。可以直接用于回归分析，无需再次标准化。

### Q5: 如何合并多个祖先群体的数据进行分析？

```python
# 添加祖先标识
df_aa['ancestry'] = 'African'
df_ea['ancestry'] = 'European'
df_ha['ancestry'] = 'Hispanic'

# 合并
df_all = pd.concat([df_aa, df_ea, df_ha], ignore_index=True)

# 在分析中控制祖先群体
# 例如使用虚拟变量或分层分析
```

### Q6: 如何保存和加载处理后的数据？

```python
# 保存为多种格式
df_aa.to_csv('data_processed.csv', index=False)  # CSV
df_aa.to_parquet('data_processed.parquet', index=False)  # Parquet（推荐，高效）
df_aa.to_pickle('data_processed.pkl')  # Pickle（保留所有信息）

# 加载
df = pd.read_csv('data_processed.csv')
df = pd.read_parquet('data_processed.parquet')
df = pd.read_pickle('data_processed.pkl')
```

---

## 推荐工作流程

1. **使用 Stata 格式（.dta）** - 最简单，pandas 原生支持
2. **先读取单个祖先群体** - 根据研究需求选择
3. **检查数据质量** - 缺失值、数据范围等
4. **选择感兴趣的 PGS** - 根据研究假设
5. **包含主成分** - 控制群体分层
6. **进行分析** - 回归、相关性等
7. **保存结果** - 便于后续使用

---

## 参考资料

- [pandas 文档](https://pandas.pydata.org/docs/)
- [pyreadstat 文档](https://github.com/Roche/pyreadstat)
- HRS 官方文档: `docs/pgenscoreddv5.pdf`

---

**最后更新**: 2024年10月

