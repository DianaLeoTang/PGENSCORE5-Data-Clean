'''
Author: Diana Tang
Date: 2025-11-14 22:28:43
LastEditors: Diana Tang
Description: some description
FilePath: /PGENSCORE5-Data-Clean/quick_start.py
'''

"""
HRS 多基因评分数据集 - 快速开始脚本

最简单的使用方式，直接运行即可
"""

import pandas as pd
from pathlib import Path

# 设置路径
data_file = Path("built/stata/PGENSCOREA_R.dta")  # 可以改为 PGENSCOREE_R.dta 或 PGENSCOREH_R.dta

# 读取数据（一行代码搞定！）
df = pd.read_stata(data_file)

# 查看数据
print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())

print("\n列名（前20个）:")
print(df.columns[:20].tolist())

# 现在你可以使用 df 进行分析了！
# 例如：
# - df.describe()  # 描述性统计
# - df['A5_BMI_GIANT15']  # 访问特定列
# - df[['HHID', 'PN', 'A5_BMI_GIANT15']]  # 选择多列

