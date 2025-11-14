'''
Author: Diana Tang
Date: 2025-11-14 23:46:01
LastEditors: Diana Tang
Description: some description
FilePath: /PGENSCORE5-Data-Clean/demo.py
'''
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