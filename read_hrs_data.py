"""
HRS 多基因评分数据集读取和分析示例脚本

使用方法:
1. 确保已安装必要的库: pip install pandas numpy
2. 运行脚本: python read_hrs_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ==================== 配置 ====================
# 设置数据文件路径（相对于脚本位置）
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "built" / "stata"

# 选择要读取的祖先群体: 'A' (非洲), 'E' (欧洲), 'H' (西班牙裔), 或 'all' (全部)
ANCESTRY = 'A'  # 可以改为 'E', 'H', 或 'all'

# ==================== 函数定义 ====================

def load_hrs_data(ancestry='A'):
    """
    加载 HRS 多基因评分数据
    
    参数:
        ancestry: 'A' (非洲), 'E' (欧洲), 'H' (西班牙裔), 或 'all' (全部)
    
    返回:
        pandas DataFrame 或字典（如果 ancestry='all'）
    """
    files = {
        'A': 'PGENSCOREA_R.dta',
        'E': 'PGENSCOREE_R.dta',
        'H': 'PGENSCOREH_R.dta'
    }
    
    names = {
        'A': 'African',
        'E': 'European',
        'H': 'Hispanic'
    }
    
    if ancestry == 'all':
        data = {}
        for key, filename in files.items():
            filepath = DATA_DIR / filename
            if filepath.exists():
                print(f"正在读取 {names[key]} 祖先数据: {filename}")
                data[key] = pd.read_stata(filepath)
                data[key]['ancestry'] = names[key]
                print(f"  ✓ 成功加载: {data[key].shape[0]} 行, {data[key].shape[1]} 列")
            else:
                print(f"  ✗ 文件不存在: {filepath}")
        return data
    else:
        if ancestry not in files:
            raise ValueError(f"ancestry 必须是 'A', 'E', 'H', 或 'all'")
        
        filepath = DATA_DIR / files[ancestry]
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        print(f"正在读取 {names[ancestry]} 祖先数据: {files[ancestry]}")
        df = pd.read_stata(filepath)
        df['ancestry'] = names[ancestry]
        print(f"✓ 成功加载: {df.shape[0]} 行, {df.shape[1]} 列")
        return df


def explore_data(df):
    """探索数据基本信息"""
    print("\n" + "=" * 60)
    print("数据探索")
    print("=" * 60)
    
    # 基本信息
    print(f"\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 变量分类
    id_cols = ['HHID', 'PN', 'VERSION']
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    pgs_cols = [col for col in df.columns 
                if col not in id_cols and not col.startswith('PC') and col != 'ancestry']
    
    print(f"\n变量分类:")
    print(f"  标识变量: {len([c for c in id_cols if c in df.columns])} 个")
    print(f"  主成分变量: {len(pc_cols)} 个")
    print(f"  PGS 变量: {len(pgs_cols)} 个")
    
    # 显示前几个变量名
    print(f"\n前10个变量名:")
    for i, col in enumerate(df.columns[:10], 1):
        print(f"  {i:2d}. {col}")
    
    return id_cols, pc_cols, pgs_cols


def check_data_quality(df, pgs_cols):
    """检查数据质量"""
    print("\n" + "=" * 60)
    print("数据质量检查")
    print("=" * 60)
    
    # 检查缺失值
    missing = df[pgs_cols].isnull().sum()
    missing_vars = missing[missing > 0]
    
    if len(missing_vars) > 0:
        print(f"\n有缺失值的 PGS 变量 ({len(missing_vars)} 个):")
        for var, count in missing_vars.head(10).items():
            pct = (count / len(df)) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")
        if len(missing_vars) > 10:
            print(f"  ... 还有 {len(missing_vars) - 10} 个变量有缺失值")
    else:
        print("\n✓ 所有 PGS 变量均无缺失值")
    
    # 检查数据标准化（随机选择5个PGS检查）
    sample_pgs = pgs_cols[:5]
    stats = df[sample_pgs].agg(['mean', 'std', 'min', 'max'])
    print(f"\n前5个 PGS 的统计摘要（检查标准化）:")
    print(stats.round(4))
    print("\n注意: 如果均值接近0，标准差接近1，说明数据已标准化")


def categorize_pgs(pgs_cols):
    """将 PGS 变量按类别分类"""
    print("\n" + "=" * 60)
    print("PGS 变量分类")
    print("=" * 60)
    
    categories = {
        '代谢性状': [col for col in pgs_cols if any(x in col for x in ['BMI', 'HEIGHT', 'WC', 'WHR'])],
        '糖尿病': [col for col in pgs_cols if 'T2D' in col],
        '认知与教育': [col for col in pgs_cols if any(x in col for x in ['COG', 'EDU'])],
        '精神健康': [col for col in pgs_cols if any(x in col for x in ['MDD', 'SCZ', 'BIP', 'ADHD', 'AUTISM', 'OCD', 'PTSD', 'ANX'])],
        '心血管': [col for col in pgs_cols if any(x in col for x in ['CAD', 'MI', 'BP', 'HTN', 'PP', 'SBP', 'DBP'])],
        '血脂': [col for col in pgs_cols if any(x in col for x in ['HDL', 'LDL', 'TC', 'TG', 'CHOLESTEROL'])],
        '肾脏': [col for col in pgs_cols if any(x in col for x in ['CKD', 'EGFR', 'BUN', 'KIDNEY'])],
        '阿尔茨海默病': [col for col in pgs_cols if any(x in col for x in ['AD', 'ALZ', 'ALZHEIMER'])],
        '生殖健康': [col for col in pgs_cols if any(x in col for x in ['MENARCHE', 'MENOPAUSE', 'AFB', 'NEB'])],
        '行为': [col for col in pgs_cols if any(x in col for x in ['SMOK', 'CPD', 'ALC', 'CANNABIS', 'DPW'])],
    }
    
    for category, vars_list in categories.items():
        if len(vars_list) > 0:
            print(f"\n{category}: {len(vars_list)} 个变量")
            print(f"  示例: {vars_list[0]}")
            if len(vars_list) > 1:
                print(f"         {vars_list[1]}")
    
    return categories


def get_principal_components(df):
    """获取主成分变量"""
    pc_cols_1_5 = [col for col in df.columns if 'PC1_5' in col]
    pc_cols_6_10 = [col for col in df.columns if 'PC6_10' in col]
    
    print("\n" + "=" * 60)
    print("主成分变量")
    print("=" * 60)
    print(f"\nPC1-5 (前5个主成分): {len(pc_cols_1_5)} 个")
    print(f"  {pc_cols_1_5}")
    print(f"\nPC6-10 (后5个主成分): {len(pc_cols_6_10)} 个")
    print(f"  {pc_cols_6_10}")
    print("\n⚠️  注意: 必须同时使用所有5个PC（PC1-5）或所有10个PC（PC1-10）")
    print("   不能单独使用某个主成分！")
    
    return pc_cols_1_5, pc_cols_6_10


def example_analysis(df, pgs_cols, pc_cols_1_5):
    """示例分析"""
    print("\n" + "=" * 60)
    print("示例分析")
    print("=" * 60)
    
    # 选择几个代表性的 PGS
    selected_pgs = []
    for keyword in ['BMI', 'T2D', 'COG', 'MDD']:
        matching = [col for col in pgs_cols if keyword in col]
        if matching:
            selected_pgs.append(matching[0])
    
    if len(selected_pgs) == 0:
        selected_pgs = pgs_cols[:5]  # 如果没找到，就用前5个
    
    print(f"\n选择的 PGS 变量: {selected_pgs[:5]}")
    
    # 描述性统计
    print("\n描述性统计:")
    desc = df[selected_pgs[:5]].describe()
    print(desc.round(4))
    
    # 相关性分析
    print("\n相关性矩阵:")
    corr = df[selected_pgs[:5]].corr()
    print(corr.round(4))
    
    # 示例：如何使用主成分
    print("\n" + "-" * 60)
    print("主成分使用示例:")
    print("-" * 60)
    print("在回归分析中，应该这样使用主成分:")
    print("""
    # 示例代码:
    from sklearn.linear_model import LinearRegression
    
    # 选择自变量（PGS + 主成分）
    X = df[['A5_BMI_GIANT15'] + pc_cols_1_5]
    y = df['A5_HEIGHT_GIANT14']  # 或其他结局变量
    
    # 删除缺失值
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    # 拟合模型
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    print(f"R² = {model.score(X_clean, y_clean):.4f}")
    """)


def save_data(df, output_format='csv'):
    """保存处理后的数据"""
    print("\n" + "=" * 60)
    print("保存数据")
    print("=" * 60)
    
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    ancestry_name = df['ancestry'].iloc[0] if 'ancestry' in df.columns else 'unknown'
    base_filename = f"PGENSCORE_{ancestry_name}_processed"
    
    if output_format == 'csv':
        filepath = output_dir / f"{base_filename}.csv"
        df.to_csv(filepath, index=False)
        print(f"✓ 已保存为 CSV: {filepath}")
    
    elif output_format == 'parquet':
        try:
            filepath = output_dir / f"{base_filename}.parquet"
            df.to_parquet(filepath, index=False)
            print(f"✓ 已保存为 Parquet: {filepath}")
        except ImportError:
            print("✗ 需要安装 pyarrow: pip install pyarrow")
    
    elif output_format == 'excel':
        try:
            filepath = output_dir / f"{base_filename}.xlsx"
            df.to_excel(filepath, index=False)
            print(f"✓ 已保存为 Excel: {filepath}")
        except ImportError:
            print("✗ 需要安装 openpyxl: pip install openpyxl")
    
    else:
        print(f"✗ 不支持的格式: {output_format}")


# ==================== 主程序 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("HRS 多基因评分数据集读取工具")
    print("=" * 60)
    
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"\n✗ 错误: 数据目录不存在: {DATA_DIR}")
        print(f"请确保数据文件在正确的位置")
        sys.exit(1)
    
    try:
        # 1. 加载数据
        if ANCESTRY == 'all':
            data_dict = load_hrs_data('all')
            # 合并所有数据
            df = pd.concat([data_dict[k] for k in data_dict.keys()], ignore_index=True)
            print(f"\n✓ 合并后数据: {df.shape[0]} 行, {df.shape[1]} 列")
        else:
            df = load_hrs_data(ANCESTRY)
        
        # 2. 探索数据
        id_cols, pc_cols, pgs_cols = explore_data(df)
        
        # 3. 数据质量检查
        check_data_quality(df, pgs_cols)
        
        # 4. PGS 分类
        categories = categorize_pgs(pgs_cols)
        
        # 5. 主成分信息
        pc_cols_1_5, pc_cols_6_10 = get_principal_components(df)
        
        # 6. 示例分析
        example_analysis(df, pgs_cols, pc_cols_1_5)
        
        # 7. 保存数据（可选）
        save_choice = input("\n是否保存处理后的数据? (y/n): ").lower()
        if save_choice == 'y':
            format_choice = input("选择格式 (csv/parquet/excel, 默认csv): ").lower() or 'csv'
            save_data(df, format_choice)
        
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)
        print("\n提示:")
        print("- 数据已加载到变量 'df' 中")
        print("- 可以使用 df.head(), df.info() 等查看数据")
        print("- 详细使用说明请参考 'Python使用指南.md'")
        
        return df
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    df = main()
    
    # 在交互式环境中，数据会保留在 df 变量中
    # 可以继续使用 df 进行分析

