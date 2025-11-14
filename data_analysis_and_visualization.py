"""
多基因评分(PGS)数据分析和可视化脚本

本脚本展示仅基于PGS数据本身可以进行的分析和生成的图表类型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
FIGS_DIR = BASE_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# 祖先群体映射
ANCESTRY_FILES = {
    'African': 'PGENSCORE_African.csv',
    'European': 'PGENSCORE_European.csv',
    'Hispanic': 'PGENSCORE_Hispanic.csv'
}


def load_data(ancestry='African'):
    """加载数据"""
    file_path = OUTPUT_DIR / ANCESTRY_FILES[ancestry]
    df = pd.read_csv(file_path)
    return df


def get_pgs_columns(df):
    """获取所有PGS列名"""
    return [col for col in df.columns if col.startswith(('A5_', 'E5_', 'H5_'))]


def categorize_pgs(columns):
    """将PGS变量按类别分组"""
    categories = {
        '认知能力': ['GENCOG', 'EDU'],
        '身体特征': ['BMI', 'HEIGHT', 'WC', 'WHR'],
        '精神健康': ['SCZ', 'MDD', 'BIP', 'ADHD', 'AUTISM', 'OCD', 'PTSD', 'ANX', 'NEUROTICISM', 'DEPSYMP', 'WELLBEING', 'EXTRAVERSION', 'XDISORDER'],
        '心血管疾病': ['CAD', 'MI', 'HTN', 'SBP', 'DBP', 'PP'],
        '代谢疾病': ['T2D', 'HBA1C', 'HDL', 'LDL', 'TC', 'TG'],
        '神经退行性疾病': ['AD', 'ALZ', 'PROXYALZ'],
        '肾脏疾病': ['CKD', 'BUN', 'EGFR'],
        '行为特征': ['EVRSMK', 'CPD', 'SC', 'SI', 'AI', 'DPW', 'ALC', 'CANNABIS', 'AB'],
        '生殖健康': ['MENARCHE', 'MENOPAUSE', 'AFB', 'NEB'],
        '其他': ['LONGEVITY', 'CORTISOL', 'CRP']
    }
    
    categorized = {cat: [] for cat in categories.keys()}
    categorized['其他'] = []
    
    for col in columns:
        assigned = False
        for category, keywords in categories.items():
            if any(keyword in col for keyword in keywords):
                categorized[category].append(col)
                assigned = True
                break
        if not assigned:
            categorized['其他'].append(col)
    
    return categorized


def analysis_1_distribution_analysis(df, ancestry='African'):
    """
    分析1: 分布分析
    结论: 了解各PGS变量的分布特征（正态性、偏度、异常值）
    图表: 直方图、密度图、箱线图
    """
    print("\n" + "="*70)
    print("分析1: PGS变量分布分析")
    print("="*70)
    
    pgs_cols = get_pgs_columns(df)
    
    # 选择几个代表性的变量
    key_vars = {
        '认知能力': [col for col in pgs_cols if 'GENCOG' in col or 'EDU' in col][:2],
        '身体特征': [col for col in pgs_cols if 'BMI' in col or 'HEIGHT' in col][:2],
        '精神健康': [col for col in pgs_cols if 'MDD' in col or 'SCZ' in col][:2],
        '代谢疾病': [col for col in pgs_cols if 'T2D' in col or 'HDL' in col][:2]
    }
    
    # 创建分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for category, vars_list in key_vars.items():
        if vars_list and plot_idx < 4:
            for var in vars_list[:1]:  # 每个类别只画一个
                data = df[var].dropna()
                axes[plot_idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
                axes[plot_idx].set_title(f'{category}: {var}', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('PGS值')
                axes[plot_idx].set_ylabel('频数')
                axes[plot_idx].axvline(data.mean(), color='red', linestyle='--', 
                                      label=f'均值: {data.mean():.2f}')
                axes[plot_idx].legend()
                plot_idx += 1
                break
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存分布分析图: {ancestry}_distribution_analysis.png")
    plt.close()
    
    # 统计信息
    stats = {}
    for var in pgs_cols[:10]:  # 前10个变量
        data = df[var].dropna()
        if len(data) > 0:
            stats[var] = {
                '均值': data.mean(),
                '标准差': data.std(),
                '中位数': data.median(),
                '偏度': data.skew(),
                '缺失值': df[var].isna().sum()
            }
    
    print("\n主要PGS变量统计摘要:")
    for var, stat in list(stats.items())[:5]:
        print(f"\n{var}:")
        print(f"  均值: {stat['均值']:.4f}, 标准差: {stat['标准差']:.4f}")
        print(f"  偏度: {stat['偏度']:.4f} (偏度>1表示右偏，<-1表示左偏)")
    
    return stats


def analysis_2_correlation_analysis(df, ancestry='African'):
    """
    分析2: 相关性分析
    结论: 发现不同PGS变量之间的关联模式（哪些性状在遗传上相关）
    图表: 相关性热力图、散点图矩阵
    """
    print("\n" + "="*70)
    print("分析2: PGS变量相关性分析")
    print("="*70)
    
    pgs_cols = get_pgs_columns(df)
    
    # 选择代表性变量进行相关性分析
    selected_vars = []
    categories = ['GENCOG', 'BMI', 'MDD', 'T2D', 'HDL', 'SBP', 'SCZ', 'EDU']
    for cat in categories:
        matches = [col for col in pgs_cols if cat in col]
        if matches:
            selected_vars.append(matches[0])
    
    if len(selected_vars) < 3:
        selected_vars = pgs_cols[:15]  # 如果不够，取前15个
    
    # 计算相关性矩阵
    corr_data = df[selected_vars[:15]].corr()
    
    # 绘制热力图
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))  # 只显示下三角
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=[col.replace('A5_', '').replace('E5_', '').replace('H5_', '') 
                            for col in selected_vars[:15]],
                yticklabels=[col.replace('A5_', '').replace('E5_', '').replace('H5_', '') 
                            for col in selected_vars[:15]])
    plt.title(f'{ancestry} PGS变量相关性热力图', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存相关性热力图: {ancestry}_correlation_heatmap.png")
    plt.close()
    
    # 找出强相关对
    strong_corr_pairs = []
    for i in range(len(corr_data.columns)):
        for j in range(i+1, len(corr_data.columns)):
            corr_val = corr_data.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr_pairs.append((
                    corr_data.columns[i], 
                    corr_data.columns[j], 
                    corr_val
                ))
    
    if strong_corr_pairs:
        print(f"\n发现 {len(strong_corr_pairs)} 对强相关变量 (|r| > 0.5):")
        for var1, var2, corr in strong_corr_pairs[:10]:
            print(f"  {var1} <-> {var2}: r = {corr:.3f}")
    
    return corr_data, strong_corr_pairs


def analysis_3_category_comparison(df, ancestry='African'):
    """
    分析3: 类别间比较
    结论: 不同类别PGS变量的整体特征差异
    图表: 分组箱线图、小提琴图
    """
    print("\n" + "="*70)
    print("分析3: PGS变量类别比较")
    print("="*70)
    
    pgs_cols = get_pgs_columns(df)
    categorized = categorize_pgs(pgs_cols)
    
    # 计算每个类别的平均绝对PGS值
    category_stats = {}
    for category, cols in categorized.items():
        if cols:
            # 计算每个样本在该类别下的平均绝对PGS值
            category_data = df[cols].abs().mean(axis=1)
            category_stats[category] = category_data
    
    # 创建箱线图
    fig, ax = plt.subplots(figsize=(14, 8))
    data_to_plot = [category_stats[cat] for cat in category_stats.keys() if len(category_stats[cat]) > 0]
    labels = [cat for cat in category_stats.keys() if len(category_stats[cat]) > 0]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_title(f'{ancestry} 不同类别PGS变量的分布比较', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均绝对PGS值')
    ax.set_xlabel('PGS类别')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_category_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存类别比较图: {ancestry}_category_comparison.png")
    plt.close()
    
    # 统计信息
    print("\n各类别PGS变量统计:")
    for category, data in category_stats.items():
        if len(data) > 0:
            print(f"\n{category}:")
            print(f"  变量数量: {len(categorized[category])}")
            print(f"  平均PGS值: {data.mean():.4f}")
            print(f"  标准差: {data.std():.4f}")
    
    return category_stats


def analysis_4_extreme_values(df, ancestry='African'):
    """
    分析4: 极值分析
    结论: 识别具有极端PGS值的个体（高风险/高保护性）
    图表: 极值散点图、异常值检测图
    """
    print("\n" + "="*70)
    print("分析4: 极值个体识别")
    print("="*70)
    
    pgs_cols = get_pgs_columns(df)
    
    # 选择几个关键疾病风险PGS
    key_risk_vars = {
        'T2D': [col for col in pgs_cols if 'T2D' in col][:1],
        'CAD': [col for col in pgs_cols if 'CAD' in col][:1],
        'MDD': [col for col in pgs_cols if 'MDD' in col][:1],
        'SCZ': [col for col in pgs_cols if 'SCZ' in col][:1]
    }
    
    # 创建极值分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for disease, vars_list in key_risk_vars.items():
        if vars_list and plot_idx < 4:
            var = vars_list[0]
            data = df[var].dropna()
            
            # 计算Z分数
            z_scores = np.abs((data - data.mean()) / data.std())
            extreme_idx = z_scores > 2  # Z分数>2为极端值
            
            axes[plot_idx].scatter(data[~extreme_idx], z_scores[~extreme_idx], 
                                  alpha=0.5, label='正常值', s=20)
            axes[plot_idx].scatter(data[extreme_idx], z_scores[extreme_idx], 
                                  color='red', label='极端值 (|Z|>2)', s=30)
            axes[plot_idx].axhline(y=2, color='red', linestyle='--', label='阈值 (Z=2)')
            axes[plot_idx].set_title(f'{disease} PGS极值分析: {var}', fontsize=11, fontweight='bold')
            axes[plot_idx].set_xlabel('PGS值')
            axes[plot_idx].set_ylabel('Z分数 (绝对值)')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            
            print(f"\n{disease} ({var}):")
            print(f"  极端值数量 (|Z|>2): {extreme_idx.sum()}")
            print(f"  极端值比例: {extreme_idx.sum()/len(data)*100:.2f}%")
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_extreme_values.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存极值分析图: {ancestry}_extreme_values.png")
    plt.close()


def analysis_5_pca_visualization(df, ancestry='African'):
    """
    分析5: 主成分分析可视化
    结论: 了解样本在多维PGS空间中的分布和聚类模式
    图表: PCA散点图、3D PCA图
    """
    print("\n" + "="*70)
    print("分析5: PGS主成分分析")
    print("="*70)
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    pgs_cols = get_pgs_columns(df)
    
    # 选择部分变量进行PCA（太多变量会计算很慢）
    selected_cols = pgs_cols[:30]  # 选择前30个变量
    
    # 准备数据
    X = df[selected_cols].dropna()
    
    if len(X) < 10:
        print("数据不足，跳过PCA分析")
        return
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 绘制PCA图
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
    plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title(f'{ancestry} PGS数据主成分分析 (前30个变量)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_pca_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存PCA分析图: {ancestry}_pca_analysis.png")
    plt.close()
    
    print(f"\nPCA结果:")
    print(f"  PC1解释方差: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2解释方差: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")


def analysis_6_missing_data_pattern(df, ancestry='African'):
    """
    分析6: 缺失值模式分析
    结论: 了解数据完整性，哪些PGS变量缺失较多
    图表: 缺失值热力图、缺失值条形图
    """
    print("\n" + "="*70)
    print("分析6: 缺失值模式分析")
    print("="*70)
    
    pgs_cols = get_pgs_columns(df)
    
    # 计算缺失率
    missing_rates = {}
    for col in pgs_cols:
        missing_rate = df[col].isna().sum() / len(df) * 100
        missing_rates[col] = missing_rate
    
    # 排序
    sorted_missing = sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)
    
    # 绘制缺失率条形图（前20个）
    top_missing = sorted_missing[:20]
    fig, ax = plt.subplots(figsize=(14, 8))
    vars_names = [var.replace('A5_', '').replace('E5_', '').replace('H5_', '') 
                  for var, _ in top_missing]
    missing_values = [rate for _, rate in top_missing]
    
    bars = ax.barh(vars_names, missing_values, color='coral')
    ax.set_xlabel('缺失率 (%)', fontsize=12)
    ax.set_title(f'{ancestry} PGS变量缺失率 (前20个)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, missing_values)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'{ancestry}_missing_data.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存缺失值分析图: {ancestry}_missing_data.png")
    plt.close()
    
    print(f"\n缺失值统计:")
    print(f"  完全缺失的变量数: {sum(1 for _, rate in missing_rates.items() if rate == 100)}")
    print(f"  无缺失的变量数: {sum(1 for _, rate in missing_rates.items() if rate == 0)}")
    print(f"  平均缺失率: {np.mean(list(missing_rates.values())):.2f}%")


def analysis_7_multi_ancestry_comparison():
    """
    分析7: 多群体比较分析
    结论: 比较不同祖先群体的PGS分布差异
    图表: 分组箱线图、密度对比图、群体差异热力图
    """
    print("\n" + "="*70)
    print("分析7: 多群体PGS比较分析")
    print("="*70)
    
    # 加载所有三个群体的数据
    all_data = {}
    for ancestry in ['African', 'European', 'Hispanic']:
        try:
            df = load_data(ancestry)
            all_data[ancestry] = df
            print(f"✓ 已加载 {ancestry} 数据: {df.shape[0]} 个样本")
        except Exception as e:
            print(f"✗ 加载 {ancestry} 数据失败: {e}")
    
    if len(all_data) < 2:
        print("至少需要2个群体的数据才能进行比较")
        return
    
    # 找到共同的PGS变量
    all_pgs_cols = {}
    for ancestry, df in all_data.items():
        pgs_cols = get_pgs_columns(df)
        all_pgs_cols[ancestry] = set(pgs_cols)
    
    # 找到所有群体都有的变量（需要标准化变量名）
    # 因为不同群体的变量名前缀不同（A5_, E5_, H5_）
    # 我们需要提取变量名的主体部分
    def get_base_name(col):
        """提取变量名的主体部分（去掉祖先前缀）"""
        for prefix in ['A5_', 'E5_', 'H5_']:
            if col.startswith(prefix):
                return col[len(prefix):]
        return col
    
    # 标准化所有变量名
    standardized_vars = {}
    for ancestry, cols in all_pgs_cols.items():
        standardized_vars[ancestry] = {get_base_name(col): col for col in cols}
    
    # 找到共同变量
    common_vars = set(standardized_vars['African'].keys())
    for ancestry in standardized_vars.keys():
        common_vars = common_vars.intersection(set(standardized_vars[ancestry].keys()))
    
    print(f"\n找到 {len(common_vars)} 个共同PGS变量")
    
    if len(common_vars) == 0:
        print("没有找到共同的PGS变量，无法进行比较")
        return
    
    # 选择代表性的变量进行比较
    key_vars = ['GENCOG_CHARGE15', 'BMI_GIANT15', 'HEIGHT_GIANT14', 'MDD_PGC13', 
                'T2D_DIAGRAM12', 'HDL_GLGC13', 'SBP_COGNET17', 'SCZ_PGC14']
    
    selected_vars = [v for v in key_vars if v in common_vars][:8]
    
    if not selected_vars:
        selected_vars = list(common_vars)[:8]
    
    # 1. 分组箱线图比较
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    comparison_stats = {}
    
    for idx, var_base in enumerate(selected_vars[:8]):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        
        for ancestry in all_data.keys():
            if var_base in standardized_vars[ancestry]:
                actual_col = standardized_vars[ancestry][var_base]
                data = all_data[ancestry][actual_col].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    labels.append(ancestry)
                    
                    # 保存统计信息
                    if var_base not in comparison_stats:
                        comparison_stats[var_base] = {}
                    comparison_stats[var_base][ancestry] = {
                        'mean': data.mean(),
                        'std': data.std(),
                        'median': data.median(),
                        'n': len(data)
                    }
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            var_display = var_base.replace('_', ' ')[:30]
            ax.set_title(f'{var_display}', fontsize=10, fontweight='bold')
            ax.set_ylabel('PGS值')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('不同祖先群体PGS分布比较', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'multi_ancestry_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存分组箱线图: multi_ancestry_boxplot_comparison.png")
    plt.close()
    
    # 2. 密度对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, var_base in enumerate(selected_vars[:8]):
        ax = axes[idx]
        
        for ancestry in all_data.keys():
            if var_base in standardized_vars[ancestry]:
                actual_col = standardized_vars[ancestry][var_base]
                data = all_data[ancestry][actual_col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=50, alpha=0.5, label=ancestry, density=True)
        
        var_display = var_base.replace('_', ' ')[:30]
        ax.set_title(f'{var_display}', fontsize=10, fontweight='bold')
        ax.set_xlabel('PGS值')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('不同祖先群体PGS密度分布对比', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'multi_ancestry_density_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存密度对比图: multi_ancestry_density_comparison.png")
    plt.close()
    
    # 3. 群体差异热力图（均值差异）
    if len(selected_vars) > 0 and len(all_data) >= 2:
        # 计算每个变量在不同群体间的均值差异
        mean_matrix = []
        var_names = []
        
        for var_base in selected_vars[:15]:
            means = []
            for ancestry in ['African', 'European', 'Hispanic']:
                if ancestry in all_data and var_base in standardized_vars.get(ancestry, {}):
                    actual_col = standardized_vars[ancestry][var_base]
                    data = all_data[ancestry][actual_col].dropna()
                    if len(data) > 0:
                        means.append(data.mean())
                    else:
                        means.append(np.nan)
                else:
                    means.append(np.nan)
            
            if not all(np.isnan(means)):
                mean_matrix.append(means)
                var_names.append(var_base.replace('_', ' ')[:25])
        
        if mean_matrix:
            mean_df = pd.DataFrame(mean_matrix, 
                                  index=var_names,
                                  columns=['African', 'European', 'Hispanic'])
            
            plt.figure(figsize=(10, max(8, len(var_names)*0.3)))
            sns.heatmap(mean_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       center=0, square=False, linewidths=0.5,
                       cbar_kws={"label": "平均PGS值"})
            plt.title('不同祖先群体PGS均值比较', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('祖先群体', fontsize=12)
            plt.ylabel('PGS变量', fontsize=12)
            plt.tight_layout()
            plt.savefig(FIGS_DIR / 'multi_ancestry_mean_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✓ 已保存均值热力图: multi_ancestry_mean_heatmap.png")
            plt.close()
    
    # 4. 统计摘要
    print("\n" + "-"*70)
    print("多群体比较统计摘要")
    print("-"*70)
    
    for var_base in list(comparison_stats.keys())[:5]:
        print(f"\n{var_base}:")
        for ancestry in comparison_stats[var_base].keys():
            stats = comparison_stats[var_base][ancestry]
            print(f"  {ancestry}:")
            print(f"    均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
            print(f"    中位数: {stats['median']:.4f}, 样本数: {stats['n']}")
        
        # 计算群体间差异
        if len(comparison_stats[var_base]) >= 2:
            means = [stats['mean'] for stats in comparison_stats[var_base].values()]
            if len(means) >= 2:
                max_diff = max(means) - min(means)
                print(f"    群体间最大差异: {max_diff:.4f}")
    
    return comparison_stats


def generate_summary_report(ancestry='African'):
    """生成分析总结报告"""
    report = f"""
# {ancestry} PGS数据分析报告

## 可进行的分析类型

### 1. 描述性统计分析
- **分布特征**: 各PGS变量的均值、标准差、偏度、峰度
- **结论**: 了解数据的集中趋势和离散程度，判断数据是否符合正态分布
- **图表类型**: 
  - 直方图 (Histogram)
  - 密度图 (Density Plot)
  - 箱线图 (Box Plot)
  - Q-Q图 (正态性检验)

### 2. 相关性分析
- **变量关联**: 不同PGS变量之间的相关关系
- **结论**: 
  - 发现遗传上相关的性状（如BMI与腰围、抑郁与焦虑）
  - 识别共病风险的遗传基础
  - 发现意外的关联模式
- **图表类型**:
  - 相关性热力图 (Correlation Heatmap)
  - 散点图矩阵 (Scatter Plot Matrix)
  - 网络图 (Network Graph，显示强相关关系)

### 3. 类别比较分析
- **类别特征**: 不同类别PGS变量的整体特征
- **结论**: 
  - 认知能力、身体特征、精神健康等不同类别的遗传风险分布
  - 哪些类别的遗传变异更大
- **图表类型**:
  - 分组箱线图 (Grouped Box Plot)
  - 小提琴图 (Violin Plot)
  - 雷达图 (Radar Chart，多类别比较)

### 4. 极值分析
- **高风险个体**: 识别具有极端PGS值的个体
- **结论**: 
  - 哪些个体在特定疾病上具有高遗传风险
  - 哪些个体具有保护性遗传因素
- **图表类型**:
  - 极值散点图 (Extreme Values Scatter)
  - Z分数分布图
  - 异常值检测图 (Outlier Detection)

### 5. 降维可视化
- **多维空间**: 将高维PGS数据降维到2D/3D可视化
- **结论**: 
  - 样本在多维遗传空间中的分布模式
  - 是否存在明显的聚类结构
- **图表类型**:
  - PCA散点图 (Principal Component Analysis)
  - t-SNE图 (t-Distributed Stochastic Neighbor Embedding)
  - UMAP图 (Uniform Manifold Approximation and Projection)
  - 3D PCA图

### 6. 缺失值分析
- **数据完整性**: 分析哪些PGS变量缺失较多
- **结论**: 
  - 数据质量评估
  - 哪些变量需要特别注意
- **图表类型**:
  - 缺失值热力图 (Missing Data Heatmap)
  - 缺失值条形图 (Missing Data Bar Chart)

### 7. 时间趋势分析（如果有多个版本）
- **版本比较**: 比较不同版本的PGS值
- **结论**: PGS计算方法改进的影响
- **图表类型**: 版本对比图

### 8. 多群体比较（如果有多个祖先群体数据）
- **群体差异**: 比较不同祖先群体的PGS分布
- **结论**: 不同群体的遗传风险差异
- **图表类型**: 
  - 分组比较图
  - 密度对比图

## 主要发现方向

1. **遗传相关性网络**: 哪些疾病/性状在遗传上相关
2. **多病共患模式**: 识别可能同时具有多种疾病遗传风险的个体
3. **保护性因素**: 识别具有保护性遗传因素的个体
4. **数据质量**: 评估数据完整性和可靠性
5. **群体特征**: 不同祖先群体的遗传风险分布特征

## 注意事项

- PGS值本身不代表实际疾病状态，只是遗传风险评分
- 需要结合环境因素和表型数据才能得出实际结论
- 相关性不等于因果关系
- 缺失值可能影响分析结果

---
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = FIGS_DIR / f'{ancestry}_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 已保存分析报告: {report_path}")


def main():
    """主函数"""
    print("="*70)
    print("PGS数据分析和可视化")
    print("="*70)
    
    # 分析African数据作为示例
    ancestry = 'African'
    
    print(f"\n正在加载 {ancestry} 数据...")
    df = load_data(ancestry)
    print(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    
    pgs_cols = get_pgs_columns(df)
    print(f"PGS变量数量: {len(pgs_cols)}")
    
    # 执行各项分析
    try:
        analysis_1_distribution_analysis(df, ancestry)
        analysis_2_correlation_analysis(df, ancestry)
        analysis_3_category_comparison(df, ancestry)
        analysis_4_extreme_values(df, ancestry)
        analysis_5_pca_visualization(df, ancestry)
        analysis_6_missing_data_pattern(df, ancestry)
        generate_summary_report(ancestry)
        
        # 多群体比较分析
        print("\n" + "="*70)
        analysis_7_multi_ancestry_comparison()
        
        print("\n" + "="*70)
        print("所有分析完成！")
        print("="*70)
        print(f"\n图表已保存到: {FIGS_DIR}")
        print(f"报告已保存到: {FIGS_DIR / f'{ancestry}_analysis_report.md'}")
        
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

