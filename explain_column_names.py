"""
列名解释工具 - 解析PGS变量命名规则并生成解释文档

功能：
1. 解析数据字典文件，提取变量说明
2. 解释PGS变量的命名规则
3. 生成列名解释文档（CSV和Markdown格式）
"""

import pandas as pd
from pathlib import Path
import re

# ==================== 配置 ====================
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "output"

# 数据字典文件映射
CODEBOOK_FILES = {
    'A': DOCS_DIR / 'PGENSCOREA_R.txt',
    'E': DOCS_DIR / 'PGENSCOREE_R.txt',
    'H': DOCS_DIR / 'PGENSCOREH_R.txt'
}

# 祖先群体名称
ANCESTRY_NAMES = {
    'A': 'African (非洲祖先)',
    'E': 'European (欧洲祖先)',
    'H': 'Hispanic (西班牙裔祖先)'
}

# PGS命名规则说明
PGS_NAMING_RULES = {
    '前缀': {
        'A5_': '非洲祖先 (African Ancestry)',
        'E5_': '欧洲祖先 (European Ancestry)',
        'H5_': '西班牙裔祖先 (Hispanic Ancestry)'
    },
    '常见缩写': {
        'GENCOG': 'General Cognition (一般认知能力)',
        'BMI': 'Body Mass Index (体重指数)',
        'HEIGHT': 'Height (身高)',
        'SCZ': 'Schizophrenia (精神分裂症)',
        'EDU': 'Educational Attainment (教育程度)',
        'EVRSMK': 'Ever Smoker (是否吸烟)',
        'WC': 'Waist Circumference (腰围)',
        'WHR': 'Waist-Hip Ratio (腰臀比)',
        'NEUROTICISM': 'Neuroticism (神经质)',
        'WELLBEING': 'Wellbeing (主观幸福感)',
        'DEPSYMP': 'Depressive Symptoms (抑郁症状)',
        'CAD': 'Coronary Artery Disease (冠心病)',
        'MI': 'Myocardial Infarction (心肌梗死)',
        'CORTISOL': 'Cortisol (皮质醇)',
        'T2D': 'Type 2 Diabetes (2型糖尿病)',
        'BIP': 'Bipolar Disorder (双相情感障碍)',
        'ADHD': 'Attention Deficit Hyperactivity Disorder (注意力缺陷多动障碍)',
        'XDISORDER': 'Cross Disorder (交叉障碍)',
        'MENARCHE': 'Menarche (初潮年龄)',
        'MENOPAUSE': 'Menopause (绝经年龄)',
        'MDD': 'Major Depressive Disorder (重度抑郁症)',
        'CPD': 'Cigarettes Per Day (每日吸烟量)',
        'EXTRAVERSION': 'Extraversion (外向性)',
        'AUTISM': 'Autism (自闭症)',
        'LONGEVITY': 'Longevity (长寿)',
        'AB': 'Antisocial Behavior (反社会行为)',
        'OCD': 'Obsessive Compulsive Disorder (强迫症)',
        'AFB': 'Age at First Birth (首次生育年龄)',
        'NEB': 'Number of Ever Born (生育子女数)',
        'PTSD': 'Post-Traumatic Stress Disorder (创伤后应激障碍)',
        'HDL': 'High-Density Lipoprotein (高密度脂蛋白)',
        'LDL': 'Low-Density Lipoprotein (低密度脂蛋白)',
        'TC': 'Total Cholesterol (总胆固醇)',
        'ANX': 'Anxiety (焦虑症)',
        'AD': 'Alzheimer\'s Disease (阿尔茨海默病)',
        'ALZ': 'Alzheimer\'s Disease (阿尔茨海默病)',
        'BUN': 'Blood Urea Nitrogen (血尿素氮)',
        'CKD': 'Chronic Kidney Disease (慢性肾脏病)',
        'DBP': 'Diastolic Blood Pressure (舒张压)',
        'EGFR': 'Estimated Glomerular Filtration Rate (肾小球滤过率)',
        'AI': 'Age of Initiation (起始年龄)',
        'DPW': 'Drinks Per Week (每周饮酒量)',
        'SC': 'Smoking Cessation (戒烟)',
        'SI': 'Smoking Initiation (吸烟起始)',
        'HBA1C': 'Hemoglobin A1c (糖化血红蛋白)',
        'HTN': 'Hypertension (高血压)',
        'CANNABIS': 'Cannabis Use (大麻使用)',
        'ALC': 'Alcohol (酒精)',
        'PP': 'Pulse Pressure (脉搏压)',
        'SBP': 'Systolic Blood Pressure (收缩压)',
        'CRP': 'C-Reactive Protein (C反应蛋白)',
        'TG': 'Triglycerides (甘油三酯)'
    },
    '研究联盟缩写': {
        'CHARGE': 'Cohorts for Heart and Aging Research in Genomic Epidemiology',
        'GIANT': 'Genetic Investigation of ANthropometric Traits',
        'PGC': 'Psychiatric Genomics Consortium',
        'SSGAC': 'Social Science Genetic Association Consortium',
        'DIAGRAM': 'DIAbetes Genetics Replication and Meta-analysis',
        'CARDIOGRAM': 'Coronary ARtery DIsease Genome-wide Replication and Meta-analysis',
        'GLGC': 'Global Lipid Genetics Consortium',
        'CKDGEN': 'Chronic Kidney Disease Genetics Consortium',
        'COGENT': 'Continental Origins and Genetic Epidemiology Network',
        'MAGIC': 'Meta-Analyses of Glucose and Insulin-related traits Consortium',
        'IGAP': 'International Genomics of Alzheimer\'s Project',
        'ReproGen': 'Reproductive Genetics Consortium',
        'GSCAN': 'GWAS & Sequencing Consortium of Alcohol and Nicotine use',
        'ANGST': 'Anxiety NeuroGenetics Study',
        'IOCDF': 'International Obsessive Compulsive Disorder Foundation',
        'BROAD': 'Broad Antisocial Behavior Consortium',
        'SOCGEN': 'Sociogenome Consortium',
        'EADB': 'European Alzheimer\'s Disease Biobank',
        'TAG': 'Tobacco and Genetics',
        'GPC': 'Genetics of Personality Consortium',
        'CORNET': 'Cortisol Network',
        'ICCUKB': 'International Cannabis Consortium UK Biobank'
    }
}


def parse_codebook(codebook_file):
    """
    解析数据字典文件，提取变量说明
    
    返回: dict {变量名: 说明}
    """
    variable_descriptions = {}
    
    if not codebook_file.exists():
        print(f"警告: 数据字典文件不存在: {codebook_file}")
        return variable_descriptions
    
    with open(codebook_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 使用正则表达式提取变量定义
    # 格式: 变量名 (可能有多行) 然后是描述
    pattern = r'^([A-Z0-9_]+)\s+(.+?)(?=^[A-Z0-9_]+\s+|^={80,}|$)'
    
    # 更简单的模式：查找变量名和紧随其后的描述
    lines = content.split('\n')
    current_var = None
    current_desc = []
    
    for i, line in enumerate(lines):
        # 查找变量定义行（变量名在单独一行，后面跟着描述）
        var_match = re.match(r'^([A-Z0-9_]+)\s+(.+)$', line.strip())
        if var_match:
            # 保存之前的变量
            if current_var and current_desc:
                variable_descriptions[current_var] = ' '.join(current_desc).strip()
            
            # 开始新变量
            current_var = var_match.group(1)
            current_desc = [var_match.group(2)]
        elif current_var and line.strip() and not line.strip().startswith('='):
            # 继续收集描述
            if 'Section:' not in line and 'Level:' not in line and 'Type:' not in line:
                if not re.match(r'^-+$', line.strip()):
                    current_desc.append(line.strip())
    
    # 保存最后一个变量
    if current_var and current_desc:
        variable_descriptions[current_var] = ' '.join(current_desc).strip()
    
    return variable_descriptions


def explain_pgs_name(var_name):
    """
    解释PGS变量名的含义
    
    返回: dict {组件: 含义}
    """
    explanation = {
        'variable_name': var_name,
        'ancestry': 'Unknown',
        'trait': 'Unknown',
        'consortium': 'Unknown',
        'year': 'Unknown',
        'full_description': ''
    }
    
    # 提取祖先前缀
    if var_name.startswith('A5_'):
        explanation['ancestry'] = 'African (非洲祖先)'
        trait_part = var_name[3:]
    elif var_name.startswith('E5_'):
        explanation['ancestry'] = 'European (欧洲祖先)'
        trait_part = var_name[3:]
    elif var_name.startswith('H5_'):
        explanation['ancestry'] = 'Hispanic (西班牙裔祖先)'
        trait_part = var_name[3:]
    else:
        trait_part = var_name
    
    # 提取性状和联盟信息
    # 格式通常是: TRAIT_CONSORTIUMYEAR 或 TRAIT_CONSORTIUM_YEAR
    parts = trait_part.split('_')
    
    if len(parts) >= 2:
        # 最后一部分通常是联盟+年份
        last_part = parts[-1]
        
        # 提取年份（通常是2位或4位数字）
        year_match = re.search(r'(\d{2,4})$', last_part)
        if year_match:
            explanation['year'] = year_match.group(1)
            consortium_part = last_part[:year_match.start()]
        else:
            consortium_part = last_part
        
        # 性状部分是除了最后一部分的所有部分
        trait_parts = parts[:-1]
        explanation['trait'] = '_'.join(trait_parts)
        explanation['consortium'] = consortium_part
    else:
        explanation['trait'] = trait_part
    
    # 查找缩写含义
    trait_meanings = []
    for abbrev, meaning in PGS_NAMING_RULES['常见缩写'].items():
        if abbrev in explanation['trait']:
            trait_meanings.append(meaning)
    
    if trait_meanings:
        explanation['trait_meaning'] = ' / '.join(trait_meanings)
    else:
        explanation['trait_meaning'] = explanation['trait']
    
    # 查找联盟含义
    if explanation['consortium'] in PGS_NAMING_RULES['研究联盟缩写']:
        explanation['consortium_meaning'] = PGS_NAMING_RULES['研究联盟缩写'][explanation['consortium']]
    else:
        explanation['consortium_meaning'] = explanation['consortium']
    
    return explanation


def generate_column_explanation(ancestry_code='A'):
    """
    为指定祖先群体生成列名解释
    """
    print(f"\n{'=' * 70}")
    print(f"生成列名解释: {ANCESTRY_NAMES[ancestry_code]}")
    print(f"{'=' * 70}")
    
    # 读取数据字典
    codebook_file = CODEBOOK_FILES[ancestry_code]
    print(f"\n读取数据字典: {codebook_file}")
    descriptions = parse_codebook(codebook_file)
    print(f"提取到 {len(descriptions)} 个变量说明")
    
    # 读取CSV文件获取所有列名
    csv_file = OUTPUT_DIR / f"PGENSCORE_{ANCESTRY_NAMES[ancestry_code].split()[0]}.csv"
    if not csv_file.exists():
        print(f"错误: CSV文件不存在: {csv_file}")
        return None
    
    print(f"读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file, nrows=1)  # 只读第一行获取列名
    columns = df.columns.tolist()
    print(f"找到 {len(columns)} 个列")
    
    # 生成解释
    explanations = []
    for col in columns:
        if col in ['HHID', 'PN', 'VERSION']:
            # 基础变量
            if col == 'HHID':
                desc = 'Household Identifier (家庭标识符)'
            elif col == 'PN':
                desc = 'Person Number (人员编号)'
            elif col == 'VERSION':
                desc = 'Data Version (数据版本号)'
            else:
                desc = col
            explanations.append({
                'column_name': col,
                'type': '基础变量',
                'ancestry': '',
                'trait': '',
                'consortium': '',
                'year': '',
                'description': desc,
                'full_description': desc
            })
        elif col.startswith('PC'):
            # 主成分
            desc = descriptions.get(col, 'Principal Component (主成分)')
            explanations.append({
                'column_name': col,
                'type': '主成分',
                'ancestry': '',
                'trait': '',
                'consortium': '',
                'year': '',
                'description': 'Principal Component for population stratification control',
                'full_description': desc
            })
        else:
            # PGS变量
            exp = explain_pgs_name(col)
            exp['type'] = '多基因评分 (PGS)'
            exp['description'] = descriptions.get(col, f"{exp['trait_meaning']} - {exp['consortium_meaning']}")
            exp['full_description'] = descriptions.get(col, '')
            explanations.append(exp)
    
    # 转换为DataFrame
    df_explanation = pd.DataFrame(explanations)
    
    # 保存为CSV
    output_csv = OUTPUT_DIR / f"Column_Explanation_{ANCESTRY_NAMES[ancestry_code].split()[0]}.csv"
    df_explanation.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存列名解释CSV: {output_csv}")
    
    # 生成Markdown文档
    output_md = OUTPUT_DIR / f"Column_Explanation_{ANCESTRY_NAMES[ancestry_code].split()[0]}.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(f"# 列名解释文档 - {ANCESTRY_NAMES[ancestry_code]}\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## PGS变量命名规则\n\n")
        f.write("PGS变量名格式: `[祖先前缀][性状]_[研究联盟][年份]`\n\n")
        f.write("### 祖先前缀\n\n")
        for prefix, meaning in PGS_NAMING_RULES['前缀'].items():
            f.write(f"- `{prefix}`: {meaning}\n")
        
        f.write("\n### 常见性状缩写\n\n")
        for abbrev, meaning in list(PGS_NAMING_RULES['常见缩写'].items())[:30]:  # 显示前30个
            f.write(f"- `{abbrev}`: {meaning}\n")
        
        f.write("\n### 研究联盟缩写\n\n")
        for abbrev, meaning in PGS_NAMING_RULES['研究联盟缩写'].items():
            f.write(f"- `{abbrev}`: {meaning}\n")
        
        f.write("\n---\n\n")
        f.write("## 完整列名列表\n\n")
        
        # 按类型分组
        for var_type in ['基础变量', '主成分', '多基因评分 (PGS)']:
            type_data = df_explanation[df_explanation['type'] == var_type]
            if len(type_data) > 0:
                f.write(f"### {var_type}\n\n")
                for _, row in type_data.iterrows():
                    f.write(f"#### {row['column_name']}\n\n")
                    if row['type'] == '多基因评分 (PGS)':
                        f.write(f"- **类型**: {row['type']}\n")
                        f.write(f"- **祖先群体**: {row['ancestry']}\n")
                        f.write(f"- **性状**: {row['trait']} ({row.get('trait_meaning', '')})\n")
                        f.write(f"- **研究联盟**: {row['consortium']} ({row.get('consortium_meaning', '')})\n")
                        if row['year']:
                            f.write(f"- **年份**: {row['year']}\n")
                    f.write(f"- **说明**: {row['description']}\n")
                    if row['full_description']:
                        f.write(f"- **详细描述**: {row['full_description']}\n")
                    f.write("\n")
    
    print(f"✓ 已保存Markdown文档: {output_md}")
    
    return df_explanation


def main():
    """主函数"""
    print("=" * 70)
    print("HRS 多基因评分数据集 - 列名解释工具")
    print("=" * 70)
    
    # 为所有三个祖先群体生成解释
    all_explanations = {}
    
    for ancestry_code in ['A', 'E', 'H']:
        try:
            exp_df = generate_column_explanation(ancestry_code)
            if exp_df is not None:
                all_explanations[ancestry_code] = exp_df
        except Exception as e:
            print(f"\n✗ 处理 {ANCESTRY_NAMES[ancestry_code]} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成总结
    print(f"\n{'=' * 70}")
    print("生成完成总结")
    print(f"{'=' * 70}")
    
    for ancestry_code, exp_df in all_explanations.items():
        print(f"\n{ANCESTRY_NAMES[ancestry_code]}:")
        print(f"  - 总列数: {len(exp_df)}")
        print(f"  - 基础变量: {len(exp_df[exp_df['type'] == '基础变量'])}")
        print(f"  - 主成分: {len(exp_df[exp_df['type'] == '主成分'])}")
        print(f"  - PGS变量: {len(exp_df[exp_df['type'] == '多基因评分 (PGS)'])}")
        print(f"  - 输出文件: Column_Explanation_{ANCESTRY_NAMES[ancestry_code].split()[0]}.csv")
        print(f"  - 输出文件: Column_Explanation_{ANCESTRY_NAMES[ancestry_code].split()[0]}.md")
    
    print(f"\n所有解释文档已保存到: {OUTPUT_DIR}")
    print("\n提示:")
    print("- CSV文件可以用Excel打开查看")
    print("- Markdown文件可以用文本编辑器或Markdown查看器打开")
    print("- 数据字典的完整说明在 docs/ 目录下的 .txt 文件中")


if __name__ == "__main__":
    main()

