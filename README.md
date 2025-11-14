# PGENSCORE5-Data-Clean
# HRS 多基因评分数据集 V5 项目说明

## 项目概述

本项目是**健康与退休研究（Health and Retirement Study, HRS）**的多基因评分（Polygenic Scores, PGS）数据集第5版，包含2006-2012年期间收集的遗传数据。该数据集为老年人精准医疗研究提供重要的遗传学基础数据。

## 数据集基本信息

- **数据集名称**: Polygenic Scores V5
- **数据收集时间**: 2006-2012
- **版本**: Release 5 - October 2024
- **数据来源**: HRS (Health and Retirement Study)
- **联系邮箱**: hrsquestions@umich.edu

## 数据结构

### 祖先群体分类

数据集包含三个不同祖先群体的多基因评分数据：

1. **Section A: 非洲祖先（African Ancestry, AA）**
   - 样本量: 3,100 个受访者
   - 文件前缀: `PGENSCOREA_R`

2. **Section E: 欧洲祖先（European Ancestry, EA）**
   - 样本量: 12,090 个受访者
   - 文件前缀: `PGENSCOREE_R`

3. **Section H: 西班牙裔祖先（Hispanic Ancestry, HA）**
   - 样本量: 2,381 个受访者
   - 文件前缀: `PGENSCOREH_R`

### 数据格式

数据集以多种统计软件格式提供，方便不同研究需求：

- **SAS格式**: `.sas7bdat` 文件
- **SPSS格式**: `.sav` 文件
- **Stata格式**: `.dta` 文件
- **原始数据**: `.da` 文件（在 `polys/data/` 目录下）

## 核心变量

### 标识变量

- **HHID**: 家庭标识符（Household Identifier），唯一标识原始家庭
- **PN**: 人员编号（Person Number），在原始家庭内唯一标识受访者

### 主成分分析（Principal Components, PC）

每个祖先群体包含10个祖先特异性主成分，用于控制群体分层：

- **PC1_5A 到 PC1_5E**: 前5个主成分（必须同时使用全部5个或全部10个）
- **PC6_10A 到 PC6_10E**: 后5个主成分
- **注意**: 主成分已打乱（SHUFFLED），使用时必须使用完整的PC集合

## 多基因评分（PGS）类别

数据集包含超过100种不同的多基因评分，涵盖以下主要类别：

### 1. 认知与教育
- 一般认知能力（CHARGE 2015, 2018）
- 教育程度（SSGAC 2016, 2018）
- 神经质（SSGAC 2016）
- 外向性（GPC 2016）

### 2. 精神健康
- 精神分裂症（PGC 2014）
- 双相情感障碍（PGC 2011）
- 重度抑郁症（PGC 2013, 2018）
- 焦虑症（ANGST 2016）
- 注意力缺陷多动障碍（ADHD, PGC 2010, 2017）
- 自闭症（PGC 2017）
- 强迫症（OCD, IOCDF 2017）
- 创伤后应激障碍（PTSD, PGC 2018）
- 反社会行为（BROAD 2017）
- 心理健康交叉障碍（PGC 2013）

### 3. 心血管疾病
- 冠心病（CAD, CARDIoGRAM 2011）
- 心肌梗死（MI, CARDIoGRAM 2015）
- 高血压（COGENT 2017）
- 收缩压/舒张压（COGENT 2017）
- 脉搏压（COGENT 2017）

### 4. 代谢性疾病
- 2型糖尿病（T2D, DIAGRAM 2012, 2024）
  - 包含非洲、欧洲、东亚、西班牙裔、南亚和合并GWAS版本
- 糖化血红蛋白（HbA1c, MAGIC 2017）
- 体重指数（BMI, GIANT 2015, 2018）
- 身高（GIANT 2014, 2018）
- 腰围（GIANT 2015）
- 腰臀比（WHR, GIANT 2015）

### 5. 血脂代谢
- 高密度脂蛋白（HDL, GLGC 2013）
- 低密度脂蛋白（LDL, GLGC 2013）
- 总胆固醇（TC, GLGC 2013）
- 甘油三酯（TG, GLGC 2013）

### 6. 肾脏疾病
- 慢性肾脏病（CKD, CKDGen 2019）
- 肾小球滤过率（eGFR, CKDGen 2019）
- 血尿素氮（BUN, CKDGen 2019）
- 包含欧洲和跨祖先GWAS版本

### 7. 神经退行性疾病
- 阿尔茨海默病（AD, IGAP 2013, 2019, PGC 2021, EADB 2022）
  - 包含含/不含APOE基因的版本
  - 包含不同p值阈值（pT=0.01, pT=1）的版本

### 8. 生殖健康
- 初潮年龄（ReproGen 2014, 2017）
- 绝经年龄（ReproGen 2015, 2021）
- 首次生育年龄（Sociogenome 2016）
- 生育子女数（Sociogenome 2016）

### 9. 行为特征
- 吸烟行为（TAG 2010, GSCAN 2019）
  - 是否吸烟
  - 每日吸烟量
  - 吸烟起始年龄
  - 戒烟
- 饮酒行为（GSCAN 2019, PGC 2018）
  - 每周饮酒量
  - 酒精依赖
- 大麻使用（ICCUKB 2018）

### 10. 其他健康指标
- 主观幸福感（SSGAC 2016）
- 抑郁症状（SSGAC 2016）
- 长寿（CHARGE 2015）
- 血浆皮质醇（CORNET 2014）
- C反应蛋白（CHARGE 2022）

## 数据来源联盟

多基因评分基于以下国际遗传学联盟的研究：

- **CHARGE**: Cohorts for Heart and Aging Research in Genomic Epidemiology
- **GIANT**: Genetic Investigation of ANthropometric Traits
- **PGC**: Psychiatric Genomics Consortium
- **SSGAC**: Social Science Genetic Association Consortium
- **DIAGRAM**: DIAbetes Genetics Replication and Meta-analysis
- **CARDIoGRAM**: Coronary ARtery DIsease Genome-wide Replication and Meta-analysis
- **GLGC**: Global Lipid Genetics Consortium
- **CKDGen**: Chronic Kidney Disease Genetics Consortium
- **COGENT**: Continental Origins and Genetic Epidemiology Network
- **MAGIC**: Meta-Analyses of Glucose and Insulin-related traits Consortium
- **IGAP**: International Genomics of Alzheimer's Project
- **ReproGen**: Reproductive Genetics Consortium
- **GSCAN**: GWAS & Sequencing Consortium of Alcohol and Nicotine use
- **ANGST**: Anxiety NeuroGenetics Study
- **IOCDF**: International Obsessive Compulsive Disorder Foundation
- **BROAD**: Broad Antisocial Behavior Consortium
- **Sociogenome**: Sociogenome Consortium
- **EADB**: European Alzheimer's Disease Biobank

## 文件结构

```
PGENSCORE5/
├── built/                    # 已构建的数据文件
│   ├── sas/                  # SAS格式数据
│   ├── spss/                 # SPSS格式数据
│   └── stata/                # Stata格式数据
├── docs/                     # 文档文件
│   ├── PGENSCORE.txt         # 主文档
│   ├── PGENSCOREA_R.txt      # 非洲祖先数据字典
│   ├── PGENSCOREE_R.txt      # 欧洲祖先数据字典
│   ├── PGENSCOREH_R.txt      # 西班牙裔祖先数据字典
│   └── pgenscoreddv5.pdf     # 详细文档（PDF）
└── polys/                    # 原始数据和处理脚本
    ├── data/                 # 原始数据文件（.da格式）
    ├── sas/                  # SAS导入脚本
    ├── spss/                 # SPSS导入脚本
    └── stata/                # Stata导入脚本
```

## 使用注意事项

### 1. 主成分使用
- 必须同时使用所有5个PC（PC1-5）或所有10个PC（PC1-10）
- 主成分已打乱，不能单独使用某个PC
- 用于控制群体分层和混杂因素

### 2. 数据标准化
- 所有PGS已标准化（均值为0，标准差为1）
- 可以直接用于回归分析

### 3. 缺失值处理
- 大部分PGS变量无缺失值
- 部分变量（如女性特有的生殖健康指标）在男性中有缺失值
- 使用前需检查缺失值模式

### 4. 祖先特异性
- 不同祖先群体的PGS基于不同的GWAS研究
- 使用时需确保PGS与样本祖先匹配
- 某些PGS可能包含跨祖先版本

### 5. 版本信息
- 数据集版本号存储在 `VERSION` 变量中
- 当前版本为5（Release 5 - October 2024）

## 应用场景

该数据集适用于以下研究领域：

1. **精准医疗研究**: 评估遗传风险对健康结局的影响
2. **基因-环境交互**: 研究遗传因素与环境因素的交互作用
3. **疾病预测**: 开发基于多基因评分的疾病风险预测模型
4. **健康老龄化**: 研究遗传因素对老年人健康的影响
5. **社会遗传学**: 研究遗传因素与社会经济地位、教育等的关联
6. **药物基因组学**: 评估遗传因素对药物反应的影响

## 数据引用

使用本数据集进行研究时，请引用：

- HRS Polygenic Scores V5 (Release 5 - October 2024)
- Health and Retirement Study, University of Michigan
- 相关原始GWAS研究（见各PGS变量说明）

## 技术支持

如有问题，请联系：
- **邮箱**: hrsquestions@umich.edu
- **网站**: https://hrs.isr.umich.edu/

## 更新历史

- **Version 5**: October 2024
  - 新增多个PGS变量
  - 更新部分现有PGS
  - 改进数据质量控制

## 免责声明

本数据集仅供研究使用。使用前请仔细阅读HRS数据使用协议，并遵守相关伦理和隐私保护规定。

---

**最后更新**: 2024年10月  
**文档维护**: 根据HRS官方文档整理

