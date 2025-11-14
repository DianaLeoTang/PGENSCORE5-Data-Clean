"""
HRS 多基因评分数据集 - 批量转换为CSV格式

功能：
1. 读取全部三个祖先群体的数据（非洲、欧洲、西班牙裔）
2. 按照原始格式的行列结构保存为CSV文件
3. 保持原始数据的完整性，不修改任何列或行

使用方法:
python convert_all_to_csv.py
"""

import pandas as pd
from pathlib import Path
import sys

# ==================== 配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "built" / "stata"
OUTPUT_DIR = BASE_DIR / "output"

# 数据文件映射
DATA_FILES = {
    'A': {
        'filename': 'PGENSCOREA_R.dta',
        'output_name': 'PGENSCORE_African.csv',
        'description': '非洲祖先（African Ancestry）'
    },
    'E': {
        'filename': 'PGENSCOREE_R.dta',
        'output_name': 'PGENSCORE_European.csv',
        'description': '欧洲祖先（European Ancestry）'
    },
    'H': {
        'filename': 'PGENSCOREH_R.dta',
        'output_name': 'PGENSCORE_Hispanic.csv',
        'description': '西班牙裔祖先（Hispanic Ancestry）'
    }
}


def check_environment():
    """检查环境和依赖"""
    print("=" * 70)
    print("HRS 多基因评分数据集 - CSV转换工具")
    print("=" * 70)
    
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"\n✗ 错误: 数据目录不存在: {DATA_DIR}")
        print("请确保数据文件在正确的位置")
        sys.exit(1)
    
    # 检查输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n✓ 数据目录: {DATA_DIR}")
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    
    # 检查必要的库
    try:
        import pandas as pd
        print(f"✓ pandas 版本: {pd.__version__}")
    except ImportError:
        print("\n✗ 错误: 未安装 pandas")
        print("请运行: pip install pandas")
        sys.exit(1)


def convert_to_csv(ancestry_code):
    """
    将指定祖先群体的数据转换为CSV格式
    
    参数:
        ancestry_code: 'A', 'E', 或 'H'
    
    返回:
        (成功标志, 数据形状, 输出文件路径)
    """
    config = DATA_FILES[ancestry_code]
    input_file = DATA_DIR / config['filename']
    output_file = OUTPUT_DIR / config['output_name']
    
    print(f"\n{'=' * 70}")
    print(f"处理: {config['description']}")
    print(f"{'=' * 70}")
    print(f"输入文件: {input_file}")
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"✗ 错误: 文件不存在: {input_file}")
        return False, None, None
    
    try:
        # 读取Stata格式数据
        print("正在读取数据...")
        df = pd.read_stata(input_file)
        
        # 显示数据基本信息
        print(f"✓ 成功读取数据")
        print(f"  - 行数: {df.shape[0]:,}")
        print(f"  - 列数: {df.shape[1]:,}")
        print(f"  - 列名示例（前5个）: {list(df.columns[:5])}")
        
        # 检查缺失值
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"  - 缺失值总数: {missing_count:,}")
        else:
            print(f"  - 缺失值: 无")
        
        # 保存为CSV（保持原始格式，不添加索引）
        print(f"\n正在保存为CSV格式...")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✓ 成功保存: {output_file}")
        print(f"  - 文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return True, df.shape, output_file
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    """主函数"""
    # 检查环境
    check_environment()
    
    # 统计信息
    success_count = 0
    total_rows = 0
    total_cols = 0
    results = []
    
    # 处理所有三个数据集
    for ancestry_code in ['A', 'E', 'H']:
        success, shape, output_file = convert_to_csv(ancestry_code)
        
        if success:
            success_count += 1
            total_rows += shape[0]
            total_cols = max(total_cols, shape[1])  # 列数可能不同，取最大值
            results.append({
                'ancestry': DATA_FILES[ancestry_code]['description'],
                'rows': shape[0],
                'cols': shape[1],
                'output_file': output_file.name
            })
    
    # 显示总结
    print(f"\n{'=' * 70}")
    print("转换完成总结")
    print(f"{'=' * 70}")
    print(f"\n成功转换: {success_count} / 3 个数据集")
    print(f"总行数: {total_rows:,}")
    print(f"最大列数: {total_cols}")
    
    if results:
        print(f"\n详细结果:")
        print(f"{'祖先群体':<20} {'行数':<12} {'列数':<10} {'输出文件'}")
        print("-" * 70)
        for r in results:
            print(f"{r['ancestry']:<20} {r['rows']:<12,} {r['cols']:<10} {r['output_file']}")
    
    print(f"\n所有CSV文件已保存到: {OUTPUT_DIR}")
    print("\n提示:")
    print("- CSV文件可以直接用Excel、pandas等工具打开")
    print("- 数据格式与原始Stata文件完全一致")
    print("- 可以使用以下代码读取CSV文件:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('output/PGENSCORE_African.csv')")
    
    return success_count == 3


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

