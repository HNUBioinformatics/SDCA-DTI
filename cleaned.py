import pandas as pd

def filter_smiles_salts_and_mixtures(input_path, output_path, smiles_column='smiles'):
    """
    读取CSV文件，丢弃所有SMILES中包含'.'的行，并将结果保存到新文件。

    参数:
        input_path (str): 原始CSV文件的路径。
        output_path (str): 清洗后数据要保存的新CSV文件路径。
        smiles_column (str): 文件中包含SMILES的列名。
    """
    try:
        print(f"正在从以下路径加载数据: {input_path}")
        df = pd.read_csv(input_path)
        original_count = len(df)
        print(f"成功加载原始文件，包含 {original_count} 条记录。")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'。请检查路径。")
        return

    # 检查SMILES列是否存在
    if smiles_column not in df.columns:
        print(f"错误：在文件中找不到名为 '{smiles_column}' 的列。")
        return

    print("\n开始过滤代表盐、溶剂或混合物的样本...")
    
    # 核心过滤逻辑：
    # 1. .astype(str) 确保该列为字符串类型，避免错误
    # 2. .str.contains(r'\.') 检查字符串中是否包含点号 '.'
    # 3. '~' 操作符用于取反，即选择所有不包含点号的行
    cleaned_df = df[~df[smiles_column].astype(str).str.contains(r'\.')].copy()
    final_count = len(cleaned_df)
    
    print("过滤完成！")

    # 保存到新文件
    try:
        cleaned_df.to_csv(output_path, index=False)
        print(f"已将过滤后的干净数据保存到: {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    # 打印最终报告
    print("\n" + "="*50)
    print("         数 据 过 滤 完 成 - 总 结 报 告")
    print("="*50)
    print(f"原始文件样本数量: {original_count}")
    print(f"过滤后新文件样本数量: {final_count}")
    print(f"共丢弃样本数量: {original_count - final_count}")
    print("="*50)

if __name__ == '__main__':
    # ===============================================================
    # === 请在这里配置你的文件路径 ===
    
    # 你的原始数据文件
    original_file = "E:\\code_DTI\\mycode\\dataset\\drugbank.csv"
    
    # 你希望保存过滤后数据的新文件名
    filtered_file = "E:\\code_DTI\\mycode\\dataset\\drugbank_cleaned.csv"

    # ===============================================================

    filter_smiles_salts_and_mixtures(input_path=original_file, 
                                     output_path=filtered_file, 
                                     smiles_column='smiles')