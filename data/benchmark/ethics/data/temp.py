import os
import pandas as pd

# 获取当前目录
current_dir = os.getcwd()

# 找到所有 csv 文件
csv_files = [file for file in os.listdir(current_dir) if file.endswith('.csv')]

# 检查是否有找到csv文件
if not csv_files:
    print("当前目录下没有找到CSV文件。")
else:
    # 读取所有 CSV 并合并
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # 合并所有DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # 保存到新的csv文件
    merged_df.to_csv('merged_output.csv', index=False)
    print(f"已合并 {len(csv_files)} 个文件，生成 merged_output.csv。")
