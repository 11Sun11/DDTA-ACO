import csv
import sys
import os
import pandas as pd
import glob
from pathlib import Path
import numpy as np
#将文件夹中的所有文件沿着行拼接成一个文件

def merge(folder_path):
        
    # 第一步：获取所有 CSV 文件的路径，按字母序排序
    file_paths = sorted(glob.glob(f"{folder_path}/*.csv"))  # 替换为你的文件路径

    # 设置目标列数
    target_columns = 20

    # 初始化一个空的 DataFrame 列表
    dataframes = []

    # 第二步：逐行读取数据并处理
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 用逗号分割行并转换为列表
                    row = line.strip().split(',')

                    # 使用最后一列的值补全
                    last_value = row[-1] if row else None

                    # 如果列数少于目标列，进行补全
                    while len(row) < target_columns:
                        row.append(last_value)

                    # 如果列数超过目标列，保留前20列
                    if len(row) > target_columns:
                        row = row[:target_columns]

                    # 将处理后的行添加到 DataFrame
                    dataframes.append(row)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # 将列表转换为 DataFrame，并定义列数
    merged_data = pd.DataFrame(dataframes)

    # 定义输出文件路径
    merged_file = os.path.join(folder_path, "merged_output.csv")

    # 第三步：将合并后的数据写入新的 CSV 文件
    merged_data.to_csv(merged_file, index=False, header=False)
if __name__ == '__main__':
    # merge("./Draw/data/Supple_Elite/chengdu/True/E_Cost")
    merge("./Draw/data/Supple_Elite/porto/True/E_Cost")
    # merge("./code/ACO-MulT-TTE/Draw/data/Supple_Elite/chengdu/True/E_Cost")
    # ./code/ACO-MulT-TTE
    # merge("./Draw/data/Supple_Elite")