'''
Author: Xin-Can Wang
Date: 2025-01-04 13:48:32
LastEditors: Xin-Can Wang
LastEditTime: 2025-01-15 10:35:17
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/Draw/com.py
'''
import pandas as pd
from scipy.stats import wilcoxon
import csv

# 读取 CSV 文件

def Wilcoxon_sum_test(file):
    df = pd.read_csv(file)  # 假设文件名为 results.csv

    # 获取算法的列名
    algorithms = df.columns

    # 设置显著性水平
    alpha = 1

    # 选择第一个算法
    algo1 = algorithms[1]

    # 保存比较结果的字典
    results = {}
    algorithm="KC_ACO"
    # com_algs=["KC_ACO","RH","TOA","SP","PERP","LACO","LSA"]
    com_algs=["KC_ACO","Tr","Cr","O","PERP","LACO","LSA"]
    # 进行 Wilcoxon 检验
    for j,algo1 in enumerate(algorithms):
        results = {}
        for i,algo2 in enumerate(algorithms[j+1:]):
            # 进行 Wilcoxon 检验
            stat, p_value = wilcoxon(df[algo1], df[algo2])
            
            # 判断结果
            if p_value < alpha:
                # 如果 p 值小于显著性水平，进一步比较效果
                if df[algo1].mean() < df[algo2].mean():
                    result = f"优于{df[algo1].mean()} {df[algo2].mean()} {df[algo1].std()} {df[algo2].std()}"  # algo1 的效果更好
                else:
                    result = f"差于{df[algo1].mean()} {df[algo2].mean()} {df[algo1].std()} {df[algo2].std()}"  # algo1 的效果更差
            else:
                result = "约等于"  # 无显著差异
            results[com_algs[i+j+1]] = {
                'statistic': stat,
                'p-value': p_value,
                'result': result
            }
        

    # 输出结果
        for algo2, result in results.items():
            print(f"Comparison between {com_algs[j]} and {algo2}:")
            print(f"  Statistic: {result['statistic']}, p-value: {result['p-value']}")
            print(f"  Result: {result['result']}\n")
def com_test(file):
    com_res=dict()
   
    with open(file,"r") as f:
        reader=csv.reader(f)
        for row in reader:
            com_res["better"]=[0 for _ in range(len(row)-1)]
            com_res["similar"]=[0 for _ in range(len(row)-1)]
            com_res["worse"]=[0 for _ in range(len(row)-1)]
            break
        for row in reader:
            row=[float(r) for r in row]
            for i,data in enumerate(row[1:]):
                if abs(row[0]-data)<0.001:
                    com_res["similar"][i]+=1
                elif row[0]<data:
                    com_res["better"][i]+=1
                else:
                    com_res["worse"][i]+=1
    print(com_res)

if __name__ == '__main__':
    Wilcoxon_sum_test("./code/ACO-MulT-TTE/Draw/data/Update/multi_Fitn_porto.csv")
    # Wilcoxon_sum_test("./code/ACO-MulT-TTE/Draw/data/Repair/multi_Fitn_chengdu.csv")
    # com_test("./code/ACO-MulT-TTE/Draw/data/KCM5/multi_Fitn.csv")
    # "./code/ACO-MulT-TTE/Draw/data/Elite/com_chengdu.csv"
    # rank_var_para("./code/ACO-MulT-TTE/Draw/data/KCM/multi_Fitn.csv")