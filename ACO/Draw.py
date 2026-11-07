'''
Author: Xin-Can Wang
Date: 2024-11-20 21:19:42
LastEditors: Xin-Can Wang
LastEditTime: 2025-01-21 19:23:37
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/ACO/Draw.py
'''
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# 为蚁群优化算法构建迭代曲线图
def Convergence_curve(Fitn,name=""):
    iterations=list(range(len(Fitn)))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, Fitn, marker='o', linestyle='-', color="#A586A6", label='Objective Value')
    plt.title('Convergence Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.xticks(iterations)
    # plt.grid(True)
    plt.legend()
    plt.ylim(bottom=max(min(Fitn) -0.05,0), top=min(max(Fitn),10) + 0.1)  # 调整 y 轴范围
    plt.axhline(y=min(Fitn), color='r', linestyle='--', label='Best Value Achieved')
    plt.legend()
    # 显示图形
    plt.savefig(f"./code/ACO-MulT-TTE/Draw/pdf/Convergence_curve_{name}.pdf")
    plt.show()
def Visualization(net,ant):
    G=net.G
    CS_list=net.CS_list
    POI_list=net,POI_list
    pos = nx.spring_layout(G)  # 使用 spring 布局
    # 绘制所有边
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=10, font_color='black', font_weight='bold', arrows=True)

    # 突出显示选定的轨迹
    nx.draw_networkx_edges(G, pos, edgelist=ant.path, edge_color='red', width=3)

    # 突出显示选定的节点
    nx.draw_networkx_nodes(G, pos, nodelist=CS_list, node_color='blue', node_shape='o', node_size=3000, label='Charging station')
    nx.draw_networkx_nodes(G, pos, nodelist=POI_list, node_color='orange', node_shape='s', node_size=3000, label='Point of Interest')

    # 在选定节点上添加标签
    for node in ant.path:
        x, y = pos[node]
        plt.text(x, y, f'{node}', fontsize=12, ha='center', color='black', fontweight='bold')

    plt.title("Road Network with Selected Path and Nodes")
    plt.show()


# 利用保存的迭代最优数据来绘制至今最优的收敛曲线
def iter_to_curve(readfile):
    with open(readfile, "r") as file:
        reader = csv.reader(file)   
    # 逐行读取数据
        i=0
        for row in reader:
            i+=1
            row=[float(data) for data in row]
            best_iter=iter_to_best(row)
            draw_by_data(best_iter,i)
# 将迭代最优数据转换为至今最优数据
def iter_to_best(row):
    best_iter=[row[0]]
    for data in row[1:]:
        if data<best_iter[-1]:
            best_iter.append(data)
        else:
            best_iter.append(best_iter[-1])
    return best_iter
def mean_curve(readfile):
   # 读取 CSV 文件
    data=[]
    with open(readfile, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row)==0:
                continue
            row=[float(item) for item in row]
            # 确保每行的长度一致
            while len(row) < 100:
                    row.append(row[-1])  # 用 0 填充
            
            data.append(iter_to_best(row))
    df = pd.DataFrame(data)
    mean_values = df.mean()
    mean_list = mean_values.tolist()
    draw_by_data(mean_list,"mean")
def draw_by_data(best_iter,name):
    iterations=list(range(len(best_iter)))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_iter, marker='o', linestyle='-', color="#A586A6", label='Objective Value')
    plt.title('Convergence Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    # plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0, top=0.7)  # 调整 y 轴范围
    plt.xticks(np.arange(0, len(best_iter), 1))
    plt.axhline(y=min(best_iter), color='r', linestyle='--', label='Best Value Achieved')
    plt.legend()
    # 显示图形
    plt.savefig(f"./code/ACO-MulT-TTE/Draw/pdf/Convergence_curve_{name}.pdf")
    plt.show()
# iter_to_curve("./code/ACO-MulT-TTE/Draw/data/Conver.csv")
# mean_curve("./code/ACO-MulT-TTE/Draw/data/Conver_diversity2.csv")
# data=[4221.08914732933,8370.312982797623,3319.063004255295,3771.7784855365753,199.70482778549194,159.08948755264282,1588.7650411128998]
# print([i/sum(data) for i in data])
def combine_csv(source,target):
    with open(source,"r") as f:
        reader=csv.reader(f)
        with open(target,"a") as t:
            writer=csv.writer(t)
            for row in reader:
                writer.writerow(row)
if __name__ == '__main__':
    # combine_csv("./code/ACO-MulT-TTE/Draw/data/Repair/E_Conver_diversity_True_porto.csv","./code/ACO-MulT-TTE/Draw/data/Repair/E_iters_porto.csv")   
    mean_curve("./code/ACO-MulT-TTE/Draw/data1/Conver_diversity_base.csv")
        