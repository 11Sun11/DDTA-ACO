'''
Author: Xin-Can Wang
Date: 2024-11-28 19:24:15
LastEditors: Xin-Can Wang
LastEditTime: 2025-03-31 22:15:32
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/Draw/Draw.py
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import os
import re
from matplotlib.ticker import MaxNLocator
# 提取一个矩阵中的非零值，并绘制密度分布图
def non_zero_den(matrix,name=""):
    # 提取非零值
    non_zero_values = matrix[matrix != 0]
    # 绘制密度分布图
    plt.figure(figsize=(8, 6))
    sns.kdeplot(non_zero_values, fill=True, color='blue', bw_adjust=0.5)
    plt.title('Density Distribution of Non-Zero Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f"./code/SCO-MulT-TTE/Draw/pdf/density_{name}.pdf")
    plt.grid()
    plt.show()
def get_mean(file_path):
# 读取 CSV 文件
# # 替换为您的 CSV 文件路径
    data = pd.read_csv(file_path)
    # 计算每一列的平均值
    column_means = data.mean()
    col_std=data.std()
    for m,s in zip(column_means,col_std):
        print(f"{round(m,4)}({round(s,4)})")
    print([round(d,4) for d in column_means])
    print([round(d,4) for d in col_std])
    # 打印结果
    return column_means,col_std
def get_succ_o_inf(file_path):
# 读取 CSV 文件
# # 替换为您的 CSV 文件路径
    succ=0
    all_ins=0
    with open(file_path,"r") as f:
        reader1=csv.reader(f)
        for row in reader1:
            row1=float(row[0])
            if row1 <1000:
                succ+=1
    # 计算每一列的平均值
            all_ins+=1
    succ/=all_ins
    print(round(succ,4),end=" ")
    # 打印结果
    return succ
def get_min(file_path):
    data=[]
    min_list=[]
    with open(file_path,"r") as f:
        reader1=csv.reader(f)
        for row in reader1:
            row=[float(r) for r in row]
            min_list.append(min(row))
    array_list=np.array(min_list)
    array_list=array_list.reshape(-1,20)
    row_means = np.mean(array_list, axis=1)
    row_means = np.round(row_means, 2)
    print(row_means)  # 输出: [2. 5. 8.]
    return min_list,row_means
def get_single_col(file,column_index):
    column_data=[]
    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:  # 确保行有足够的列
                column_data.append(float(row[column_index]))
    return column_data
def com_two1(files):
    subs=[]
    for f in files:
        f1=f[0]
        f2=f[1]
        out1=get_min(f1)
        out2=get_single_col(f2,0)
        sub=[abs(i-j) for i,j in zip(out1,out2)]
        sub=[i for i in sub if i<200]
        subs.append(sub)
        # out1=get_min(f1)
        # out2=get_single_col(f2,0)
        # subs.append([abs(i-j) for i,j in zip(out1,out2)])
    violin_mutli_draw(subs,"The difference",["Chengdu","Porto"])
    # outs=[out1,out2]
    # outs=np.array(outs)
    # outs=outs.T
    # np.savetxt("./code/SCO-MulT-TTE/Draw/data/Elite/diff_surr.csv", outs, delimiter=",", comments='', fmt='%.4f')
    return
def get_o(matrix):
    ins_number,algs_number=matrix.shape
    succ=[0 for i in range(algs_number)]
    Fitn=[]
    for row in matrix:
        flag=True
        for i,number in enumerate(row):
            if number<1000:
                succ[i]+=1
                flag=False
        if flag:
            Fitn.append(row)
    succ=[round(i/ins_number,3) for i in succ]
    if len(Fitn)>0:
        
        Fitn=np.array(Fitn)
    mean=np.mean(matrix,axis=0).tolist()
    std=np.std(matrix,axis=0).tolist()
    for i,j in zip(mean,std):
        print(f"{round(i,4)}({round(j,4)})",end="&")
    print()
    print(succ)

def com_all(f_other,f_main,isFit=True):
    if isFit:
        _,my_data=get_min(f_main)
    else:
        my_data= np.loadtxt(f_main, delimiter=',', skiprows=0)
    my_data=my_data.reshape(-1,1)
    matrix = np.loadtxt(f_other, delimiter=',', skiprows=0)
    com_data=np.hstack((matrix[:20,:],my_data))
    get_o(com_data)
    all_alg=["PERP","RH","TOA","SCSHT","LSCO","LSA","DTC-SCO"]
    ranks={}
    succ={}
    com_data=np.round(com_data,4)
    # print(np.mean(com_data,axis=0))
    # print(com_data)
    for i,alg in enumerate(all_alg):
        ranks[alg]=[]
        succ[alg]=[]
    average_rank=[]
    all_ranks=[]
    rank_count=[]
    for row in com_data:
        row_ranks=[]
        sorted_row = sorted(row)
        count=0
        for i in range(len(sorted_row)):
            if row[i]>1000:
                succ[all_alg[i]].append(1)
            else:
                count+=1
                succ[all_alg[i]].append(0)

            ranks[all_alg[i]].append(sorted_row.index(row[i])+1)
            row_ranks.append(sorted_row.index(row[i])+1)
        all_ranks.append(row_ranks)
        rank_count.append(count)
    # print(all_ranks)
    sorted_indices = sorted(range(len(rank_count)), key=lambda i: rank_count[i], reverse=True)
    select_ranks=dict()
    # all_succ=[sum[i] for i in all_ranks]
    avg_ranks={}
    for rank,data in ranks.items():
        avg_rank=round(np.mean(data),4)
        std_rank=round(np.std(data),4)
        select_ranks[rank]=[data[i] for i in sorted_indices[:10]]
        avg_ranks[rank]=f"{avg_rank}({std_rank})"

    print(avg_ranks)
    # print(succ)
    # print(ranks)
    # print(select_ranks)
    

def com_two2(f1,f2):
    out1=get_min(f1)
    out2=get_min(f2)
    outs=[out1,out2]
    outs=np.array(outs)
    outs=outs.T
    np.savetxt("./code/SCO-MulT-TTE/Draw/data/Elite/com_chengdu.csv", outs, delimiter=",", comments='', fmt='%.4f')
    return
def get_oinf(file_path):
# 读取 CSV 文件
    data=[]
    
    with open(file_path,"r") as f:
        reader1 = csv.reader(f)
        number=0
        succ_test=[0 for _ in range(6)]
        for row1 in reader1:
            number+=1
            row=[]
            for i,r in enumerate(row1):
                
                if r[0]=="t":
                    numbers = re.findall(r'\d+', r)
                    r=numbers[0]
                r=float(r)
                # if (i==0 or i>=5) and number>=5:
                #     r*=20
                row.append(float(r))
            for i,d in enumerate(row):
                if row[i]=="inf" or row[i]>=1000:
                    continue
                elif i>=5:
                    # row[i]*=20
                    if abs(row[i]-1000)<=0.01:
                        continue
                    else:
                        succ_test[i]+=1
                else:
                    succ_test[i]+=1
            if row[0]=="inf":
                continue
           
            data.append([float(r) for r in row])
    # succ_test=[i/20 for i in succ_test]
    
    data=np.array(data)
    column_means=np.mean(data,axis=0)
    col_std=np.std(data,axis=0)
    print([round(i/number,4) for i in succ_test])
    print([round(d,4) for d in column_means])
    print([round(d,4) for d in col_std])
    return data,column_means,col_std
def get_oinf_single(file_path):
# 读取 CSV 文件
    data=[]
    
    with open(file_path,"r") as f:
        reader1 = csv.reader(f)
        number=0
        succ_test=0
        row=[[] for _ in range(1)]
        for row1 in reader1:
            number+=1
            
            for i,r in enumerate(row1):
                if float(r)<=1000:
                    r=float(r)
                    # if (i==0 or i>=5) and number>=5:
                    #     r*=20
                    succ_test+=1
                    row[i].append(float(r))
                else:
                   pass
           
            # data.append([float(r) for r in row])
    data=np.array(row)
    column_means=np.mean(data,axis=1)
    col_std=np.std(data,axis=1)
    print(round(succ_test/number,4))
    for m,s in zip(column_means,col_std):
        print(f"{round(m,4)}({round(s,4)})")
    # print([col_std])
    return data,column_means,col_std
def sumcol_count(file_path):
    df = pd.read_csv(file_path)

    # 沿列求和
    column_sums = df.sum()
    summ=sum(column_sums)
    # 输出结果
    print([i/summ for i in column_sums])
# 对单个文件中每一行不同的算法设置进行排名
def rank_var_para(file1):
    # with open(file1, "r") as csvfile1:
    #     reader1 = csv.reader(csvfile1)
    #     data1 = list(reader1)
    data1,_,_=get_oinf(file1)

    # 比较两个CSV文件的数据
    rank = [[] for _ in range(len(data1[0]))]
    for i in range(len(data1)):
        row1 = list(data1[i])
        sorted_row = sorted(row1)
        for k, r in enumerate(row1):
            rank[k].append(sorted_row.index(row1[k])+1)
    avg_rank=[sum(i)/len(i) for i in rank]
    print([sum(i)/len(i) for i in rank])
    rank=np.array(rank)
    rank_std=np.std(rank,axis=1)
    rank_mean=np.std(rank,axis=1)
    return rank, avg_rank,rank_std
# get_mean("./code/SCO-MulT-TTE/Draw/data/modeule_time_rec.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/runT_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/early_stop/Fitn_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/ants_num/Fitn_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/short_inc/Fitn_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/adapt/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/adapt/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/adapt/runT_optimizers.csv")
# # get_mean("./code/SCO-MulT-TTE/Draw/data/short_inc/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/early_stop/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/early_stop/runT_optimizers.csv")

# get_mean("./code/SCO-MulT-TTE/Draw/data/ants_num/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/ants_num/runT_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/ants_num/Fitn_optimizers.csv")

# get_oinf("./code/SCO-MulT-TTE/Draw/data/com/Fitn_optimizers2.csv")
# get_oinf("./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers2.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/q0/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/q0/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/q0/runT_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/alpha/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/alpha/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/alpha/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/q0/runT_optimizers.csv")
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/aco_type/Fitn_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/aco_type/Fitn_optimizers.csv")
# 在搜索的早期阶段更加倾向于使用启发式信息，在后期更加倾向于使用信息素矩阵
# rank_var_para("./code/SCO-MulT-TTE/Draw/data/ants_num/runT_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/runT_optimizers_diffpart.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/runT_optimizers2.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/Fitn_optimizers2.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/runT_part_optimizers.csv")
# get_mean("./code/SCO-MulT-TTE/Draw/data/Fitn_part_optimizers.csv")
# sumcol_count("./code/SCO-MulT-TTE/Draw/data/modeule_time_diffpart_rec0.csv")
# time_rec=[683.1103162765503,1460.3267414569855,628.5299456119537,42.0631206035614,12.739458799362183,66.88500380516052,3.3899664878845215]
# time_rec=[965.1305439472198,9126.146344661713,2096.1023840904236,314.4297432899475,22.117453575134277,674.6287071704865,137.05121397972107]
# print([i/(20*747.562397) for i in time_rec])

def mean_with_curve(pre, files,city,matrix="Fitn"):
    city=city.capitalize()
    plt.figure(figsize=(10, 6))
    # colors=['blue','orange']
    colors = ["#A586A6", "#91c8e8"]
    colors = ["#F25c05", "#91c8e8"]
    labels=["DDTA-ACO","w/o EAS"]
    markers=["v","x"]
    for i,file in enumerate(files):
    #     df = pd.read_csv(file, header=None)

        full_path = os.path.join(pre, file)
        # full_path = file
    # # 将数据转换为 NumPy 数组
    #     data = df['value'].to_numpy()
        data=fill_data2(full_path)
        data = np.array(data)
        Q1_1 = np.percentile(data.T, 30, axis=1)  # 第一四分位数
        Q3_1 = np.percentile(data.T, 70, axis=1)  # 第三四分位数
        x=np.array(list(range(len(data[0]))))
        # 计算均值和标准差
        mean_y = np.mean(data, axis=0)
        # print(mean_y[-1])
        std_y = np.std(data, axis=0)
        print(f"{round(mean_y[-1],4)}({round(std_y[-1],4)})")

        # 创建图形
        

        # 绘制均值曲线
        plt.plot(x, mean_y, color=colors[i], label=labels[i], marker=markers[i], linewidth=2)

        # 绘制幅状图（标准差范围）
        plt.fill_between(x,Q1_1, Q3_1, color=colors[i], alpha=0.2)

    # 设置图表属性
    plt.title(f'Convergence Curves on {city}', fontweight="bold", fontsize=20)
    plt.xlabel('Iterations', fontweight="bold", fontsize=20)
    plt.ylabel(matrix, fontweight="bold", fontsize=20)
    # if city=="Chengdu":
    #     y_right=3
    # else:
    #     y_right=1.2
    # plt.ylim(0,y_right)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6)) 
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlim(1,20)
    plt.xticks([0,5,10,15,20], fontsize=18)
    # plt.xticks([0,5,10,15,20], fontsize=18)
    # plt.xticks([0,5,10,15,20], )
    # plt.([0,1,2,3,4,5])
    plt.legend(fontsize=18)
    # plt.grid()
    full_path = os.path.join(pre, f"mean_com_{city}.pdf")
    plt.savefig(full_path)
    # 显示图形
    plt.show()
def draw_feasible_ratio(pre,files):
    colors=["#CDD5C6","#FFE2C3","#B7BCBF"]
    # colors=["#D99789","#FFE2C3","#B7BCBF"]
    colors=["#F27405","#F25061","#B7BCBF"]
    datas_std=[]
    datas=[]
    plt.figure(figsize=(4, 4))
    ratios=[[] for _ in range(2)]
    for i,file in enumerate(files):
        file=os.path.join(pre,file)
        with open(file,"r") as f:
            reader=csv.reader(f)
            for row in reader:
                ratios[i].append(len(row))
    
        # ratios=np.array(ratios).T
        # # ratios=np.array(ratios)
        # column_means = np.mean(ratios,axis=0)
        # col_std=np.std(ratios,axis=0)
        # datas.append(column_means)
        # datas_std.append(col_std)
    x1 = np.array([0])  # 第一组 x 位置
    x2 = np.array([1])  # 第二组 x 位置，向右偏移
    x=[x1,x2]
    # labels=["SL-SCO","w/o RPA","w/o EAS"]
    labels=["Chengdu","Porto"]
    # 设置柱子宽度
    width = 0.15  # 根据数据量调整宽度
    plt.boxplot(ratios, labels=labels)
    for i in range(len(ratios)):
        y = ratios[i]
        x = np.random.normal(i + 1, 0.05, size=len(y))  # 在 x 轴上添加一些随机噪声以便更好地显示
        plt.scatter(x, y, color=colors[i], alpha=0.8, s=5)  # 散点图
    # for j,k in enumerate(datas):
    #     for i in range(3):
    #         plt.bar(x[j] + i * width, k[i], width,yerr=datas_std[j][i], label=labels[i] if j==0 else "", color=colors[i])
    # plt.bar(labels,column_means,yerr=col_std, color=colors)
    # plt.title('RunT Analysis', fontweight="bold", fontsize=18)
    # plt.xlabel('Iterations', fontweight="bold", fontsize=18)
    plt.ylabel('Feasible Ratio', fontweight="bold", fontsize=18)
    # plt.ylim(0,1)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(rotation=45)
    # plt.yticks(fontsize=14)
    # plt.xlim(1,20)
    # plt.xticks([0,5,10,15,20], fontsize=14)
    # plt.([0,1,2,3,4,5])
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    full_path = os.path.join(pre, "feasible_ratio2.pdf")
    plt.savefig(full_path)
    # 显示图形
    plt.show()
    return
def draw_iters(pre,files_list):
    datas=[]
    datas_std=[]
    errors=[]
    errors_neg=[]
    # colors=["#F27405","#F25061","#B7BCBF"]
    # colors=["#CDD5C6","#FFE2C3","#B7BCBF"]
    colors=["#A9D9D9","#F2C063","#F2B694"]

    # for file in files:
    for files in files_list:
        ratios=[[] for _ in range(len(files))]
        for file in files:
            ratios=[]
            file=os.path.join(pre,file)
            with open(file,"r") as f:
                reader=csv.reader(f)
                for row in reader:
                    ratios.append(len(row))
            ratios=np.array(ratios).T
            Q1_1 = np.percentile(ratios, 15, axis=0)  # 第一四分位数
            Q3_1 = np.percentile(ratios, 85, axis=0)  # 第三四分位数
            # ratios=np.array(ratios)
            column_means = np.mean(ratios,axis=0)
            col_std=np.std(ratios,axis=0)
            datas.append(column_means)
            datas_std.append(col_std)
            errors.append(column_means-Q1_1)  # 计算上误差
            errors_neg.append(Q3_1-column_means)  # 计算下误差
        datas.append(100)
    x1 = np.array([0])  # 第一组 x 位置
    x2 = np.array([0.5])  # 第二组 x 位置，向右偏移
    x=[x1,x2]
    # labels=["DTC-SCO","w/o RPA","w/o EAS"]
    ticks=["Chengdu","Porto"]
    labels=["DTC-SCO","w/o EAS","w/o RPA"]
    # 设置柱子宽度
    width = 0.15  # 根据数据量调整宽度
    for j in range(2):
        for i in range(3):
            plt.bar(x[j] + i * width, datas[j*3+i], width, label=labels[i] if j==0 else "", color=colors[i])
    # plt.bar(labels,column_means,yerr=col_std, color=colors)
    # plt.title('RunT Analysis', fontweight="bold", fontsize=18)
    # plt.xlabel('Iterations', fontweight="bold", fontsize=18)
   
    plt.ylabel('Iterations', fontweight="bold", fontsize=18)
    # plt.ylim(0,1)
    # plt.xticks(fontweight='bold', fontsize=18)
    plt.xticks([0.15,0.65],["Chengdu","Porto"],fontweight='bold', fontsize=18)
    plt.yticks(fontsize=16,rotation=45)
    # plt.yticks(fontsize=14)
    # plt.xlim(1,20)
    # plt.xticks([0,5,10,15,20], fontsize=14)
    # plt.([0,1,2,3,4,5])
    plt.legend()
    # plt.grid()
    plt.tight_layout()
    full_path = os.path.join(pre, "iters.pdf")
    plt.savefig(full_path)      
    return
def draw_bar(pre, files):
    plt.figure(figsize=(10, 6))
    # colors=['blue','orange']
    colors = ["#A586A6", "#91c8e8"]
    colors=["#A9D9D9","#F25C05","#F2C063","#F2B694","#F27405"]
    # colors = ["#F25c05", "#91c8e8"]
    #A9D9D9
#F25C05
#F2C063
#F2B694
#F27405
    files = os.path.join(pre, files)
    data = pd.read_csv(files)
    # 计算每一列的平均值
    data=np.array(data)
    data[:,0]=data[:,0]-data[:,4]
    data[:,1]=data[:,1]-data[:,5]
    data[:,4]=data[:,4]+data[:,5]
   
    column_means = np.mean(data,axis=0)
    col_std=np.std(data,axis=0)
    labels=["SC","MulT-TTE","FEs","RPA","EAS","MEM"]
    labels=["SC","FEs","MulT-TTE","EAS","RPA"]
    data,_=get_mean(files)
    # for i,file in enumerate(files):
    # #     df = pd.read_csv(file, header=None)

    #     full_path = os.path.join(pre, file)
    # # # 将数据转换为 NumPy 数组
    # #     data = df['value'].to_numpy()
    #     data=fill_data2(full_path)
    #     data = np.array(data)
    #     Q1_1 = np.percentile(data.T, 30, axis=1)  # 第一四分位数
    #     Q3_1 = np.percentile(data.T, 70, axis=1)  # 第三四分位数
    #     x=np.array(list(range(len(data[0]))))
    #     # 计算均值和标准差
    #     mean_y = np.mean(data, axis=0)
    #     print(mean_y[-1])
    #     std_y = np.std(data, axis=0)

        # 创建图形
        

        # 绘制均值曲线
    plt.bar(labels,column_means[:5],yerr=col_std[:5], color=colors)

    # 绘制幅状图（标准差范围）
    # plt.fill_between(x,Q1_1, Q3_1, color=colors[i], alpha=0.2)

    # 设置图表属性
    plt.title('RunT Analysis', fontweight="bold", fontsize=18)
    # plt.xlabel('Iterations', fontweight="bold", fontsize=18)
    plt.ylabel('RunT(s)', fontweight="bold", fontsize=18)
    # plt.ylim(0,1)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.xlim(1,20)
    # plt.xticks([0,5,10,15,20], fontsize=14)
    # plt.([0,1,2,3,4,5])
    plt.legend()
    # plt.grid()
    full_path = os.path.join(pre, "runT_Porto.pdf")
    plt.savefig(full_path)
    # 显示图形
    plt.show()

def read_conv_file(file):
    with open(file,"r") as f:
        ins_min=[]
        multi_ins_res=[]
        reader=csv.reader(f)
        for i,row in enumerate(reader):
            if i%20==0:
                ins_min=[]
                if i!=0:
                    multi_ins_res.append(ins_min)
            row=[float(r) for r in row]
            row_min=min(row)
            ins_min.append(row_min)
        multi_ins_res=np.array(multi_ins_res)
        mean_res=np.mean(multi_ins_res,axis=1)
        std_res=np.std(multi_ins_res)
    return mean_res,std_res
def com_read(file_com):
    datas=[]
    with open(file_com,"r") as file:
        reader=csv.reader(file_com)
        for row in reader:
            row=[float(r) for r in row]
            datas.append(row)
    return np.array(datas)
def rank_com(file_slaco,file_com):
    slaco_data=read_conv_file(file_slaco)
    com_data=com_read(file_com)
    datas=np.column_stack((slaco_data,com_data))
    print(datas)
    ranks=rank_rows_ascending(datas)
    print(ranks)
    return

def rank_rows_ascending(matrix):
    # 使用 argsort 得到排序索引
    sorted_indices = np.argsort(matrix, axis=1)
    # 创建一个与原始矩阵形状相同的数组
    ranks = np.empty_like(sorted_indices)
    # 为每个元素赋予排名
    ranks[np.arange(matrix.shape[0])[:, None], sorted_indices] = np.arange(1, matrix.shape[1] + 1)
    return ranks
    


            
        


def draw_bar2(pre, files):
    plt.figure(figsize=(10, 6))
    # colors=['blue','orange']
    colors = ["#A586A6", "#91c8e8"]
    colors=["#A9D9D9","#F25C05","#F2C063","#F2B694","#F27405"]
    # colors = ["#F25c05", "#91c8e8"]
    #A9D9D9
#F25C05
#F2C063
#F2B694
#F27405
    datas=[]
    datas_std=[]
    errors=[]
    errors_neg=[]
    total_time=[]
    for file in files:
        file = os.path.join(pre, file)
        data = pd.read_csv(file)
        # 计算每一列的平均值
        data=np.array(data)
        data[:,0]=data[:,0]-data[:,4]
        data[:,1]=data[:,1]-data[:,5]
        data[:,4]=data[:,4]+data[:,5]
        Q1_1 = np.percentile(data, 25, axis=0)  # 第一四分位数
        Q3_1 = np.percentile(data, 75, axis=0)  # 第三四分位数
        # ratios=np.array(ratios)
        column_means = np.mean(data,axis=0)
        total_time.append(column_means[0]+column_means[1]+column_means[3]+column_means[4])
        col_std=np.std(data,axis=0)
        errors.append(column_means-Q1_1)  # 计算上误差
        errors_neg.append(Q3_1-column_means)  # 计算下误差
        column_means = np.mean(data,axis=0)
        col_std=np.std(data,axis=0)
        labels=["SC","CRTE","FEs","RPA","EAS","MEM"]
        labels=["SC","FEs","CRTE","EAS","RPA"]
        indexes=[]
        # data,_=get_mean(files)
        datas.append(column_means)
        datas_std.append(col_std)
    # for i,file in enumerate(files):
    # #     df = pd.read_csv(file, header=None)

    #     full_path = os.path.join(pre, file)
    # # # 将数据转换为 NumPy 数组
    # #     data = df['value'].to_numpy()
    #     data=fill_data2(full_path)
    #     data = np.array(data)
    Q1_1 = np.percentile(data, 30, axis=1)  # 第一四分位数
    Q3_1 = np.percentile(data, 70, axis=1)  # 第三四分位数
    #     x=np.array(list(range(len(data[0]))))
    #     # 计算均值和标准差
    #     mean_y = np.mean(data, axis=0)
    #     print(mean_y[-1])
    #     std_y = np.std(data, axis=0)

        # 创建图形
        

        # 绘制均值曲线
    # plt.bar(labels,column_means[:5],yerr=col_std[:5], color=colors)
    x1 = np.array([0])  # 第一组 x 位置
    x2 = np.array([1.15])  # 第二组 x 位置，向右偏移
    x=[x1,x2]
    # 设置柱子宽度
    width = 0.2  # 根据数据量调整宽度

    # 创建柱状图
    for j,k in enumerate(datas):
        for i in range(5):
            plt.bar(x[j] + i * width, k[i]/total_time[j], width, label=labels[i] if j==0 else "", color=colors[i])
            plt.text(x[j] + i * width, k[i]/total_time[j] + 0.01, f'{k[i]/total_time[j]:.1%}', ha='center', va='bottom', fontweight="bold", fontsize=14)

    # 绘制幅状图（标准差范围）
    # plt.fill_between(x,Q1_1, Q3_1, color=colors[i], alpha=0.2)

    # 设置图表属性
    plt.title('RunT Analysis', fontweight="bold", fontsize=18)
    # plt.xlabel('Iterations', fontweight="bold", fontsize=18)
    plt.ylabel('RunT Ratio', fontweight="bold", fontsize=18)
    plt.ylim(0,0.6)
    plt.xticks([0.4,1.55],["Chengdu","Porto"],fontweight='bold', fontsize=18)
    plt.yticks(fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.xlim(1,20)
    # plt.xticks([0,5,10,15,20], fontsize=14)
    # plt.([0,1,2,3,4,5])
    plt.legend()
    # plt.grid()
    full_path = os.path.join(pre, "RunT_Chengdu_Porto.pdf")
    plt.savefig(full_path)
    # 显示图形
    plt.show()
def draw_barwoRPA(pre, files):
    plt.figure(figsize=(10, 6))
    # colors=['blue','orange']
    colors = ["#A586A6", "#91c8e8"]
    colors=["#A9D9D9","#F25C05","#F2C063","#F2B694","#F27405"]
    # colors = ["#F25c05", "#91c8e8"]
    #A9D9D9
#F25C05
#F2C063
#F2B694
#F27405
    datas=[]
    datas_std=[]
    errors=[]
    total_time=[]
    errors_neg=[]
    for file in files:
        file = os.path.join(pre, file)
        data = pd.read_csv(file)
        # 计算每一列的平均值
        data=np.array(data)
        data[:,0]=data[:,0]-data[:,4]
        data[:,1]=data[:,1]-data[:,5]
        data[:,4]=data[:,4]+data[:,5]
        Q1_1 = np.percentile(data, 25, axis=0)  # 第一四分位数
        Q3_1 = np.percentile(data, 75, axis=0)  # 第三四分位数
        # ratios=np.array(ratios)
        column_means = np.mean(data,axis=0)
        total_time.append(column_means[0]+column_means[1]+column_means[3]+column_means[4])
        col_std=np.std(data,axis=0)
        errors.append(column_means-Q1_1)  # 计算上误差
        errors_neg.append(Q3_1-column_means)  # 计算下误差
        column_means = np.mean(data,axis=0)
        col_std=np.std(data,axis=0)
        labels=["SC","MulT-TTE","FEs","RPA","EAS","MEM"]
        labels=["SC","FEs","MulT-TTE","EAS","RPA"]
        indexes=[]
        # data,_=get_mean(files)
        datas.append(column_means)
        datas_std.append(col_std)
    # for i,file in enumerate(files):
    # #     df = pd.read_csv(file, header=None)

    #     full_path = os.path.join(pre, file)
    # # # 将数据转换为 NumPy 数组
    # #     data = df['value'].to_numpy()
    #     data=fill_data2(full_path)
    #     data = np.array(data)
    Q1_1 = np.percentile(data, 30, axis=1)  # 第一四分位数
    Q3_1 = np.percentile(data, 70, axis=1)  # 第三四分位数
    #     x=np.array(list(range(len(data[0]))))
    #     # 计算均值和标准差
    #     mean_y = np.mean(data, axis=0)
    #     print(mean_y[-1])
    #     std_y = np.std(data, axis=0)

        # 创建图形
        

        # 绘制均值曲线
    # plt.bar(labels,column_means[:5],yerr=col_std[:5], color=colors)
    x1 = np.array([0])  # 第一组 x 位置
    x2 = np.array([1.15])  # 第二组 x 位置，向右偏移
    x=[x1,x2]
    # 设置柱子宽度
    width = 0.2  # 根据数据量调整宽度

    # 创建柱状图
    for j,k in enumerate(datas):
        for i in range(5):
            # plt.bar(x[j] + i * width, k[i]/total_time, width, label=labels[i] if j==0 else "", color=colors[i])
            plt.text(x[j] + i * width, k[i]+0.01,f'{k[i]/total_time[j]:.1%}', ha='center', va='bottom', fontweight="bold", fontsize=14)
            plt.bar(x[j] + i * width, k[i], width, label=labels[i] if j==0 else "", color=colors[i])
            # plt.text(x[j] + i * width, k[i], f'{k[i]/total_time:.1%}', ha='center', va='bottom', fontweight="bold", fontsize=14)

    # 绘制幅状图（标准差范围）
    print(total_time)
    # plt.fill_between(x,Q1_1, Q3_1, color=colors[i], alpha=0.2)

    # 设置图表属性
    plt.title('RunT Analysis', fontweight="bold", fontsize=18)
    # plt.xlabel('Iterations', fontweight="bold", fontsize=18)
    plt.ylabel('RunT', fontweight="bold", fontsize=18)
    # plt.ylim(0,1)
    plt.xticks([0.4,1.55],["Chengdu","Porto"],fontweight='bold', fontsize=18)
    plt.yticks(fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.xlim(1,20)
    # plt.xticks([0,5,10,15,20], fontsize=14)
    # plt.([0,1,2,3,4,5])
    plt.legend()
    # plt.grid()
    full_path = os.path.join(pre, "RunT_Chengdu_Porto_wo_CRTE.pdf")
    plt.savefig(full_path)
    # 显示图形
    plt.show()
# 填充数据
def fill_data(df):
    max_length = df.apply(lambda row: len(row.dropna()), axis=1).max()
    for index, row in df.iterrows():
        last_value = row.dropna().iloc[-1]  # 获取该行最后一个有效值
        while len(row.dropna()) < max_length:
            row = row.append(pd.Series([last_value]), ignore_index=True)
        df.loc[index] = row
    return df.to_numpy()
def fill_data2(file):
    data=[]
    max_len=0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0]=="inf":
                continue
            best_value=float(row[0])
            new_row=[best_value]
            for value in row:
                if float(value)<best_value:
                    best_value=float(value)
                new_row.append(best_value)
            if len(new_row)>max_len:
                max_len=len(new_row)
            # while len(new_row)<20:
            #     new_row.append(best_value)
            data.append(new_row)
        new_data=[]
        for new_row in data:
            while len(new_row)<21:
                new_row.append(new_row[-1])
            if len(new_row)>21:
                new_row=new_row[:21]
        
            new_data.append(new_row)
    return np.array(new_data)
def violin_mutli_draw(data, titles, xlabels, name="violin"):
    # 创建figure和子图
    fig, ax = plt.subplots()
    # data=np.array(data)
    # 遍历子图并绘制小提琴
    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(data) + 1),
        widths=0.8,
    )
    ax.set_title(titles)
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("dataset")
    ax.set_ylabel("The estimation error")
    # ax.set_xticklabels([f"Group{i+1}" for i in range(len(data[i]))])
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plt.cm.viridis(j / (len(data) - 1)))
        pc.set_edgecolor("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cbars"].set_color("black")
    # legend_elements = [
    #     Patch(
    #         facecolor=parts["bodies"][i].get_facecolor(),
    #         edgecolor="black",
    #         label=f"Position {i+1}",
    #     )
    #     for i in range(len(parts["bodies"]))
    # ]

    # # 添加统一的图例
    # fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8)
    # 调整子图间距
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.savefig("./code/SCO-MulT-TTE/Draw/data/ants_num" + name+titles + ".pdf")
    # 显示图形
    plt.show()
# mean_with_curve("./code/SCO-MulT-TTE/Draw/data/Elite",["Conver_diversity_False.csv","Conver_diversity_True.csv"])
# mean_with_curve("./code/SCO-MulT-TTE/Draw/data/aco_type",["E_Conver_diversity_aco.csv","E_Conver_diversity_mm.csv"])
# 双轴
def RunT_fitn(pre,name1,name2):
    # x=np.arange(0.2, 1, 0.2).tolist()
    x=[30,40,50,60,70]
    # x=[4,6,8,10,12]
    data1,y1,std1=rank_var_para(os.path.join(pre,name1))
    data2,y2,std2=rank_var_para(os.path.join(pre,name2))
    # violin_mutli_draw(data1,"Fitn",x)
    # violin_mutli_draw(data2,"RunT",x)
    fig, ax1 = plt.subplots()

    # 第一个指标
    ax1.set_xlabel(r'The number of ants $N$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    # ax1.set_xlabel(r'The select probablity $\lambda$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.set_ylabel('Fitn', color='tab:blue',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.plot(x, y1, color='tab:blue', label='Fitn',marker="s")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.fill_between(x, y1 - std1, y1 + std1, color='tab:blue', alpha=0.2, label='Standard Deviation')
    ax1.set_ylim(0,len(x)+1)
    ax1.set_yticks(np.arange(1, len(x) + 1))
    ax1.set_yticklabels(np.arange(1, len(x) + 1))
    
    # ax1.set_yticklabels(np.arange(1, len(x) + 1))
    # 创建第二个 Y 轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('RunT', color='tab:orange',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax2.plot(x, y2, color='tab:orange', label='RunT', marker='v')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    # ax2.fill_between(x, y2 - std2, y2 + std2, color='tab:orange', alpha=0.2, label='Standard Deviation')
    fig.tight_layout()  # 防止标签重叠
    # plt.title('Fitn-RunT')
    ax2.set_ylim(0,len(x))
    ax1.set_xticks([round(i,2) for i in x])
    ax2.set_yticks(np.arange(1, len(x) + 1))
    ax2.set_yticklabels(np.arange(1, len(x) + 1))
    # ax2.set_xticklabels([round(i,2) for i in x])
    ax2.set_xticklabels(x)

    full_path = os.path.join(pre, "sample.pdf")
    plt.savefig(full_path)
# 单轴

# 不同消融实验的运行时间分布图
def runT_density(pre,file_list,dataset):
    # with open(file1, "r") as csvfile:
    #     reader1 = csv.reader(csvfile)
    #     data1 = list(reader1)
    plt.cla()
    city=dataset.capitalize()
    runT=[]
    for file in file_list:
        file1=os.path.join(pre,f"{file}{dataset}.csv")
    # file2=os.path.join(pre,f"{file2}{dataset}.csv")
    # file3=os.path.join(pre,f"{file2}{dataset}.csv")
        matrix = np.loadtxt(file1, delimiter=',')
    # matrix2 = np.loadtxt(file2, delimiter=',')
        runT.append(matrix)
    # inedxes=[]
    runT=[runT[0],runT[1],runT[2],runT[3]]
    # runT=[runT[2][:,0],runT[0],runT[1],runT[2][:,1]]
    # runT=runT.T
    labels=["DTC-ACO","w/o CRTE","w/o RPA","w/o EAS"]
    # alphas=[0.2,0.2,0.2,0.2,0.2]
    alphas=[0.25,0.25,0.25,0.25]
    # colors = ["#A586A6", "#F2c53d", "#D99789", "#91c8e8", "#cdd5c6", "#F2561d"]
    colors = ["#D99789","#91c8e8","#A586A6", "#F2c53d",   "#cdd5c6", "#F2561d"]
    # for i in range(len(data1)):
    #     for j in range(len(data1[0])):
    #         num = float(data1[i][j])
    #         if num < 1000:
    #             succuess[j] += 1
    #             fit_martix[j].append(float(data1[i][j]))
    #             ave_fit[j] += float(data1[i][j])
    for i in range(4):
        # if i == 4 or i == 6:
        #     sub += 1
        #     continue
        # else:
        #     item = i - sub
        sns.kdeplot(
            runT[i], fill=True, color=colors[i], alpha=alphas[i], label=labels[i]
        )
        # sns.histplot(runT[i], bins=30, stat='probability', color=colors[i], alpha=0.5, label='频数')

    plt.title(f"Density distribution on {city}", fontweight="bold", fontsize=18)
    plt.xlabel("RunT", fontweight="bold", fontsize=18)
    plt.ylabel("Density", fontweight="bold", fontsize=18)
    plt.legend()
    plt.grid(False)
    # if city =="Porto":
    #     y_right=550
    # else:
    y_right=400
    plt.xlim(0, y_right)
    # plt.ylim(0, 0.0251)
    # plt.yticks(ticks=plt.yticks()[0], labels=['' for _ in plt.yticks()[0]])
    plt.yticks([])
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6)) 
    ticks = plt.yticks()[0]
    plt.gca().set_yticklabels(['' for _ in ticks])
    # plt.yticks([1,2,3,4,5])
    # plt.yticklabels([1,2,3,4,5]), fontweight="bold", fontsize=18
    plt.savefig(f"{pre}/Density_RunT_variants_{city}.pdf")
    plt.show()
    # ave_fit = [
    #     round(ave_fit[i] / (succuess[i] + 0.001), 5) for i in range(len(ave_fit))
    # ]

    # succuess = [round(i / len(data1), 5) for i in succuess]
    # var_fit = [round(i, 5) for i in var_fit]
    # print(succuess)
    # print(ave_fit)
    # print(var_fit)
def aveage_read_var(data1, title):
    # with open(file1, "r") as csvfile:
    #     reader1 = csv.reader(csvfile)
    #     data1 = list(reader1)
    data1=data1.T
    # labels = ["Greedy", "FRVCRP", "SP", "RH", "TOA", "MultiCRPlanner"]
    labels=[30,40,50,60,70]
    alphas=[0.2,0.2,0.2,0.2,0.2]
    colors = ["#A586A6", "#F2c53d", "#D99789", "#91c8e8", "#cdd5c6", "#F2561d"]
    ave_fit = [0 for i in range(len(data1[0]))]
    var_fit = []
    fit_martix = [[] for _ in range(len(data1[0]))]
    succuess = [0 for i in range(len(data1[0]))]
    # for i in range(len(data1)):
    #     for j in range(len(data1[0])):
    #         num = float(data1[i][j])
    #         if num < 1000:
    #             succuess[j] += 1
    #             fit_martix[j].append(float(data1[i][j]))
    #             ave_fit[j] += float(data1[i][j])
    sub = 0
    for i in range(5):
        # if i == 4 or i == 6:
        #     sub += 1
        #     continue
        # else:
        #     item = i - sub
        sns.kdeplot(
            y=data1[i], fill=True, color=colors[i], alpha=alphas[i], label=labels[i]
        )
    plt.title(f"Density Plot of Comparsion result on {title}")
    plt.ylabel("Rank")
    plt.xlabel("Density")
    plt.legend()
    plt.grid(False)
    plt.ylim(1, 5)
    plt.yticks([1,2,3,4,5])
    # plt.yticklabels([1,2,3,4,5])
    plt.savefig(f"Density__{title}.pdf")
    plt.show()
    ave_fit = [
        round(ave_fit[i] / (succuess[i] + 0.001), 5) for i in range(len(ave_fit))
    ]

    succuess = [round(i / len(data1), 5) for i in succuess]
    var_fit = [round(i, 5) for i in var_fit]
    # print(succuess)
    # print(ave_fit)
    print(var_fit)
# aveage_read_var("./code/SCO-MulT-TTE/Draw/data/ants_num/Fitn_optimizers.csv","Fitn")
def RunT_fitn_single(pre,mid,name1,name2):
    # x=np.arange(0.2, 1, 0.2).tolist()
    x=[30,40,50,60,70]
    # x=[4,6,8,10,12]
    folder_path=os.path.join(pre,mid)
    data1,y1,std1=rank_var_para(os.path.join(folder_path,name1))
    data2,y2,std2=rank_var_para(os.path.join(folder_path,name2))
    # violin_mutli_draw(data1,"Fitn",x)
    # violin_mutli_draw(data2,"RunT",x)
    Q1_1 = np.percentile(data1, 20, axis=1)  # 第一四分位数
    Q3_1 = np.percentile(data1, 80, axis=1)  # 第三四分位数
    Q1_2 = np.percentile(data2, 20, axis=1)  # 第一四分位数
    Q3_2 = np.percentile(data2, 80, axis=1)  # 第三四分位数
    fig, ax1 = plt.subplots(figsize=(10, 9))
    # 第一个指标
    ax1.set_xlabel(r'The number of ants $N$',fontname="Times New Roman", fontweight="bold", fontsize=16)
    # ax1.set_xlabel(r'The select probablity $\lambda$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.set_ylabel('Rank',fontname="Times New Roman", fontweight="bold", fontsize=16)
    ax1.plot(x, y1, color='tab:blue', label='Fitn',marker="s")
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.fill_between(x, Q1_1, Q3_1, color='tab:blue', alpha=0.2)
    ax1.set_ylim(0.5,len(x)+1)
    ax1.set_yticks(np.arange(1, len(x)+1))
    ax1.set_yticklabels(np.arange(1, len(x)+1))
   
    # ax1.set_yticklabels(np.arange(1, len(x) + 1))
    # 创建第二个 Y 轴
    
    # ax2.set_ylabel('RunT', color='tab:orange',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.plot(x, y2, color='tab:orange', label='RunT', marker='v',markersize=10)
    # ax1.tick_params(axis='y', labelcolor='tab:orange')
    ax1.fill_between(x, Q1_2, Q3_2, color='tab:orange', alpha=0.2)
    fig.tight_layout()  # 防止标签重叠
    plt.legend()
    # plt.title('Fitn-RunT')
    # ax1.set_ylim(0,len(x))
    ax1.set_xticks([round(i,2) for i in x])
    # ax1.set_yticks(np.arange(1, len(x) + 1))
    # ax1.set_yticklabels(np.arange(1, len(x) + 1))
    # ax2.set_xticklabels([round(i,2) for i in x])
    ax1.set_xticklabels(x)

    full_path = os.path.join(folder_path, f"{mid}.pdf")
    plt.savefig(full_path)
# RunT_fitn("./code/SCO-MulT-TTE/Draw/data/q0","Fitn_optimizers.csv","runT_optimizers.csv")
# RunT_fitn("./code/SCO-MulT-TTE/Draw/data/early_stop","Fitn_optimizers.csv","runT_optimizers.csv")
# RunT_fitn_single("./code/SCO-MulT-TTE/Draw/data","ants_num","Fitn_optimizers.csv","runT_optimizers.csv")
# 针对不同参数调查的Rank曲线和FitnFitn密度分布混合图
def RunT_fitn_density(pre,mid,name1,name2,x):
    folder_path=os.path.join(pre,mid)
    data1,y1,std1=rank_var_para(os.path.join(folder_path,name1))
    aveage_read_var(data1, "rank_Fitn")
    data2,y2,std2=rank_var_para(os.path.join(folder_path,name2))
    # violin_mutli_draw(data1,"Fitn",x)
    # violin_mutli_draw(data2,"RunT",x)
    Q1_1 = np.percentile(data1, 15, axis=1)  # 第一四分位数
    Q3_1 = np.percentile(data1, 85, axis=1)  # 第三四分位数
    Q1_2 = np.percentile(data2, 15, axis=1)  # 第一四分位数
    Q3_2 = np.percentile(data2, 85, axis=1)  # 第三四分位数
    fig, ax1 = plt.subplots(figsize=(6.4, 6))
    data=data1
    # labels = ["Greedy", "FRVCRP", "SP", "RH", "TOA", "MultiCRPlanner"]
    # labels=[30,40,50,60,70]
    labels=[0.1,0.3,0.5,0.7,0.9]
    # labels=[4,6,8,10,12]
    # alphas=[0.5,0.5,0.5,0.5,0.5]
    alphas=[0.27,0.27,0.27,0.27,0.27]
    colors = [ "#F2c53d", "#A586A6","#D99789", "#F2561d", "#91c8e8", "#cdd5c6"]
    # for i in range(len(data1)):
    #     for j in range(len(data1[0])):
    #         num = float(data1[i][j])
    #         if num < 1000:
    #             succuess[j] += 1
    #             fit_martix[j].append(float(data1[i][j]))
    #             ave_fit[j] += float(data1[i][j])
    sub = 0
    ax2 = ax1.twiny()
    patterns=['x', '\\', '|', '-','+']
    # ax2.set_xlabel('Density',fontname="Times New Roman", fontweight="bold", fontsize=20)
    ax2.set_xlabel('Density', fontweight="bold", fontsize=20)
    plt.xticks(fontsize=18)
    for i in range(len(x)):
        # if i == 4 or i == 6:
        #     sub += 1
        #     continue
        # else:
        #     item = i - sub
        # sns.kdeplot(
        #     y=data[i], fill=True, color=colors[i], alpha=alphas[i], label=labels[i],ax=ax2
        # )
        # kde = sns.kdeplot(y=data[i], fill=True, color=colors[i], alpha=alphas[i], label=x[i], ax=ax2, bw_adjust=1.8)
        kde = sns.kdeplot(y=data[i], fill=True, color=colors[i], alpha=alphas[i], label=x[i], ax=ax2)
        # 获取密度填充区域的路径
        # x = kde.get_lines()[0].get_xdata()
        # y = kde.get_lines()[0].get_ydata()

        # # 填充区域
        # ax2.fill_between(x, y, color=colors[i], alpha=alphas[i], hatch=patterns[i], label=labels[i])
        # # for path in kde.collections:
        #     # 使用 hatch 属性添加图案
        #     path.set_hatch(patterns[i]) 
        #     path.set_color(colors[i]) # 可以选择不同的图案，如 '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
    # 第一个指标
    # ax1.set_xlabel(r'The number ofPheromone evaporation rate ants $N$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    # ax1.set_xlabel(r'The early-stop threshold of ants $\xi$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    # ax1.set_xlabel(r'The select probablity $\lambda$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.set_xlabel(r'The Pheromone evaporation rate $\rho$', fontweight="bold", fontsize=18)
    # ax1.set_xlabel(r'The Pheromone evaporation rate $\rho$',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.set_ylabel('Rank', fontweight="bold", fontsize=20)
    # ax1.set_ylabel('Rank',fontname="Times New Roman", fontweight="bold", fontsize=20)
    ax1.plot(x, y1, color='tab:blue', label='Fitn',linewidth=3)
    for i in range(len(x)):
        ax1.plot(x[i], y1[i], color='tab:blue', marker='s',markersize=10, 
                markerfacecolor=colors[i % len(colors)], 
                markeredgecolor=colors[i % len(colors)])  # 使用循环选择颜色
    
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.fill_between(x, Q1_1, Q3_1, color='tab:blue', alpha=0.2)
    ax1.set_ylim(0,len(x)+2.5)
    ax1.set_yticks(np.arange(1, len(x)+1))
    ax1.set_yticklabels(np.arange(1, len(x)+1),fontsize=18)
   
    # ax1.set_yticklabels(np.arange(1, len(x) + 1))
    # 创建第二个 Y 轴
    
    # ax2.set_ylabel('RunT', color='tab:orange',fontname="Times New Roman", fontweight="bold", fontsize=18)
    ax1.plot(x, y2, color='tab:red', label='RunT',linewidth=3)
    for i in range(len(x)):
        ax1.plot(x[i], y2[i], color='tab:red', marker='v',markersize=10,
                markerfacecolor=colors[i % len(colors)], 
                markeredgecolor=colors[i % len(colors)])  # 使用循环选择颜色
 
    # ax1.tick_params(axis='y', labelcolor='tab:orange')
    # ax1.fill_between(x, Q1_2, Q3_2, color='tab:orange', alpha=0.2)
    fig.tight_layout()  # 防止标签重叠
    # plt.legend()
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(data), frameon=False)
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.65, 1),ncol=len(data), frameon=False,fontsize=18)
    ax2.legend(ncol=3,fontsize=18)
    # plt.title('Fitn-RunT')
    # ax1.set_ylim(0,len(x))
    ax1.set_xticks([round(i,2) for i in x])
    # ax1.set_yticks(np.arange(1, len(x) + 1))
    # ax1.set_yticklabels(np.arange(1, len(x) + 1))
    # ax2.set_xticklabels([round(i,2) for i in x])
    ax1.set_xticklabels(x,fontsize=18)

    full_path = os.path.join(pre, f"density_{mid}.pdf")
    # full_path = os.path.join(folder_path, f"density_{mid}.pdf")
    # plt.show()
    plt.savefig(full_path)
    plt.show()
def extract_single_runT(file):
    data=np.loadtxt(file, delimiter=',')
    lenn=int(data.shape[0]/20)*20
    runT_true=data[:lenn]
    reshaped_array = runT_true.reshape(-1, 20)  # -1 表示自动计算行数

    # 沿行求均值
    mean_values = np.mean(reshaped_array, axis=1)
    print(mean_values)
def count_com(file):
    data=np.loadtxt(file, delimiter=',')
    mean_values = np.mean(data, axis=0)
    print([(i/106.72724977)*100 for i in mean_values])
if __name__ == '__main__':
    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data","q0","Fitn_optimizer.csv","runT_optimizers.csv")
    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data","early_stop","Fitn_optimizers.csv","runT_optimizers.csv")
    labels=[0.1,0.3,0.5,0.7,0.9]

    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data","Pho","multi_Fitn_chengdu.csv","multi_runT_chengdu.csv",labels)
    # get_oinf("./code/SCO-MulT-TTE/Draw/data1/com/runT_optimizers2.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/Elite/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/Elite/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS1/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS_one/multi_Fitn_chengdu.csv")
    
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS/multi_Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS1/multi_Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_EAS_one/multi_Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/SL-SCO/RunT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/SL-SCO/Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/SL-SCO/Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_runT_porto.csv")
    # get_min("./code/SCO-MulT-TTE/Draw/data/update_repair/old/E_Conver_diversity_False_porto.csv")
    # get_min("./code/SCO-MulT-TTE/Draw/data/Elite/E_Conver_diversity_True_porto.csv")
    # multi_runT_chengdu.csv
    # get_mean("./code/SCO-MulT-TTE/Draw/data/surr/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers_other_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers_laco_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers_other_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/Fitn_optimizers_False_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_Fitn_c.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com/Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_SL_SCO/Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_SL_SCO(max,-3)/Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/Ant_RPA/multi_Fitn_chengdu.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/Ant_RPA/multi_Fitn_chengdu.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/Ant_RPA/multi_Fitn_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/CP_RPA/multi_Fitn_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_SL_SCO(max,-3)/RunT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/Ant_RPA/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/CP_RPA/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_test/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/abl_test(max-3)/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/Ant_RPA/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/update_repair/old/multi_runT_porto.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/CP_RPA/multi_runT_porto.csv")
    
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com-SL-SCO/Fitn_chengdu.csv")
    # get_oinf("./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_True_chengdu.csv")
    # get_oinf("./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_True.csv")
    # get_oinf("./code/SCO-MulT-TTE/Draw/data/com/Fitn_optimizers_other_chengdu.csv")
    # get_oinf("./code/SCO-MulT-TTE/Draw/data/surr/multi_runT_True_chengdu.csv")
    # rank_var_para("./code/SCO-MulT-TTE/Draw/data/KCM5/multi_Fitn.csv")
    # com_two1([["./code/SCO-MulT-TTE/Draw/data/surr/E_Conver_diversity_False_chengdu.csv","./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_True_chengdu.csv"],["./code/SCO-MulT-TTE/Draw/data/surr/E_Conver_diversity_False_porto.csv","./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_True.csv"]])
    # get_oinf("./code/SCO-MulT-TTE/Draw/data1/com/runT_optimizers_porto.csv")
    # get_oinf("./code/SCO-MulT-TTE/Draw/data1/com/Fitn_optimizers_porto.csv")
    # com_all("./code/SCO-MulT-TTE/Draw/data/com/Fitn_optimizers_other_porto.csv","./code/SCO-MulT-TTE/Draw/data/com/E_Conver_diversity_False_porto.csv")
    # # com_all("./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers_other_porto.csv","./code/SCO-MulT-TTE/Draw/data/com/runT_optimizers_porto.csv",False)
    # com_all("./code/SCO-MulT-TTE/Draw/data/com/chengdu_ld/Fitn_optimizers_other_chengdu.csv","./code/SCO-MulT-TTE/Draw/data/com/chengdu_ld/E_Conver_diversity_False_chengdu.csv")
    # com_all("./code/SCO-MulT-TTE/Draw/data/com/chengdu_ld/runT_optimizers_other_chengdu.csv","./code/SCO-MulT-TTE/Draw/data/com/chengdu_ld/runT_optimizers_chengdu.csv",False)
    # get_mean("./code/SCO-MulT-TTE/Draw/data/surr/multi_runT_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/com/Succ_optimizers_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/surr/multi_runT_porto.csv")
    # get_succ_o_inf("./code/SCO-MulT-TTE/Draw/data/Repair/multi_Fitn_chengdu.csv")
    # get_succ_o_inf("./code/SCO-MulT-TTE/Draw/data/Elite/multi_Fitn_chengdu.csv")
    # get_succ_o_inf("./code/SCO-MulT-TTE/Draw/data/com/Fitn_optimizers_other_chengdu.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_porto.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/surr/multi_Fitn_chengdu.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/update_repair/multi_Fitn_porto.csv")
    # get_oinf_single("./code/SCO-MulT-TTE/Draw/data/Elite/multi_runT_chengdu.csv")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data/Repair", ["E_Conver_diversity_False_chengdu.csv"],"Chengdu")
    mean_with_curve("./Draw/data/Elite_Time/porto", ["merged_output_True.csv","merged_output_False.csv"],"Porto","Time")
    mean_with_curve("./Draw/data/Elite_Cost/chengdu", ["merged_output_True.csv","merged_output_False.csv"],"Chengdu","Expense")
    mean_with_curve("./Draw/data/Elite_Cost/porto", ["merged_output_True.csv","merged_output_False.csv"],"Porto","Expense")
    mean_with_curve("./Draw/data/Elite_Time/chengdu", ["merged_output_True.csv","merged_output_False.csv"],"Chengdu","Time")
    # mean_with_curve("./code/ACO-MulT-TTE/Draw/data/Elite_Cost/chengdu", ["merged_output_False.csv","merged_output_True.csv"],"Chengdu")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data/Elite", ["./code/SCO-MulT-TTE/Draw/data/abl_SL_SCO/E_Conver_diversity_False_chengdu.csv","./code/SCO-MulT-TTE/Draw/data/Elite/Conver_diversity_False_chengdu.csv"],"Chengdu")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data/Elite", ["./code/SCO-MulT-TTE/Draw/data1/abl_SL_SCO/E_Conver_diversity_False_porto.csv","./code/SCO-MulT-TTE/Draw/data1/Elite/Conver_diversity_False_porto.csv"],"Porto")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data1/Elite", ["E_Conver_diversity_True_porto.csv","Conver_diversity_False_porto.csv"],"Porto")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data/Repair", ["E_Conver_diversity_False_porto.csv"],"Porto")
    # mean_with_curve("./code/SCO-MulT-TTE/Draw/data1/Repair", ["E_Conver_diversity_True_porto.csv","Conver_diversity_True_porto.csv"])
    # draw_feasible_ratio("./code/SCO-MulT-TTE/Draw/data/Repair", ["Conver_diversity_False_chengdu.csv","Conver_diversity_False_porto.csv"])
    # draw_feasible_ratio("./code/SCO-MulT-TTE/Draw/data1/Repair", ["E_Conver_diversity_True_chengdu.csv","E_Conver_diversity_True_porto.csv"])
    # draw_iters("./code/SCO-MulT-TTE/Draw/data1/Repair", [["Conver_diversity_False_chengdu.csv","E_Conver_diversity_True_chengdu.csv"],["Conver_diversity_False_porto.csv","E_Conver_diversity_True_porto.csv"]])
    # runT_density("./code/ACO-MulT-TTE/Draw/data",["abl_SL_ACO/RunT_","surr//multi_runT_","Repair/multi_runT_","Elite/multi_runT_"],"chengdu")
    # runT_density("./code/ACO-MulT-TTE/Draw/data",["abl_test/multi_runT_","surr//multi_runT_","Repair/multi_runT_","Elite/multi_runT_"],"porto")
    # runT_density("./code/ACO-MulT-TTE/Draw/data1",["surr//multi_runT_","Repair/multi_runT_","Elite/multi_runT_"],"porto")
    # runT_density("./code/SCO-MulT-TTE/Draw/data","Repair/multi_runT_","Elite/multi_runT_","chengdu")
    # extract_single_runT("./code/SCO-MulT-TTE/Draw/data/Repair/multi_runT_chengdu.csv")
    # count_com("./code/SCO-MulT-TTE/Draw/data/modeule_time_rec_chengdu_r_False_e_True_s_True_15.csv")
    # draw_bar2("./code/ACO-MulT-TTE/Draw/data", ["modeule_time_rec_chengdu.csv","modeule_time_rec_porto.csv"])
    # draw_barwoRPA("./code/SCO-MulT-TTE/Draw/data", ["modeule_time_rec_porto_r_True_e_True_s_False_6.csv","modeule_time_rec_porto_r_True_e_True_s_False_13.csv"])
    # draw_barwoRPA("./code/SCO-MulT-TTE/Draw/data", ["modeule_time_rec_chengdu_r_True_e_True_s_False_6.csv","modeule_time_rec_chengdu_r_True_e_True_s_False_13.csv"])
    # com_two1("./code/SCO-MulT-TTE/Draw/data/Repair/Conver_diversity_True_chengdu.csv","./code/SCO-MulT-TTE/Draw/data/Repair/E_Conver_diversity_True_chengdu.csv")
    # get_mean("./code/SCO-MulT-TTE/Draw/data/q0/runT_optimizers.csv")
    
    # x=[30,40,50,60,70]
    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data1","ants_num","Fitn_optimizers.csv","runT_optimizers.csv",x)
    # x=[4,6,8,10,12]
    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data1","early_stop","Fitn_optimizers.csv","runT_optimizers.csv",x)
    # x=np.arange(0.2, 1, 0.2).tolist()
    # x=[round(i,2) for i in x]
    # RunT_fitn_density("./code/SCO-MulT-TTE/Draw/data1","q0","Fitn_optimizer.csv","runT_optimizers.csv",x)