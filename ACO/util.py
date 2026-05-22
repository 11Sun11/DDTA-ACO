'''
Author: Xin-Can Wang
Date: 2024-10-21 14:29:37
LastEditors: Xin-Can Wang
LastEditTime: 2025-02-08 21:50:03
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/ACO/util.py
'''
import re
import numpy as np
import csv
import os
import random
import numpy as np
from scipy.stats import truncnorm
from ACO.utility import *
def get_ex(args,simple=False):
    if simple is False:
        path=f"{args.info_path}ins_{args.data}.csv"
    else:
        path=f"{args.info_path}ins_{args.data}_simple.csv"
    with open(path,"r") as f:
        ins=[]
        reader=csv.reader(f)
        for row in reader:
            row=[float(r) for r in row]
            ins.append(row)
            if len(ins)>=65:
                break
    args.ins=ins
def set_ins(args,i):
    ins=args.ins[i]
    args.s,args.des=int(ins[0]),int(ins[1])
    args.req.s,args.req.des=int(ins[0]),int(ins[1])
    args.req.t_min=float(ins[5])
    args.req.cost_min=float(ins[6])
    args.req.min_alpha=args.req.cost_min
    # args.req.min_alpha=args.t_e[sd[0]][sd[1]][1]
    # args.req.s=args.net.edge_list[args.s][0]
    # args.req.des=args.net.edge_list[args.des][1]
    # if args.net.ki(args)==-1 or (not nx.has_path(args.net.G_B, args.req.s, args.req.des)):
    #      continue
    # args.req.par_min(args.net.G_B,args.vehicle)
    args.target_time=0
    args.week=int(ins[3])
    args.day=int(ins[4])
    args.req.t0=args.time=int(ins[2])
def find_all_paths(graph, source, target):
    paths = []
    stack = [(source, [source])]

    while stack:
        node, path = stack.pop()

        if node == target:
            paths.append(path)

        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
            else:
                # 处理环路情况
                if neighbor == target:
                    paths.append(path + [neighbor])
                    if len(paths)>3:
                        return paths
    return paths

def truncated_gaussian(mean, std_dev, size):
    data = []
    pro=[0 for _ in range(20)]
    while len(data) < size:
        sample = np.random.normal(mean, std_dev)
        if 0 <= sample <1:
            data.append(sample)
            pro[int(floor_dec(sample)/0.05)]+=1
    pro=[item/sum(pro) for item in pro]
    return pro
def tru(sample):
    pro=[0 for _ in range(20)]
    for data in sample:
        pro[int(floor_dec(data)/0.05)]+=1
    pro=[item/sum(pro) for item in pro]
    return pro
# 随机生成指定长度的整数
def generate_random_integers(length, count):
    if length <= 0 or count <= 0:
        raise ValueError("Length and count must be positive integers")
    
    lower_bound = 10**(length - 1)
    upper_bound = 10**length - 1
    
    # 生成多个随机整数
    return np.random.randint(lower_bound, upper_bound + 1, size=count)

# 生成 5 个 6 位的随机整数
def gen_instances(args):
    ins=[]
    phase="test"
    data = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
    for _, l in enumerate(data):
        # s,d,通行时间,日期
        ins.append(np.asarray([l[1][0],l[1][-1],l[-1],l[2],l[3],l[4]]))
    csv_file_name = "ins.csv"
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头（可选）
        writer.writerow(["S", "D", "Time","Weekday","Day","clock"])
        # 写入每一对 (source, destination, additional info)
        for pair in ins:
            writer.writerow(pair)
    return ins

def get_instance(args,choice="sd"):
    if choice=="sd":
        csv_file_name = f"{args.info_path}ran_ins_{args.data}.csv"
    elif choice=="simple":
        csv_file_name = f"{args.info_path}ran_ins_{args.data}_simple.csv"
    else:
        csv_file_name = f"{args.info_path}ins.csv"
    with open(csv_file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        # 跳过表头（如果有的话）
        header = next(reader)
        # 初始化列表以存储读取的数据
        ins = []
        # 读取每一行
        for row in reader:
            ins.append(row)  # 每一行都是一个列表 
    return ins
def regen_instance():
    # if choice=="sd":
    #     csv_file_name = args.ins_path
    # else:
    csv_file_name = './code/ACO-MulT-TTE/Draw/data/ins_with_t.csv'
    with open(csv_file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        # 跳过表头（如果有的话）
        header = next(reader)
        # 初始化列表以存储读取的数据
        ins = dict()
        # 读取每一行
        for row in reader:
            if row[0] not in ins:
                ins[row[0]]=dict()
            if row[1] not in ins[row[0]]:
                ins[row[0]][row[1]]=[[],[]]
            ins[row[0]][row[1]][0].append(float(row[2]))
            numbers = re.findall(r'\d+', row[3])
            ins[row[0]][row[1]][1].append(float(numbers[0]))
        new_ins=[]
        for nodei in ins.keys():
            for nodej in ins[nodei].keys():
                mean_time=sum(ins[nodei][nodej][0]) / len(ins[nodei][nodej][0])
                mean_energy=sum(ins[nodei][nodej][1]) / len(ins[nodei][nodej][1])
                new_ins.append([nodei,nodej,mean_time,mean_energy])
                write("new_ins_time_energy.csv",[nodei,nodej,mean_time,mean_energy])
            
    return ins
# def reget_instance():
#     # if choice=="sd":
#     #     csv_file_name = args.ins_path
#     # else:
#     csv_file_name = './code/ACO-MulT-TTE/Draw/data/new_ins_time_energy.csv'
#     with open(csv_file_name, mode='r', newline='') as file:
#         reader = csv.reader(file)
#         # 跳过表头（如果有的话）
#         header = next(reader)
#         # 初始化列表以存储读取的数据
#         ins = dict()
#         # 读取每一行
#         for row in reader:
#             ins[row[0]]=dict()
#             ins[row[0]][row[1]]=[]
#             ins[row[0]][row[1]].append(float(row[2]))
#             ins[row[0]][row[1]].append(float(row[3]))
#     return ins
def floor_dec(num):
    return (num // 0.05) * 0.05

def truncated_gauss(mu, sigma):
    while True:
        value = random.gauss(mu, sigma)
        if 0 <= value <= 1:
            return floor_dec(value)

def com_scaler(file_path,out_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # 跳过表头（如果有的话）
        # 初始化列表以存储读取的数据
        ins = dict()
        # 读取每一行
        for row in reader:
            if row[0] in ins.keys():
                if row[1] in ins[row[0]].keys():
                    ins[row[0]][row[1]][0].append(row[2])
                    ins[row[0]][row[1]][1].append(row[3])
                else:
                    ins[row[0]][row[1]][0].append(row[2])
                    ins[row[0]][row[1]][1].append(row[3])
            else:
                ins[row[0]]=dict()
                ins[row[0]][row[1]]=[[row[2]],[row[3]]]
    with open(out_path, mode='w', newline='') as writer:
        writer=csv.writer(file)
        for u in ins.keys():
            for v in ins[u].keys():
                writer.writerow([u,v,sum(ins[u][v])/len(ins[u][v])])
    return

def normalize_probabilities(probabilities):
        """归一化概率列表，使其总和为1"""
        total = sum(probabilities)  # 计算总和
        if (total-0)<=0.000001:  # 如果总和为 0，返回原始列表
            return [1/len(probabilities) for _ in range(len(probabilities))]
        return [p / total for p in probabilities]  # 归一化