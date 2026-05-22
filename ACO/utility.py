import math
import networkx as nx
from torch.utils.data import Dataset
import csv
import numpy as np
# 定义充电站用户需求
class CS:
    # n为1*3的向量
    def __init__(self, n, f, P, p):
        self.n = n
        self.f = f
        self.P = P
        self.p = p
        self.O = [0, 0, 0, 0]
        for i in range(len(n)):
            if n[i] > 0:
                self.O[i] = i
            else:
                self.O[i] = 0

class eval_unit:
    def __init__(self, path, index, t):
        self.path=path
        self.index=index
        self.t=t

class EV:
    def __init__(self, args):
        self.M = args.Mass
        self.O = args.O
        self.A = args.A
        self.P_a = args.P_a
        self.Q = args.Q
        self.Cd = args.C_d
        self.Cr = args.C_r
        self.eff1 = args.eff1  # 0.997
        self.eff2 = args.eff2  # 0.808
        self.eff3 = args.eff3  # 0.153
        self.eta_add = args.eta_add
        self.eta_re = args.eta_re

    def nolinear_changing1(self, alpha, init, P):
        end = init + alpha
        time = 0
        if end > 0.9:
            time += ((end - 0.9) * self.Q) / (P * self.eff3)
            end = 0.9
        if end > 0.8:
            time += ((end - 0.8) * self.Q) / (P * self.eff2)
            end = 0.8
        time += (end * self.Q) / (P * self.eff1)
        return time
    def nolinear_charging(self,alpha,init,P):
        t=self.get_time(init+alpha)-self.get_time(init)
        return t*7/P
    def get_time(self,SoC):
        if SoC<0.6:
            return 250*SoC
        elif SoC<0.925:
            return 487.329*SoC-142.3974
        else:
            return 932.836*SoC-554.491375 
class Re:
    def __init__(
        self,
      args
    ):
        # time:将一天划分为1440个时隙（0-1439),week:(0-6),day:(0-365)
        self.s = args.s
        self.des = args.des
        self.g = 9.81
        self.pho = args.ph
        self.aux = args.aux
        self.t0 = args.t0
        self.t_l = args.t_l
        self.W = args.W
        self.init = args.init
        self.min = args.minn
        self.max = args.maxx
        # 计划出行时间的日期，以及周几
        # 待调查的参数
        self.p0=0.5
        self.date=args.date
        self.week=args.week
        self.v=args.v

    def init_paths(self, path, G_T, G, EV):
        t_min = 0
        E_min = 0
        alpha = self.init
        td_time = []
        t = self.t0
        for j in range(0, len(path) - 1):
            temp = path[j]
            if G_T.nodes[temp]["iscs"] == 1:
                alpha = 100
                t_min += 0.15
            if G_T.has_edge(str(temp), str(path[j + 1])):
                tr, com_E = self.com_E(EV, G_T, G, str(temp), str(path[j + 1]))
                E_min += com_E
                alpha = alpha - com_E
                if alpha > 100:
                    alpha = 100
                td_time.append(tr)
                t = (t + tr) % 24
                if alpha < 0:
                    self.flag = False
                t_min += tr
            else:
                pass
        self.t_min = abs(t_min)
        self.cost_min = abs(E_min)
        self.t_hover = self.init / 10
        self.cost_hover = EV.Q + E_min

    def par_min(self, G, EV):
        cost_min = 0
        t_min = 0
        E_min = 0
        path = nx.shortest_path(G, source=self.s, target=self.des, weight="length")
        for i in range(1, len(path)):
            e = G[path[i - 1]][path[i]]
            P_t = (
                    EV.M
                    * (
                        self.g
                        * (
                            math.sin(math.radians(float((e["angle"]))))
                            + EV.Cr * math.cos((math.radians((float(e["angle"])))))
                        )
                    )
                    + 0.5
                    * EV.Cd
                    * EV.A
                    * (self.pho )
                    * self.v
                    * self.v
                ) * self.v
            t_l=e["length"] / self.v
            if P_t > 0:
                com_E = (P_t / EV.eta_add + self.aux * EV.P_a) * t_l 
            else:
                com_E = (P_t / EV.eta_re + self.aux * EV.P_a) * t_l 
            E_min += com_E
            cost_min += com_E 
            t_min += t_l
        self.t_min = t_min/60
        self.cost_min = cost_min/3600000
        # self.min_alpha=max(E_min/EV.Q/1000,1)
        self.min_alpha=self.cost_min

    def com_E(self, EV, G_T, G_B, s, d,v=None):
        if v is None:
            v=self.v
        t0 = 0
        if G_T.has_edge(s, d):
            attr = G_T.get_edge_data(s, d)
            E_Set = []
            total_E=0
            for k in range(len(attr["path"]) - 1):
                E_Set.append(G_B.get_edge_data(attr["path"][k], attr["path"][k + 1]))
            for e in E_Set:
                # 加速度的单位是km/h2
                # 将m/s换算成km/h
                t_l = float(e.get("length")) / v
                t0 += t_l
                # EV.M:kg g:m/s2 v:m/s A:m2; pho:kg/m3
                P_t = (
                    EV.M
                    * (
                        self.g
                        * (
                            math.sin(math.radians(float((e["angle"]))))
                            + EV.Cr * math.cos((math.radians((float(e["angle"])))))
                        )
                    )
                    + 0.5
                    * EV.Cd
                    * EV.A
                    * (self.pho * (273.15 / (e["T"] + 273.15)))
                    * v
                    * v
                ) * v
                if P_t > 0:
                    com_E = (P_t / EV.eta_add + self.aux * EV.P_a) * t_l / 1000
                else:
                    com_E = (P_t / EV.eta_re + self.aux * EV.P_a) * t_l / 1000
                total_E+=com_E
            return t0*60, total_E/3600000
        return 10, 10

    def com_f(self, time, cost, penalty):
        if isinstance(time,list) and isinstance(cost,list):
            fit=[]
            time_list=[]
            cost_list=[]
            for t,e,p in zip(time,cost,penalty):
                n_f,n_t,n_e=self.com_f(t,e,p)
                fit.append(n_f)
                time_list.append(n_t)
                cost_list.append(n_e)
            return fit,time_list,cost_list
        else:
            nor_t= time / self.t_min
            nor_c=  cost / self.cost_min
            fit = self.W[0] * time / self.t_min + self.W[1] * cost / self.cost_min+penalty
            # if fit == 0.5:
            #     print("time:" + str(time))
            #     print("cost:" + str(cost))
            return fit,nor_t,nor_c
class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)
def floor_dec(num):
    return (num // 0.05) * 0.05
def write(filename,data,path='./code/ACO-MulT-TTE/Draw/data/'):
    with open(path+filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
            # 将列表写入文件的一行
        writer.writerow(data)
def multi_writer(filename,datas):
    for data in datas:
        with open('./code/ACO-MulT-TTE/Draw/data/'+filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
                # 将列表写入文件的一行
            writer.writerow(data)
    
def softmax(z):
    # 计算每个元素的指数
    e_z = np.exp(z - np.max(z))  # 减去最大值以提高数值稳定性
    return e_z / e_z.sum()

class Time_f:
    def __init__(self, points=[]):
        # points should be a list of tuples (x, y) sorted by x
        self.points = sorted(points)

    def add_point(self, points):
        if not isinstance(points, list):
            points = [points]
            k=0
        for point in points:
            x, y = point
            # Check if a point with the same x already exists
            for i in range(len(self.points)):
                if self.points[i][0] == x:
                    # Replace the existing point
                    self.points[i] = (x, y)
                    break
                elif self.points[i][0]>x:
                    self.points.insert(i, (x,y))
                    if self.points[i+1][0]-x>1:
                        self.points.insert(i+1,points[-1])
                    if x-self.points[i-1][0]>1:
                        self.points.insert(i,points[0])
                    break
            else:
                # If no replacement was made, append the new point
                self.points.append(point)
                # Re-sort the points after addition
                self.points = sorted(self.points)

    def get_x_for_y(self, y):
        # Iterate through the segments and find the corresponding x for the given y
        for i in range(len(self.points) - 1):
            (x0, y0) = self.points[i]
            (x1, y1) = self.points[i + 1]
            
            if y0 <= y <= y1 or y1 <= y <= y0:
                # Linear interpolation to find x for the given y
                if y1 != y0:  # Avoid division by zero
                    slope = (x1 - x0) / (y1 - y0)
                    x = x0 + slope * (y - y0)
                    return x
        return None  # Return None if y is out of range

    def get_y_for_x(self, x):
        # Iterate through the segments and find the corresponding y for the given x
        for i in range(len(self.points) - 1):
            (x0, y0) = self.points[i]
            (x1, y1) = self.points[i + 1]
            
            if x0 <= x <= x1:
                # Linear interpolation to find y for the given x
                if x1 != x0:  # Avoid division by zero
                    slope = (y1 - y0) / (x1 - x0)
                    y = y0 + slope * (x - x0)
                    return y
        return None  # Return None if x is out of range

    def __str__(self):
        return str(self.points)