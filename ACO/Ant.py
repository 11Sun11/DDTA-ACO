# 充电选择
import math
import numpy as np
from ACO.Net import *
import networkx as nx
from collections import deque
import Levenshtein
import copy
from ACO.DDTA_ACO import*

class CO:
    def __init__(self, opt, alpha,pre=None,index=-1):
        if alpha<0:
            print("",end="")
        # 充电操作在当前路径中的索引位置
        self.index=index
        self.opt = opt
        self.alpha = alpha
        self.pre=pre
    def __deepcopy__(self,memo):
        return CO(self.opt,self.alpha,self.pre,self.index)


class Ant:
    def __init__(self, init,s,d):
        self.s = s
        self.d = d
        self.cur = s
        self.arrival_time = [0]
        self.co_index=[]
        self.path = [s]
        self.CO = []
        self.ori_path=[]
        self.alpha=init
    def reinit(self,ori_path,CP):
        self.ori_path=ori_path
        self.CO=CP
    def recover(self,net:Net):
        from ACO.Dy_ACOV4 import ACO_optimizer
        self.path=ACO_optimizer.Conversion(self.ori_path,net.node_to_index)
        self.update_co_index(net.G)
        


      # 移动到下一个节点处
    def __deepcopy__(self,memo):
        new_ant=Ant(self.alpha,self.s,self.d)
        new_ant.CO=copy.deepcopy(self.CO)
        new_ant.path=copy.deepcopy(self.path)
        new_ant.ori_path=copy.deepcopy(self.ori_path)  
        new_ant.alpha=self.alpha
        return new_ant
    def update_co_index(self,G):
        k=0
        for i,p in enumerate(self.ori_path):
            if G.nodes[p]["iscs"]==1:
                self.CO[k].index=i
                k+=1

    def move_to_next_index(self, next_index,co=None):
        if co is not None:
            co.index=len(self.path)
            self.CO.append(co)
        self.path.append(next_index)
        # 如果是cs节点，那么选择相应的充电技术
        return  
   
    def route_str(self):
        str_route=""
        for node in self.path[:-1]:
            str_node='{:06d}'.format(node)
            str_route+=str_node+"|"
        str_route+='{:06d}'.format(self.path[-1])
        return str_route
    def dis_ant(self,other_ant,G):
        str_route1=self.route_str()
        str_route2=other_ant.route_str()
        dis_r= Levenshtein.distance(str_route1, str_route2) / (min(len(str_route1), len(str_route2)) * 0.5)
        dis_a,dis_o=self.dis_CO(other_ant,G)
        dis=(dis_r+dis_o+dis_a)/3
        return dis
    def dis_CO(self,other_ant,G):
        dict1 = self.get_dictCharge(G)
        dict2 = other_ant.get_dictCharge(G)
        dis_o = 0
        dis_a = 0
        intersection = set(dict1.keys()) & set(dict2.keys())
        key_difference = set(dict1.keys()) | set(
            dict2.keys()
        ) 
        key_difference -= intersection
        for cs in intersection:
            if dict1[cs].alpha == 0 and dict2[cs].alpha == 0:
                continue
            else:
                dis_o += not dict1[cs].opt == dict2[cs].opt
            if dict2[cs].opt == dict1[cs].opt and (
                dict1[cs].opt == 0 or dict2[cs].opt == 3
            ):
                continue
            elif (dict2[cs].opt == 0 and dict1[cs].opt == 3) or (
                dict2[cs].opt == 3 and dict1[cs].opt == 0
            ):
                dis_a += 1
                continue
            dis_a += abs(dict2[cs].alpha - dict1[cs].alpha)
        for cs in key_difference:
            if cs in dict1.keys():
                if dict1[cs].opt == 0:
                    continue
                elif dict1[cs].opt == 3:
                    dis_a += 1
                elif dict1[cs].alpha != 0:
                    dis_o += 1
                    dis_a += dict1[cs].alpha
            if cs in dict2.keys():
                if dict2[cs].opt == 0:
                    continue
                elif dict2[cs].opt == 3:
                    dis_a += 1
                elif dict2[cs].alpha != 0:
                    dis_o += 1
                    dis_a += dict2[cs].alpha
        # dis_a /= min_len
        # dis_o /= min_len
        return dis_a, dis_o


    def get_dictCharge(self,G):
        dict1 = dict()
        q = 0
        for co in self.CO:
            temp = self.ori_path[co.index]
            if G.nodes[temp]["iscs"] == 1:
                dict1[temp] = self.CO[q]
                q += 1
        return dict1
def add_neigbor(self, node, max_dis):
    queue = deque([node, 0])
    visited = set()
    while queue:
        current_node, current_distance = queue.popleft()
        # 如果当前距离超过最大距离，停止扩展
        if current_distance > max_dis:
            continue
        # 标记当前节点为已访问
        visited.add(current_node)
        # 遍历当前节点的邻居
        for neighbor in self.G.neighbors(current_node):
            edge_weight = self.G[current_node][neighbor]["length"]
            if current_node not in self.subG:
                # 添加边到子图
                self.subG.add((current_node, neighbor))
                # 仅在邻居未被访问且距离未超过最大距离时，继续扩展
                if neighbor not in visited:
                    queue.append((neighbor, current_distance + edge_weight))


def extract_subG(self):
    self.subG = nx.Graph()
    max_dis = nx.shorest_path_length(self.G_B, self.s, self.d)
    max_dis /= 3
    paths = find_all_paths(self.s, self.d)
    for path in paths:
        for node in path:
            if node not in self.subG:
                self.subG.add_node(node)
                self.add_neigbor(self.subG, max_dis)


def drive_time(s, d):
    raise NotImplementedError


def select_next_index(self, ant):
    cur = self.cur
    node_to_visit = ant.node_to_visit
    tran_pho = self.pho[cur][node_to_visit]
    tran_pho = np.sum(self.pho[cur][node_to_visit])
    if np.random.rand() < self.q0:
        next_index = np.argmax(tran_pho)
        next_index = node_to_visit[next_index]
    else:
        next_index = self.stochastic_accept(node_to_visit, tran_pho)


@staticmethod
def stochastic_accept(index_to_visit, transition_prob):
    """
    轮盘赌
    :param index_to_visit: a list of N index (list or tuple)
    :param transition_prob:
    :return: selected index
    """
    # calculate N and max fitness value
    N = len(index_to_visit)
    # normalize
    sum_tran_prob = np.sum(transition_prob)
    norm_transition_prob = transition_prob / sum_tran_prob

    # select: O(1)
    while True:
        # randomly select an individual with uniform probability
        ind = int(N * np.random.rand())
        if np.random.rand() <= norm_transition_prob[ind]:
            return index_to_visit[ind]


# 计算顶层路网之间的消耗能量
def cal_energy(self, s, d):
    t0 = 0
    G = self.G
    G_B = self.G_B
    EV = self.vehicle
    v = drive_time
    if G.has_edge(s, d):
        attr = G.get_edge_data(str(s), str(d))
        E_Set = []
        for k in range(len(attr["path"]) - 1):
            E_Set.append(G_B.get_edge_data(attr["path"][k], attr["path"][k + 1]))
        for e in E_Set:
            # 加速度的单位是km/h2
            # 将m/s换算成km/h
            t_l = float(e.get("length")) / self.v
            if e["status_c"] > 1.4:
                t_l = t_l / (1 + (e["status_c"] - 1.4))
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
                * self.v
                * self.v
            ) * self.v
            if P_t > 0:
                com_E = (P_t / EV.eta_add + self.aux * EV.P_a) * t_l / 1000
            else:
                com_E = (P_t / EV.eta_re + self.aux * EV.P_a) * t_l / 1000
        return t0, com_E
    return 1000, 1000


def move_to_next_index(self, next_index):
    self.path.append(next_index)
    avl_time=self.arrival_time[-1] +drive_time(self.cur, next_index)
    self.arrival_time.append(avl_time)


def check_cons(self, next_index):
    energy = self.cal_energy(self.cur, next_index)
    rest_energy = self.vehicle.alpha - (energy / self.vehicle.Q)
    if rest_energy < 0:
        return 1000
    elif rest_energy<self.Re.alpha.min:
        return self.Re.alpha.min-rest_energy
    elif rest_energy>self.Re.alpha.max:
        return rest_energy-self.Re.alpha.max
    return 0
