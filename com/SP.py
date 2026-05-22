import networkx as nx

from ACO.Ant import*
import math


class Node:
    def __init__(self, id, timeh=0, energy=0, timef=0, timeg=0, parent=None):
        self.id = id
        self.h = timeh  # 从当前节点到目的节点的一个预估时间
        self.SoC = energy
        self.parent = parent
        self.f = timef  # 实际消耗代价+预估代价后得到的总代价(行驶，等待，和消耗时间)
        self.g = timeg  # 实际消耗代价+预估代价后得到的总代价(行驶，等待，和消耗时间)


def astar_search(args):
    G=args.net.G
    G_B=args.net.G_B
    net=args.net
    Re=args.req
    EV=args.vehicle
    start = Re.s
    Re.v=20
    v=20*3.6
    end = Re.des
    # 用于启发式信息
    start_node = Node(start, energy=Re.init)
    end_node = Node(end)
    # 初始化开放列表和关闭列表
    open_list = {}
    closed_set = set()
    # 将起点加入到开放列表
    open_list[start] = start_node
    while open_list:
        # 从开放列表中选择具有最小f值的节点
        min_node = min(open_list, key=lambda node: open_list[node].f)
        # 到达目标节点，构建路径并返回
        current_node = open_list[min_node]
        if current_node.id == end_node.id:
            path = []
            while current_node:
                path.append(current_node.id)
                current_node = current_node.parent
            return path[::-1]
        # 将当前节点添加到关闭列表中
        del open_list[min_node]
        closed_set.add(current_node.id)
        # 遍历所有可能的移动方向
        for child in G[current_node.id]:
            # if G.nodes[child]["iscs"] != 1:
            #     continue
            if child in closed_set:
                continue
            # 创建新节点
            new_node = Node(child, parent=current_node)
            # 计算新节点的实际代价和估计代价
            # 时间权重为当前节点的充电时间，通往下一个节点的行驶时间，预计等待充电时间之和

            comE = net.cal_energy(current_node.id, child,v=v)
            new_node.SoC = current_node.SoC - comE / EV.Q
            # if new_node.SoC < -100:
            #     continue
            time_total = G[current_node.id][child]["length"]/(1000*v)
            if G.nodes[child]["iscs"] == 1:
                opt = 0
                for opt in [1, 3, 2]:
                    if G.nodes[child]["cs"].n[opt] > 0:
                        break
                new_node.opt = opt

                if (new_node.SoC < Re.max / 100) & (opt != 0):
                    if opt == 3:
                        new_node.SoC = 1
                        time_cnow = 0.15
                    else:
                        new_node.SoC = 0.8
                        time_cnow = (
                            (Re.max / 100 - new_node.SoC)
                            * EV.Q
                            / G.nodes[child]["cs"].P[opt]
                        )
                    if math.isinf(current_node.g): 
                        continue
                    clock = int(current_node.g + time_total) % 24
                    if G.nodes[child]["cs"].n[opt] < G.nodes[child]["cs"].f[opt][clock]:
                        time_wait = (
                            G.nodes[child]["cs"].f[opt][clock]
                            - G.nodes[child]["cs"].n[opt]
                        ) * 0.5
                    else:
                        time_wait = 0
                    time_total += time_cnow + time_wait

            new_node.g = current_node.g + time_total
            flag = False
            # 如果新节点已经在关闭列表中，忽略它
            # 如果新节点已经在开放列表中，比较实际代价是否更小
            for idd, node in open_list.items():
                if new_node.id == idd and new_node.g >= node.g:
                    flag = True
                    break
            if flag is False and nx.has_path(G,child,Re.des):
                # 将新节点添加到开放列表中
                
                new_node.h = func_h(G, child, Re.des, Re.v)
                new_node.f = new_node.g + new_node.h
                open_list[new_node.id] = new_node
    # 无法找到路径
    return None


def recon_path(args,path):
    G=args.net.G
    G_B=args.net.G_B
    net=args.net
    Re=args.req
    EV=args.vehicle
    charge_path = []
    SoC = Re.init
    SoCList = []
    for i in range(len(path) - 1):
        p = path[i]
        com_E = net.cal_energy( path[i], path[i + 1])
        SoCList.append(SoC)
        if G.nodes[p]["iscs"] == 1:
            opt = 0
            for opt in [1, 3, 2]:
                if G.nodes[p]["cs"].n[opt] > 0:
                    break
            if (SoC < 0.8) & (opt != 0):
                tocharge = math.floor((0.8 - SoC) * 100) / 100
                if opt != 3:
                    SoC += tocharge
                else:
                    SoC = 1
                charge_path.append(CO(opt, tocharge))
            else:
                charge_path.append(CO(0, 0.0))
        SoC -= com_E / EV.Q
    # print(SoCList)
    if G.nodes[path[-1]]["iscs"] == 1:
        charge_path.append(CO(0, 0.0))
    sol=Ant(args.req.init,args.req.s,args.req.des)
    sol.CO=charge_path
    sol.ori_path=path
    return [sol]


def func_h(G, s, d, v):
    if nx.has_path(G, s, d):
        length = nx.shortest_path_length(G, s, d, weight="length")
    return length / v


def SCSHT_main(args):
    path = astar_search(args)
    if path is None:
        return []
    else:
        pop = recon_path(args, path)
    return pop
