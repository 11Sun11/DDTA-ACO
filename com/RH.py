import networkx
import numpy as np
import random
from ACO.Ant import *
# pr=0.05
# beta=0.98
# alpha=1
# beta=2
# rho_min=0.001
# rho_max=0.1


# class Ant:
#     def __init__(self, num_ants, iters, alpha, beta, rho, size, pr, phe_max, phe_min):
#         self.num_ants = num_ants
#         self.iters = iters
#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho
#         self.size = size
#         self.pr = pr
#         self.phe_max = phe_max
#         self.phe_min = phe_min


def BACO(G_T, G_B, Ant, ga, Re, N, POI_list, vehicle, CS_Set):
    phermone = np.ones((N, N))
    best_cost = []
    best_path = []
    for iter in range(Ant.iters):
        for ant in range(Ant.num_ants):
            start_node = Re.s
            visited = [False] * N
            start_node_index = POI_list.index(start_node)
            visited[start_node_index] = True
            path = [start_node]
            ant_paths = []
            ant_dis = []
            dis = 0
            pro = []
            for _ in range(N):
                cur_node = path[-1]
                cur_node_index = POI_list.index(cur_node)
                de = 0
                neighbors = list(G_T.neighbors(cur_node))

                if len(neighbors) < 20:
                    weights = [
                        G_T.get_edge_data(cur_node, neighbor)["length"]
                        for neighbor in neighbors
                    ]
                    sorted_neighbors = [
                        neighbor for _, neighbor in sorted(zip(weights, neighbors))
                    ]
                    neighbors = sorted_neighbors[:20]
                pro = {}
                for nextnode in neighbors:
                    nextnode_index = POI_list.index(nextnode)
                    if not visited[nextnode_index]:
                        pro[nextnode] = phermone[cur_node_index][
                            nextnode_index
                        ] ** Ant.alpha * (
                            1.0 / phermone[cur_node_index][nextnode_index] ** Ant.beta
                        )

                        de += pro[nextnode]
                if len(pro) == 0:
                    break
                spin = random.uniform(0, 1)
                acc_pro = 0
                for node in pro.keys():
                    acc_pro += pro[node] / de
                    select = node
                    if acc_pro > spin:
                        break

                path.append(select)
                select_index = POI_list.index(node)
                visited[select_index] = True
                dis += G_T[cur_node][select]["length"]
                if select == Re.d:
                    break
            ant_paths.append(path)
            ant_dis.append(dis)
        best_path.extend(ant_paths)
        best_cost.extend(ant_dis)
        sort_indices = sorted(range(len(best_cost)), key=lambda x: best_cost[x])
        best_path = [best_path[i] for i in sort_indices]
        best_fit = ga.evaluate([best_path], G_T, G_B, Re, vehicle, CS_Set)
        if len(best_path) >= Ant.size:
            best_path = best_path[: Ant.size]
            best_cost = best_cost[: Ant.Esize]
        Ant.phe_max = 1 / ((1 - Ant.rho) * best_fit)
        Ant.phe_min = (Ant.phe_max * (1 - Ant.pr ** (1 / N))) / (
            (N / 2 - 1) * (Ant.pr ** (1 / N))
        )
        update_phe(G_T, phermone, ant_paths, ant_dis, Ant.rho, Ant.phe_max, Ant.phe_min)
    return best_path, best_cost


def RH(args, paths, prior):
    G_T=args.net.G
    G_B=args.net.G_B
    Re=args.req
    v=Re.v
    net=args.net
    vehicle=args.vehicle
    Insert_paths = []
    for path in paths:
        Insert_path = []
        for i in range(len(path) - 1):
            sou = path[i]
            Insert_path.append(sou)
            cost = float("inf")
            min_cs = None
            neighbors = list(G_T.successors(sou))
            for cs in neighbors:
                if G_T.nodes[cs]["iscs"] == 1:
                    if G_T.has_edge(cs, path[i + 1]):
                        temp_cost = G_T.get_edge_data(cs, path[i + 1])["length"]
                        temp_cost += G_T.get_edge_data(sou, cs)["length"]
                        if temp_cost < cost:
                            cost = temp_cost
                            min_cs = cs
            if min_cs is not None:
                Insert_path.append(min_cs)
                if path[i + 1] not in G_T.successors(min_cs):
                    Insert_path.append(sou)
        Insert_path.append(path[-1])
        Insert_paths.append(Insert_path)
    repath = []
    for path in Insert_paths:
        SoC = Re.init
        i = 0
        CP = []
        CP_indexs = []
        ds = 0
        s = 0
        con_soc = []
        supple_soc = []
        flag = True
        # 计算可移除选项时，只需要保证在移除该选项后仍然能够顺利到达路径上的下一个充电站
        # 即当前SoC大于到下一个充电站的累计
        for i in range(len(path) - 1):
            com_E = net.cal_energy(path[i], path[i + 1],v=v)
            com_E /= vehicle.Q
            con_soc.append(com_E)
            SoC -= com_E
            if SoC < 0:
                flag = False
                break
            if G_T.nodes[path[i + 1]]["iscs"] == 1:
                supple_soc.append(Re.max - com_E)
                CP_indexs.append(i + 1)
                for opt in prior:
                    if G_T.nodes[path[i + 1]]["cs"].O[opt] > 0:
                        break
                CP.append(CO(opt, max(0,(Re.max - SoC))))
                SoC = Re.max 
        CP_indexs.append(i + 1)  # 终点
        if not flag:
            continue
        while len(CP) > 0:
            CP_index = -1
            if G_T.nodes[path[0]]["iscs"] == 1:
                CP_index += 1
            s = 0
            ds = 0
            SoC = Re.max
            for i in range(len(path) - 2):
                if G_T.nodes[path[i + 1]]["iscs"] == 1:
                    CP_index += 1
                    if SoC > sum(
                        con_soc[
                            CP_indexs[CP_index] : min(
                                CP_indexs[CP_index + 1], len(path)
                            )
                        ]
                    ):
                        if path[i] in G_T.predecessors(path[i + 2]):
                            dss = G_T.get_edge_data(path[i], path[i + 2])["length"] - (
                                G_T.get_edge_data(path[i], path[i + 1])["length"]
                                + G_T.get_edge_data(path[i + 1], path[i + 2])["length"]
                            )
                        else:
                            dss = 0
                        if dss < ds:
                            s = i
                            s_index = CP_index
                            ds = dss
            if ds != 0:
                del CP[s_index]
                con_soc[s] = net.cal_energy(path[s], path[s + 2],v=v)
                con_soc[s] /= vehicle.Q
                del con_soc[s + 1]
                del path[s + 1]
            else:
                new_ant=Ant(args.req.init,args.net.node_to_index[args.req.s],args.net.node_to_index[args.req.des])
                new_ant.ori_path=path
                new_ant.CO=CP
                repath.append(new_ant)
                break
    return repath


def update_phe(G, phermone, paths, dis, POI_list, rho, phe_max, phe_min):
    for i in range(len(phermone)):
        for j in range(len(phermone[i])):
            phermone *= rho
    for i in range(len(paths)):
        for j in range(len(paths[i] - 1)):
            cur_node = paths[i]
            next_node = paths[i + 1]
            next_node_index = POI_list.index(next_node)
            cur_node_index = POI_list.index(cur_node)
            phermone[cur_node_index][next_node_index] += (
                G.get_edge_data(cur_node_index, next_node_index)["length"] / dis[i]
            )
            phermone[cur_node_index][next_node_index] = min(
                phe_max, phermone[cur_node_index][next_node_index]
            )
            phermone[cur_node_index][next_node_index] = max(
                phe_min, phermone[cur_node_index][next_node_index]
            )
    return phermone


def BACO_Main(G_T, G_B, Ant, ga, Re, N, POI_list, vehicle, CS_Set):
    ant_paths, _ = BACO(G_T, G_B, Ant, ga, Re, N, POI_list, vehicle, CS_Set)
    repath = RH(G_T, G_B, ant_paths, Re, [1, 3, 2], vehicle)
    return repath
