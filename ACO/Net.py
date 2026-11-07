import pandas as pd
import networkx as nx
import random
from ACO.utility import *
from ACO.util import *
import ast
import re
import torch
import matplotlib.pyplot as plt
from utils.util import StandardScaler2
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
# 读取节点表和边表的 CSV 文件,构建底层路网
class Net():
    def __init__(self,para,size=None):
        self.interval=para.interval
        self.CS_rate=para.CS_rate
        self.args=para
        self.size=size
        self.CS_Set=para.CS_Set
        self.args.info_path=f"{self.args.data_path}{self.args.data}/Net/"
        if size is None:
            self.G_B=self.ConNet(para.node_path,para.edge_path)
            self.G,self.CS_list,self.POI_list,self.nodes_list=self.recover_G()
            # self.G,self.CS_list,self.POI_list,self.nodes_list=self.ConNet_std()
            # self.gen_ins()
        else:
            self.G_B=self.ConNet_limit(para.node_path,para.edge_path,size)
            self.G,self.CS_list,self.POI_list,self.nodes_list=self.recover_G()
            # self.G,self.CS_list,self.POI_list,self.nodes_list=self.ConNet_std()
            # self.G=self.simple_connected()
            # self.gen_ins()
        print("Upper-layer network is constructed!")
        para.req.s,para.req.des=random.sample(self.POI_list,2)
        # self.subG=self.extract_subG(para.req)
        self.node_list_B=list(self.G_B.nodes())
        
        # self.detail_ins()
        self.subG=self.G
        self.node_to_index,self.index_to_node=self.build_reflect(self.nodes_list)
        self.CS_to_index,self.index_to_CS=self.build_reflect(self.CS_list) 
        # self.scaler=para.scaler
        # self.scaler2=para.scaler2
    def gen_ins(self):
        POI_list=self.POI_list[:]
        if self.size is not None:
            path=f"ran_ins_{self.args.data}_simple.csv"
        else:
            path=f"ran_ins_{self.args.data}.csv"
        write(path,["s","d"])
        for j in range(10):
            random.shuffle(POI_list)
            sd_list=[]
            for i in range(0,len(POI_list),2):
                if i+1<len(POI_list):
                    if nx.has_path(self.G,POI_list[i],POI_list[i+1]) and nx.has_path(self.G_B,POI_list[i],POI_list[i+1]):
                        write(path,[POI_list[i],POI_list[i+1]],self.args.info_path)


    def build_reflect(self,list1):
        reflect=dict()
        con_reflect=dict()
        i=0
        for node in list1:
            reflect[node]=i
            con_reflect[i]=node
            i+=1
        return reflect,con_reflect
    def revise_by_SD(self,args):
        s=args.req.s
        des=args.req.des
        NodeT_set=set(self.node_list)
        revise=False
        if s not in self.node_list:
            revise=True
            lengths_s,paths_s=nx.single_source_dijkstra(self.G_B,s,weight="length")
            arr_nodes= set(paths_s.keys()).intersection(NodeT_set)
            for node in arr_nodes:
                path=paths_s[node]
                list_ex=path[1:-1]
                self.G.add_edge(s,node,length=paths_s[node],path=path)
            # for nei in list(self.G_B.successors(s)):
            #     if nei in self.node_list:
            #         self.G.add_edge(s,nei,length=lengths_s[s][nei],path=paths_s)
        if des not in self.node_list:
            revise=True
            lengths_d,paths_d=self.single_target_dijkstra(self.G_B,des)
            arr_nodes= set(paths_d.keys()).intersection(NodeT_set)
            for node in arr_nodes:
                path=paths_s[node]
                list_ex=path[1:-1]
                # isadded=bool(set(list_ex)&NodeT_set)
                # if not isadded:
                self.G.add_edge(node,des,length=lengths_d[node],path=path)
        if s not in self.G.nodes() or des not in self.G.nodes():
            return -1
        if not revise:
            return 0
        if revise:
            self.node_list=list(self.G.nodes)
            self.node_to_index,self.index_to_node=self.build_reflect(self.nodes_list)
            self.CS_to_index,self.index_to_CS=self.build_reflect(self.CS_list) 
            return 1


    def single_target_dijkstra(self,graph, target_node):
        # 反转图
        reversed_graph = graph.reverse()
        
        # 使用 Dijkstra 算法计算从目标节点到所有其他节点的最短路径
        lengths,paths  = nx.single_source_dijkstra(reversed_graph, target_node,weight="length")
        # 将路径和长度存储在字典中，并反转路径
        shortest_paths = {}
        for node, path in paths.items():
            if node in self.node_list and node != target_node:
                shortest_paths[node] = (path[::-1], lengths[node])  # 反转路径

        return shortest_paths,lengths
    
    # 构建顶层路网
    def ConNet_T(self):
        G_T = nx.DiGraph()
        # max_degree_node = max(self.G_B.degree(), key=lambda x: x[1])  # (node_id, degree)
        # nextNode, _ = max_degree_node
        Tnodes = set()
        visited = set()
        degree_dict = self.G_B.degree()
        # 将节点和度数转换为列表并按度数排序
        sorted_nodes = sorted(degree_dict, key=lambda x: x[1], reverse=False)
        # 输出排序后的节点
        for node, _ in sorted_nodes:
            if node not in visited:
                nextNode = node
                while nextNode:
                    try:
                        preNode = nextNode
                        newNode,_ = self.next_node(self.G_B, self.interval, nextNode, Tnodes, visited)
                        if newNode is None:
                            break
                        G_T.add_node(nextNode)
                        pathLength=nx.shortest_path_length(self.G_B, preNode, newNode, weight="length")
                        
                        G_T.add_edge(
                            preNode,
                            newNode,
                            length=pathLength,
                        )
                        nextNode=newNode
                    except:
                        # print(f"unreach {preNode} {nextNode}")
                        i=0
        CS_num = int(G_T.number_of_nodes() * self.CS_rate)
        if CS_num > len(self.CS_Set):
            CS_aug = Guss_aug(self.CS_Set, len(self.CS_Set) - CS_num)
            self.CS_Set.extend(CS_aug)
        node_list = list(G_T.nodes)
        CS_list = random.sample(node_list, CS_num)
        POI_list = [item for item in node_list if item not in CS_list]
        print(f"CS_list{len(CS_list)} POI_list{len(POI_list)} Node_list{len(node_list)} ")
        i=0
        for cs in CS_list:
            G_T.nodes[cs]["iscs"] = 1
            G_T.nodes[cs]["cs"] = self.CS_Set[i]
            i+=1
        self_loops = list(nx.selfloop_edges(G_T))  # 获取所有自环
        G_T.remove_edges_from(self_loops) 
        self.write_G_to_file(G_T)

        return G_T, CS_list,POI_list
        
    def gen_angle(self):
        p = random.random()
        if p < 0.8:
            angle = random.randint(0, 15)
        elif p < 0.9:
            angle = random.randint(-15, 0)
        else:
            angle = random.randint(15, 30)
        return angle
    def ConNet_PEMS08(self):
        G = nx.Graph()
        day = random.randint(0, 59)
        with open("./PEMS08/distance.csv") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for line in csv_reader:
                G.add_edge(
                    line[0],
                    line[1],
                    length=float(line[2]) / 1000,
                    v_l=10,
                    v_u=60,
                    angle=self.gen_angle(),
                    T=random.randint(20, 30),
                )
        pems08_data = np.load("./PEMS08/pems08.npz")
        t = (day * 12 * 24) + int((3 * 60) / 5)
        print(pems08_data["data"].shape)
        flow = pems08_data["data"][t, :, 0]
        occupy = pems08_data["data"][t, :, 1]
        speed = pems08_data["data"][t, :, 2]
        flow = flow.T
        occupy = occupy.T
        speed = speed.T
        print(flow.shape)
        print(speed.shape)
        print(occupy.shape)
        for i in range(occupy.shape[0]):
            G.nodes[str(i)]["speed"] = speed[i]
            G.nodes[str(i)]["flow"] = flow[i]
            G.nodes[str(i)]["occupy"] = occupy[i]
        return G

    def write_G_to_file(self,G):
        nodes_data = G.nodes(data=True)  # 获取所有节点及其属性
        nodes_df = pd.DataFrame(nodes_data, columns=['Node', 'Attributes'])
        # 将节点属性中的字典展开为多个列
        nodes_expanded = pd.json_normalize(nodes_df['Attributes'])
        nodes_df = pd.concat([nodes_df[['Node']], nodes_expanded], axis=1)

        # 写入节点数据到 CSV
        if self.size is None:
            nodes_df.to_csv(f'{self.args.info_path}GTnodes.csv', index=False)
        else:
            nodes_df.to_csv(f'{self.args.info_path}GTnodes_simple{self.size}.csv', index=False)

        # 2. 写入边属性到 CSV 文件
        edges_data = G.edges(data=True)  # 获取所有边及其属性
        edges_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Attributes'])

        # 将边属性中的字典展开为多个列
        edges_expanded = pd.json_normalize(edges_df['Attributes'])
        edges_df = pd.concat([edges_df[['Source', 'Target']], edges_expanded], axis=1)

        # 写入边数据到 CSV
        if self.size is None:
            edges_df.to_csv(f'{self.args.info_path}GTedges.csv', index=False)
        else:
            edges_df.to_csv(f'{self.args.info_path}GTedges_simple{self.size}.csv', index=False)
    def com_EB(self,s,d,v=20):
        e=self.G_B.get_edge_data(s, d)
        t_l = float(e.get("length")) / v     
        EV=self.args.vehicle
        req=self.args.req    
        # EV.M:kg g:m/s2 v:m/s A:m2; pho:kg/m3
        P_t = (
            EV.M
            * (
                req.g
                * (
                    math.sin(math.radians(float((e["angle"]))))
                    + EV.Cr * math.cos((math.radians((float(e["angle"])))))
                )
            )
            + 0.5
            * EV.Cd
            * EV.A
            * (req.pho * (273.15 / (e["T"] + 273.15)))
            * v
            * v
        ) * v
        if P_t > 0:
            com_E = (P_t / EV.eta_add + req.aux * EV.P_a) * t_l / 1000
        else:
            com_E = (P_t / EV.eta_re + req.aux * EV.P_a) * t_l / 1000
        return t_l,com_E
    def com_E(self, s, d, v=20):
        G_T=self.G
        G_B=self.G_B
        req=self.args.req
        EV=self.args.vehicle
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
                t_l = float(e.get("length")/1000) / v
                t0 += t_l
                # EV.M:kg g:m/s2 v:m/s A:m2; pho:kg/m3
                P_t = (
                    EV.M
                    * (
                        req.g
                        * (
                            math.sin(math.radians(float((e["angle"]))))
                            + EV.Cr * math.cos((math.radians((float(e["angle"])))))
                        )
                    )
                    + 0.5
                    * EV.Cd
                    * EV.A
                    * (req.pho * (273.15 / (e["T"] + 273.15)))
                    * v
                    * v
                ) * v
                if P_t > 0:
                    com_E = (P_t / EV.eta_add + req.aux * EV.P_a) * t_l / 1000
                else:
                    com_E = (P_t / EV.eta_re + req.aux * EV.P_a) * t_l / 1000
                total_E+=com_E
            return t0*60, total_E
        return 10000, 10000
    def ConNet_std(self):
        G_T = nx.DiGraph()
        G_B=self.G_B
        nodes=list(G_B.nodes())
        node_T_num=int(len(nodes)/self.interval)
        CS_num=int(node_T_num*self.CS_rate)
        node_list=list(G_B.nodes())
        degree_dict = G_B.degree()
        # 将节点和度数转换为列表并按度数排序
        sorted_nodes = sorted(degree_dict, key=lambda x: x[1], reverse=False)
        node_list = {node for node, degree in sorted_nodes if degree not in (1, 2)}
        node_T=random.sample(node_list,node_T_num)
        CS_list=random.sample(node_T,CS_num)
        POI_list=[item for item in node_T if item not in CS_list] 
        G_T.add_nodes_from(node_T)
        for cs in CS_list:
            G_T.nodes[cs]["iscs"]=1
        for poi in POI_list:
            G_T.nodes[poi]["iscs"]=0       
        lengths={}
        paths={}
        NodeT_set=set(node_T)
        for nodei in node_T:
            lengths[nodei],paths[nodei]=nx.single_source_dijkstra(G_B,nodei,weight="length")
            for nodej in paths[nodei].keys():
                path=paths[nodei][nodej]
                list_ex=path[1:-1]
                isadded=bool(set(list_ex)&NodeT_set)
                if not isadded:
                    G_T.add_edge(nodei,nodej,length=lengths[nodei][nodej],path=path)
        self_loops = list(nx.selfloop_edges(G_T))  # 获取所有自环
        G_T.remove_edges_from(self_loops) 
        self.remove_single_node_scc(G_T)
        new_node_list=list(G_T.nodes())
        POI_list=[item for item in POI_list if item in new_node_list] 
        CS_list=[item for item in CS_list if item in new_node_list] 
        # self.G_T=self.make_strongly_connected(G_T,lengths,path)
        self.write_G_to_file(G_T)
        self.G=G_T
        return G_T,CS_list,POI_list,nodes
    def remove_single_node_scc(self,graph):
    # 找到所有的强连通分量
        sccs = list(nx.strongly_connected_components(graph))
        if len(sccs) == 1:
            print("图是强连通的")
        else:
            print("图不是强连通的，强连通分量:")
        # 找出只包含一个节点的强连通分量
        single_node_sccs = [scc for scc in sccs if len(scc) <=5]
        
        # 移除这些单节点强连通分量
        for scc in single_node_sccs:
            node = list(scc)[0]
            graph.remove_node(node)
    
        return single_node_sccs 
    # 将经纬度差异转换为直线距离
    def dir_dis(self,s,d):
    # 将经纬度从度转换为弧度
        G=self.G_B
        lat1_rad = math.radians(G.nodes[s]["x"])
        lon1_rad = math.radians(G.nodes[s]["y"])
        lat2_rad = math.radians(G.nodes[d]["x"])
        lon2_rad = math.radians(G.nodes[d]["y"])

        # 计算经纬度差
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # 哈弗辛公式
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # 地球半径（单位：公里）
        R = 6371.0
        distance = R * c

        return distance
    def simple_connected(self):
        G=self.G
        G_B=self.G_B
        
        # 找到入度和出度为零的分量
        in_degree_zero = [node for node, degree in G.in_degree() if degree == 0]
        out_degree_zero = [node for node, degree in G.out_degree() if degree == 0]
        # 添加边以构成强连通图
        for i in range(len(in_degree_zero)):
            for j in range(len(out_degree_zero)):
                if i==j:
                    continue
                if nx.has_path(G_B,in_degree_zero[i],out_degree_zero[j]):
                    shortest_path = nx.shortest_path(G_B, source=in_degree_zero[i], target=out_degree_zero[j], weight='length')
                    path_length = nx.shortest_path_length(G_B, source=in_degree_zero[i], target=out_degree_zero[j], weight='length')
                    G.add_edge(in_degree_zero[i],out_degree_zero[j],length=path_length,path=shortest_path)
                
                    if nx.is_strongly_connected(G):
                        return G
                if nx.has_path(G_B,out_degree_zero[j],in_degree_zero[i]):
                    shortest_path = nx.shortest_path(G_B, source=out_degree_zero[j], target=in_degree_zero[i], weight='length')
                    path_length = nx.shortest_path_length(G_B, source=out_degree_zero[j], target=in_degree_zero[i], weight='length')
                    G.add_edge(out_degree_zero[j],in_degree_zero[i],length=path_length,path=shortest_path)
                    if nx.is_strongly_connected(G):
                        return G
                    
                # component_graph.add_edge(in_degree_zero[i], out_degree_zero[i % len(out_degree_zero)])
        strongly_connected_components = list(nx.strongly_connected_components(G))
        print(f"result:{len(strongly_connected_components)}")
        return G
    def make_strongly_connected(self,graph,length,path):
        # 找到图中的强连通分量
        sccs = list(nx.strongly_connected_components(graph))
        if len(sccs)==1:
            return graph
        
        # 创建缩图
        scc_graph = nx.DiGraph()
        scc_map = {node: i for i, scc in enumerate(sccs) for node in scc}
        scc_mapr = {i: node for i, scc in enumerate(sccs) for node in scc}
        
        # 添加缩图的边
        for u, v in graph.edges():
            # 分属于不同的强连通分量的两个节点之间存在边,那么在缩图中添加相应的边
            scc_u = scc_map[u]
            scc_v = scc_map[v]
            if scc_u != scc_v:
                scc_graph.add_edge(scc_u, scc_v)

        # 找源节点和汇节点
        sources = [scc for scc in range(len(sccs)) if all(scc_graph.in_degree(scc_map[node]) == 0 for node in sccs[scc])]
        sinks = [scc for scc in range(len(sccs)) if all(scc_graph.out_degree(scc_map[node]) == 0 for node in sccs[scc])]

        # 添加边以形成强连通
        for i in range(len(sources)):
            scc_graph.add_edge(sources[i], sinks[i % len(sinks)])
        if len(sources)<len(sinks):
            for i in range(len(sources),len(sinks)):
                scc_graph.add_edge(sources[i% len(sources)], sinks[i])

        # 将添加的边反馈到原图
        for i in range(len(sources)):
            for j in range(min(len(sources[i]),len(sinks[i % len(sinks)]))):
                src = list(sccs[sources[i]])[j]  # 源节点中的一个节点
                sink = list(sccs[sinks[i % len(sinks)]])[-(j+1)]  # 汇节点中的一个节点
                if sink in length[src]:
                    graph.add_edge(src, sink, length=length[src][sink], path=path[src][sink])  # 在原图中添加边
                    break

        return graph
        
    def recover_G(self):
        CS_list=[]
        G = nx.DiGraph()
        # 1. 从 nodes.csv 读取节点数据
        if self.size is None:
            nodes_df = pd.read_csv(f'{self.args.info_path}GTnodes.csv')
        else:
            nodes_df = pd.read_csv(f'{self.args.info_path}GTnodes_simple{self.size}.csv')
        
        # 添加节点及其属性
        for _, row in nodes_df.iterrows():
            G.add_node(int(row['Node']), **row.drop('Node').to_dict())

        # 2. 从 edges.csv 读取边数据
        if self.size is None:
            edges_df = pd.read_csv(f'{self.args.info_path}GTedges.csv')
        else:
            edges_df = pd.read_csv(f'{self.args.info_path}GTedges_simple{self.size}.csv')

        # 添加边及其属性
        for _, row in edges_df.iterrows():
            G.add_edge(int(row['Source']), int(row['Target']), **row.drop(['Source', 'Target']).to_dict())
        self.node_list=list(G.nodes())
        CS_list=[node for node in self.node_list if G.nodes[node]["iscs"]==1]
        POI_list=[node for node in self.node_list if G.nodes[node]["iscs"]==0]
        for u, v, data in G.edges(data=True):
            # 假设 path 属性存储为字符串形式的列表
            path_string = data.get("path", '[]')  # 默认值是空列表字符串
            path_list = ast.literal_eval(path_string)  # 将字符串还原为列表
            G[u][v]["path"]=path_list
        # cycles = list(nx.simple_cycles(G))
        # for cycle in cycles:
        #     # 获取环中边的长度（假设边的长度为1，若有权重可用 G.edges(data=True) 获取）
        #     longest_edge = None
        #     max_length = -1

        #     for i in range(len(cycle)):
        #         current_edge_length = G[cycle[i]][cycle[(i + 1) % len(cycle)]]['length'] 
                
        #         # 更新最长边
        #         if current_edge_length > max_length:
        #             max_length = current_edge_length
        #             longest_edge = (cycle[i], cycle[(i + 1) % len(cycle)])

        #     # 删除最长边
        #     if longest_edge:
        #         G.remove_edge(longest_edge[0], longest_edge[1])
        #         print(f"去掉环 {cycle} 中的最长边: {longest_edge}")
        CS_num = len(CS_list)
        if CS_num > len(self.CS_Set):
            CS_aug = Guss_aug(self.CS_Set, len(self.CS_Set) - CS_num)
            self.CS_Set.extend(CS_aug)
        for i,cs in enumerate(CS_list):
            G.nodes[cs]["cs"] = self.CS_Set[i]
        return G,CS_list,POI_list,self.node_list

    def ConNet_T_random(self):
        G_T = nx.DiGraph()
        G_B=self.G_B()
        nodes=list(self.G_B,nodes())
        node_T=int(len(nodes)/self.interval)
        CS_num=int(node_T*self.CS_rate)
        arc_num = int(node_T / 6)
        for i in range(len(nodes) - 2):
            nodei = nodes[i]
            towrite = str(nodei) + " "
            randlist = list(range(0, node_T))
            for _ in range(arc_num):
                try:
                    nodej = nodes[random.choice(randlist)]
                    towrite += nodej + " "
                    if (nx.has_path(G_T, nodej, nodei) is False) & (
                        nx.has_path(G_B, nodei, nodej)
                    ):
                        short_path = nx.shortest_path(G_B, nodei, nodej, weight="length")
                        path_weight = nx.shortest_path_length(
                            G_B, nodei, nodej, weight="length"
                        )
                        G_T.add_edge(nodei, nodej, length=path_weight, path=short_path)
                except nx.NetworkXNoPath:
                    pass
            node_list = list(G_T.nodes)
            CS_list = random.sample(node_list, CS_num)
            POI_list = [item for item in node_list if item not in CS_list]
            print(f"CS_list{len(CS_list)} POI_list{len(POI_list)} Node_list{len(node_list)} ")
            i=0
            for cs in CS_list:
                G_T.nodes[cs]["iscs"] = 1
                G_T.nodes[cs]["cs"] = self.CS_Set[i]
                i+=1

        return G_T, CS_list,POI_list
    # 进行节点数量的限制以方便测试
    # u,v,key,osmid,bridge,oneway,lanes,name,highway,reversed,length,ref,width,tunnel,access,maxspeed,junction,fid
    def ConNet(self,node_file, edge_file):
        nodes_df = pd.read_csv(node_file)
        edges_df = pd.read_csv(edge_file)
        self.edge_list=dict()
        # 创建一个有向图
        G = nx.DiGraph()
        # 添加节点
        node_attrs = ["y", "x", "street_cou", "highway", "ref"]
        edge_attrs = [
            "u",
            "v",
            "key",
            "osmid",
            "bridge",
            "oneway",
            "lanes",
            "name",
            "highway",
            "reversed",
            "length",
            "ref",
            "width",
            "tunnel",
            "access",
            "maxspeed",
            "junction",
            "fid",
        ]
        # chengdu数据集一共12
        for index, row in nodes_df.iterrows():
            osmid = int(row["osmid"])
            G.add_node(
                osmid,
            )
            for attr in node_attrs:
                if not pd.isna(row[attr]):
                    G.nodes[osmid][attr] = row[attr]
                else:
                    G.nodes[osmid][attr] = None
        unarc=0
        node_list=list(G.nodes())
        for index, row in edges_df.iterrows():
            row["u"]=int(row["u"])
            row["v"]=int(row["v"])
            self.edge_list[row["fid"]]=[row["u"],row["v"]]
            G.add_edge(row["u"], row["v"], length=row["length"],angle=self.gen_angle(),T=20)
            for attr in edge_attrs:
                if not pd.isna(row[attr]):
                    if attr =="osmid":
                        if '[' in row[attr]:
                            osmid=ast.literal_eval(row[attr])
                        else:
                            osmid=int(row[attr])
                        G.edges[row["u"], row["v"]][attr]=osmid
                        continue
                    G.edges[row["u"], row["v"]][attr] = row[attr]
                elif attr=="length":
                    G.edges[row["u"], row["v"]][attr] = self.gen_length(G,node_list)
                    print("this is a none length")
                    unarc+=1
                else:
                    G.edges[row["u"], row["v"]][attr] = None
        # print(f"number of edges: {G.number_of_edges()} unlength: {unarc}")

        return G
    def ConNet_limit(self,node_file, edge_file,size):
        nodes_df = pd.read_csv(node_file)
        edges_df = pd.read_csv(edge_file)
        self.edge_list=dict()
        # 创建一个有向图
        G = nx.DiGraph()
        # 添加节点
        node_attrs = ["y", "x", "street_cou", "highway", "ref"]
        edge_attrs = [
            "u",
            "v",
            "key",
            "osmid",
            "bridge",
            "oneway",
            "lanes",
            "name",
            "highway",
            "reversed",
            "length",
            "ref",
            "width",
            "tunnel",
            "access",
            "maxspeed",
            "junction",
            "fid",
        ]
        # chengdu数据集一共12
        for index, row in nodes_df.iterrows():
            osmid = int(row["osmid"])
            G.add_node(
                osmid,
            )
            for attr in node_attrs:
                if not pd.isna(row[attr]):
                    G.nodes[osmid][attr] = row[attr]
                else:
                    G.nodes[osmid][attr] = None
            if index>=size:
                break
        unarc=0
        
        node_list=list(G.nodes())
        for index, row in edges_df.iterrows():
            row["u"]=int(row["u"])
            row["v"]=int(row["v"])
            if (row["u"] in node_list) and (row["v"] in node_list):
                self.edge_list[row["fid"]]=[row["u"],row["v"]]
                G.add_edge(row["u"], row["v"], length=row["length"],angle=self.gen_angle(),T=20)
                for attr in edge_attrs:
                    if not pd.isna(row[attr]):
                        if attr =="osmid":
                            if '[' in row[attr]:
                                osmid=ast.literal_eval(row[attr])
                            else:
                                osmid=int(row[attr])
                            G.edges[row["u"], row["v"]][attr]=osmid
                            continue
                        G.edges[row["u"], row["v"]][attr] = row[attr]
                    elif attr=="length":
                        G.edges[row["u"], row["v"]][attr] = self.gen_length(G,node_list)
                        print("this is a none length")
                        unarc+=1
                    else:
                        G.edges[row["u"], row["v"]][attr] = None
        # print(f"number of edges: {G.number_of_edges()} unlength: {unarc}")

        return G
    def show_G(G,name=""):
        plt.figure(figsize=(8, 6))  # 设置图形大小
        pos = nx.spring_layout(G)  # 使用 spring 布局算法计算节点位置
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_color='black', font_weight='bold', edge_color='gray')

        # 显示图形
        plt.title("NetworkX Graph Visualization")
        plt.savefig(f"Graph{name}.pdf")
        plt.show()
        
    def next_node(self,G, intervals, node_s, Tnodes, visited):
        initnode=node_s
        queue = [(node_s, 0)]
        step = 0
        max_path_len = 0
        next_node = [[], []]
        visited_new=set()
        while queue and len(visited_new)<20:
            node_s, step = queue.pop(0)
            if step < intervals:
                for neighbor in G.neighbors(node_s):
                    if neighbor not in visited and neighbor not in Tnodes:
                        visited.add(neighbor)
                        visited_new.add(neighbor)
                        # 即两个节点之间所经过边的数量
                        path_len = nx.shortest_path_length(G, node_s, neighbor)
                        if path_len > max_path_len:
                            max_path_len = path_len
                            next_node[0] = next_node[1]
                            next_node[1] = [(neighbor, path_len)]
                        elif path_len == max_path_len:
                            next_node[1].append((neighbor, path_len))
                        elif path_len == (max_path_len - 1):
                            next_node[0].append((neighbor, path_len))
                        queue.append((neighbor, path_len))
        max_step=5
        node_list = {
            node
            for node in visited_new
            if nx.shortest_path_length(G, initnode, node) >= max_step 
        }       
        for max_step in range(intervals - 1, 0, -1):
            if node_list:
                max_degree_node = max(
                    node_list, key=lambda node: G.degree(node), default=None
                )
                pathLengt2=nx.shortest_path_length(G,initnode,max_degree_node,weight="length")
                return max_degree_node,pathLengt2
            node_list = {
                node
                for node in visited_new
                if nx.shortest_path_length(G, initnode, node) == max_step 
            }
        return None
    # 获取任意两点之间的能源
    def GetE(self,s,d):
        if "energy" not in self.G[s][d]:
            self.G[s][d]["energy"]=self.comE()
        return self.G[s][d]["energy"]
  
# 找多从指定源点到终点的所有可行路径
    def extract_subG(self,req):
        self.subG = nx.DiGraph()
        number_of_nodes=0
        sub_to_G={}
        max_dis = nx.shortest_path_length(self.G_B, req.s, req.des)
        max_dis /= 3
        paths = find_all_paths(self.G,req.s, req.des)
        nodes_in_paths=set()
        for path in paths:
            nodes_in_paths.update(path)
            pre_node=req.s
            for i in range(1,len(path)):
                node=path[i]
                if node not in self.subG:
                    sub_to_G[number_of_nodes]=node
                    number_of_nodes+=1
                    self.subG.add_edge(pre_node,node,length=self.G[pre_node][node]["length"])
                    self.subG.add_node(node)
                    attributes = self.G[pre_node][node]
                    # 添加属性到目标图的相同边
                    self.subG.add_edge(pre_node, node, **attributes)
                    neighbors = self.dfs_with_depth_limit(self.subG, self.G, node, max_depth=5)
                    nodes_in_paths.update(neighbors)
            pre_node=node
        return sub_to_G
    def dfs_with_depth_limit(self,G,subG, start, max_depth):
        visited = set()
        neighbors = set()
        def dfs(node, depth):
            if depth > max_depth:
                return
            visited.add(node)
            neighbors.add(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    subG.add_node(node)
                    attributes = G[start][node]
                    # 添加属性到目标图的相同边
                    subG.add_edge(start, node, **attributes)
                    dfs(neighbor, depth + 1)

        dfs(start, 0)
        return neighbors

    
    def GetXq(self,sol):
        X_s=[]
        X_A=[]
        lens=0
        path=sol.path
        if isinstance(sol.t, torch.Tensor):
            time = sol.t.item()  # 将张量转换为 CPU
        else:
            time=sol.t
        highway = {
            "living_street": 1,
            "morotway": 2,
            "motorway_link": 3,
            "plannned": 4,
            "trunk": 5,
            "secondary": 6,
            "trunk_link": 7,
            "tertiary_link": 8,
            "primary": 9,
            "residential": 10,
            "primary_link": 11,
            "unclassified": 12,
            "tertiary": 13,
            "secondary_link": 14,
        }
        node_type = {
            "turning_circle": 1,
            "traffic_signals": 2,
            "crossing": 3,
            "motorway_junction": 4,
            "mini_roundabout": 5,
        }
        length=len(path)
        sumoflength=0
        for t in range(length-1):
            info_Set=[]
            cur=path[t]
            next=path[t+1]
            GB_path=self.G[cur][next]["path"]
            # X_s+=self.G[cur][next]["path"][:-1]
            lens+=len(self.G[cur][next]["path"])-1
            cur_b=cur
            for next_b in GB_path[1:]:
                info=[]
                # osmid=self.G_B[cur_b][next_b]["osmid"]
                # if isinstance(osmid,list):
                # if '[' in osmid:
                    # osmid=ast.literal_eval(osmid)
                #     X_s.append(osmid[1])
                # else:
                # 对长度和累积总长度的归一化[均值，方差]
                # 道路类型，当前长度，总长度，周数，日期，时间，x_a_1,y_a_1,
                # 道路的id序列
                # 
                # 总长度
                X_s.append(self.G_B[cur_b][next_b]["fid"])
                info.append(highway[self.G_B[cur_b][next_b]['highway']] if self.G_B[cur_b][next_b]['highway'] in highway.keys() else 0) 
                info.append(self.G_B[cur_b][next_b]["length"])
                info.append(sumoflength)
                sumoflength+=self.G_B[cur_b][next_b]["length"]
                info+=[self.args.req.week,self.args.req.date]
                info.append(time)
                if time > 1440:
                    continue
                info.append(self.G_B.nodes[cur_b]["x"])
                info.append(self.G_B.nodes[cur_b]["y"])
                info.append(self.G_B.nodes[next_b]["x"])
                info.append(self.G_B.nodes[next_b]["y"])
                X_A.append(np.asarray(info))
                cur_b=next_b
        
        return X_s,X_A,lens,sumoflength
    def Get_time_length(self):
        lenn=len(self.CS_list)
        time=np.zeros((lenn,lenn))
        length=np.zeros((lenn,lenn))
        path_length=[]
        v=self.args.req.v
        # all_edge_length=np.empty((lenn, lenn), dtype=float)
        for i,nodei in enumerate(self.CS_list):
            for j,nodej in enumerate(self.CS_list):
                sum_of_length=0
                shorest_path=nx.shortest_path(self.G_B,nodei,nodej,weight="length")
                cur=shorest_path[0]
                
                for next in list(shorest_path)[1:]:
                    path_length.append(self.G_B[cur][next]["length"])
                    sum_of_length+=self.G_B[cur][next]["length"]
                    cur=next
                time_t=sum_of_length/v
                time[i][j]=time_t
                length[i][j]=sum_of_length
        time_mean=np.mean(time)
        time_std=np.std(time)
        length_mean=np.mean(length)
        length_std=np.std(length)
        edges=np.array(path_length)
        edge_mean=np.mean(edges)
        edge_std=np.std(edges)
        with open(f"{self.args.info_path}scaler.csv","w") as file:
            writer = csv.writer(file)
            writer.writerow([time_mean,time_std])
            writer.writerow([length_mean,length_std])
            writer.writerow([edge_mean,edge_std])
        print("time_mean time_std edge_mean edge_std length_mean length_std")
        print(f"{time_mean} {time_std} {edge_mean} {edge_std} {length_mean} {length_std} ")
        return time_mean,time_std,edge_mean,edge_std,length_mean,length_std
    def Read_time_length(self):
        datas=[]
        with open(f'{self.args.info_path}scaler.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                data=[float(i) for i in row]
                datas.append(data)  # 转换为整数
        time_scaler = StandardScaler2(mean=datas[0][0]/1000*60, std=datas[0][1]/1000*60)
        length_scaler = StandardScaler()
        length_scaler.fit([[0, 0]])
        # length_scaler.mean_ = [107.497195, 3010.37456]
        # length_scaler.scale_ = [131.102877, 2750.78118]
        length_scaler.mean_ = [datas[2][0], datas[0][0]*20]
        length_scaler.scale_ = [datas[2][1], datas[0][1]*20]
        self.args.scaler=time_scaler
        self.length_scaler=length_scaler
        return time_scaler,length_scaler

    def Visualize_sol(self,best_ant,traff,ants):
        coordinates = np.argwhere(traff < 0.5)
        congestion_edges=[]
        con_color=[]
        con_width=[]
        maxx=np.max(traff)
        cmap = cm.Oranges  
        cmap_red=cm.Reds
        node_shapes=dict()
        node_color=dict()
        node_size=dict()
        k=0
        E_set=set()
        V_set=set()
        for x,y in coordinates: 
            con_color.append(cmap_red(1-traff[x][y]))
            con_width.append(min(traff[x][y],5))
            x=self.index_to_node[x]
            y=self.index_to_node[y]
            congestion_edges.append((x,y))
            E_set.add((x,y))
            V_set.update((x,y))
        # 创建子图
        for ant in ants:
            ori_path=ant.ori_path
            edges = list(zip(ori_path, ori_path[1:]))
            E_set.update(edges)
            nodes = set(ori_path)
            V_set.update(nodes)
        ori_path=best_ant.ori_path
        path_edges = list(zip(ori_path, ori_path[1:]))
        path_nodes = set(ori_path)
        V_set.update(nodes)
        H = self.G.edge_subgraph(E_set).copy()  # 只保留路径边
        H.add_nodes_from(V_set)  # 添加路径相关节点
        for cur in ori_path:
            if self.G.nodes[cur]["iscs"]==1:
                node_color[cur]=cmap(best_ant.CO[k].alpha)
                node_shapes[cur]='*'
                node_size[cur]=100
            else:
                node_color[cur]='blue'
                node_shapes[cur]='o'
                node_size[cur]=50
        # 可视化子图
        try:
            pos = nx.spring_layout(H)  # 选择布局
        except  Exception as e:
            try:
                pos = nx.circular_layout(H)  # 如果 spring_layout 失败，尝试 circular_layout
            except Exception as e:
                pos = nx.random_layout(H)  # 如果 circular_layout 也失败，尝试 random_layout
        # pos = nx.circular_layout(H)  # 选择布局
        # pos = nx.random_layout(H)  # 选择布局
        nx.draw_networkx_edges(H, pos, edgelist=congestion_edges, edge_color=con_color, width=1)
        for node in H.nodes():
            if node in node_shapes.keys():
            # 路径节点使用星形（'P'），其他节点使用方形（'s'）
                shape = node_shapes[node] 
                size = node_size[node] 
                color = node_color[node]
            else:
                shape = 'o' 
                size =1
                color = 'grey'
            # 绘制节点
            plt.scatter(*pos[node], s=size, c=[color], marker=shape, edgecolor='black')
        # nx.draw(H, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=15)
        # 突出显示最短路径
        nx.draw_networkx_edges(H, pos, edgelist=path_edges, edge_color='lightgreen', width=2)
        plt.savefig(f"Subgraph_of_Shortest_Path_{self.args.s}_{self.args.des}.pdf")
        # 显示图
        plt.title("Visualize the Best Path")
        plt.show()
      # 根据旅行时间来计算消耗的能量
    def get_e_by_t(self,paths,t_total):
        total_dis=0
        G=self.G
        cur=paths[0]
        for next in paths[1:]:
            total_dis+=G[cur][next]["length"]
            cur=next
        t_l=[self.args.req.t0]
        cur=paths[0]
        E_total=0
        for next in paths[1:]:
            t_l.append(t_l[-1]+t_total*(G[cur][next]["length"])/total_dis)
            E_total+=self.cal_energy(cur,next,t_l=t_l[-1]-t_l[-2])
            cur=next
        return E_total

    def cal_energy(self, s, d,t_l=None,v=None):
        G = self.G
        G_B = self.G_B
        EV = self.args.vehicle
        total_length=G[s][d]["length"]
        if (t_l is None) and (v is None):
            v=self.args.req.v
            t_l=total_length/v
        elif t_l is not None:
            t_l*=60
            v = (float(G[s][d]["length"]))/t_l
        else:
            t_l=float(G[s][d]["length"])/v
        if G.has_edge(s, d):
            attr = G.get_edge_data(s, d)
            E_Set = []
            G[s][d]["energy"]=0
            for k in range(len(attr["path"]) - 1):
                E_Set.append(G_B.get_edge_data(attr["path"][k], attr["path"][k + 1]))
            for e in E_Set:
                # 加速度的单位是km/h2
                # 将m/s换算成km/h
                # EV.M:kg g:m/s2 v:m/s A:m2; pho:kg/m3
                t_l_sub=e["length"]/total_length*t_l
                P_t = (
                    EV.M
                    * (
                        self.args.req.g
                        * (
                            math.sin(math.radians(float((e["angle"]))))
                            + EV.Cr * math.cos((math.radians((float(e["angle"])))))
                        )
                    )
                    + 0.5
                    * EV.Cd
                    * EV.A
                    * (self.args.req.pho * (273.15 / (e["T"] + 273.15)))
                    * v
                    * v
                ) * v
                if P_t > 0:
                    G[s][d]["energy"] += (P_t / EV.eta_add + self.args.req.aux * EV.P_a) * t_l_sub 
                else:
                    G[s][d]["energy"] += (P_t / EV.eta_re + self.args.req.aux * EV.P_a) * t_l_sub 
            G[s][d]["energy"]/=3600000
            return  G[s][d]["energy"]
        return 100000

    def node_convert(self,node):
        return self.sub_to_G[node]
    

    def gen_length(self,G,node_list):
        i=random.choice(node_list)
        neigjbors_list=list(G.neighbors(i))
        while len(neigjbors_list)==0:
            i=random.choice(node_list)
            neigjbors_list=list(G.neighbors(i))
        j=random.choice(list(G.neighbors(i)))
        mean = 1.0
        std_dev = 0.1  # 可以根据需要调整标准差

        # 生成高斯分布的随机数
        random_value = np.random.normal(mean, std_dev)

        # 确保值在 0.8 到 1.2 之间
        bounded_value = np.clip(random_value, 0.8, 1.2)
        return bounded_value*G[i][j]["length"]
    def detail_ins(self):
        args=self.args
        get_ex(args)
        write(f"details_of_ins_{args.data}.csv",["s","des","pathlength","e_dis","time","week","day","t_min","cost_min"])
        for w in  range(20):
            set_ins(args,w)
            pathlength=nx.shortest_path_length(self.G,args.req.s,args.req.des,weight="length")
            e_dis=self.dir_dis(args.req.s,args.req.des)
            write(f"details_of_ins_{args.data}.csv",[args.s,args.des,round(pathlength,4),round(e_dis,4),args.time,args.week,args.day,round(args.req.t_min,4),round(args.req.cost_min,4)],self.args.info_path)









def Guss_aug(CS_Set, num):
    CS_aug = []
    n = len(CS_Set)
    while len(CS_aug) < num:
        new_CS=copy.deepcopy(CS_Set[random.randint(0,len(CS_Set))])
    return CS_aug


def read_CS(fileName):
    pattern_1 = r"(\d*)[^\d]*元/度"
    pattern_2 = r":(\d+\.\d);"
    pattern_3 = r"(\d{2}):(\d{2})-(\d{2}):(\d{2})"
    # 指定要提取的列名
    specified_column = "p_service"
    df = pd.read_excel(fileName)
    CS_Set = []
    slow_len = []
    slow_mean = [0, 0, 0, 0, 0]
    slow_std = [0, 0, 0, 0, 0]
    fast_len = []
    fast_mean = [0, 0, 0, 0, 0]
    fast_std = [0, 0, 0, 0, 0]
    # 获取指定列的所有数据
    for index, row in df.iterrows():
        if pd.notnull(row["slow_i"]):
            slow_i = re.findall(r"(\d+)(\\.\\d+)?(?=\s*KW)", row["slow_i"])
            if len(slow_i) == 0:
                slow_i = 7.0
            else:
                slow_i = slow_i[0]
        else:
            slow_i = 7.0
        if pd.notnull(row["fast_i"]):
            fast_i = re.findall(r"(\d+)(\\.\\d+)?(?=\s*KW)", row["fast_i"])
            if len(fast_i) == 0:
                fast_i = 40
            else:
                fast_i = fast_i[0]
        else:
            fast_i = 40
        opentime = row["opentime"]
        name = row["name"]
        pos = row["pos"]
        fast = row["fast"]
        slow = row["slow"]
        if pd.notnull(row["p_service"]):
            p_service = row["p_service"]
            pr = con_pr(expression=p_service)
        else:
            pr = PiecewiseFunction()
        if pd.notnull(row["p_charge"]):
            p_charge = row["p_charge"]
            pc = con_pr(expression=p_charge)
        else:
            pc = PiecewiseFunction()
        r = ran_r(8)
        cs_temp = CS(
            [0, slow, fast, r],
            [[0 for _ in range(24)], con_f(slow), con_f(fast), con_f(r)],
            [0, slow_i, fast_i, 0.15],
            [0, pc, pr, 100],
        )
        CS_Set.append(cs_temp)
    return CS_Set


def con_pr(expression):
    piecewise = PiecewiseFunction()
    matches = []
    price_list = []
    time_list = []
    if len(expression) == 0:
        random_float = round(random.uniform(0.8, 1.2), 2)
        matches.append(random_float)
        piecewise.add_segment(lambda x: 0 < x < 24, lambda x: matches[0])
        return piecewise
    pattern_1 = r"(\d*)[^\d]*元/度"
    pattern_2 = r":(\d+\.\d);"
    pattern_3 = r"(\d{2}):(\d{2})-(\d{2}):(\d{2})"
    pattern_4 = r"(\d{2}):(\d{2})~(\d{2}):(\d{2})"
    pattern_5 = r"电价:(\d+\.\d+)"
    pattern_6 = r"(?:\d{2}:\d{2}~\d{2}:\d{2}(?:,|\s|:)?|)(\d+\.\d+)"
    matches = re.findall(pattern_2, expression)
    time_ranges = re.findall(pattern_3, expression)
    if len(time_ranges) == 0:
        time_ranges = re.findall(pattern_4, expression)
    if len(time_ranges) != 0:
        if len(matches) != len(time_ranges):
            matches = re.findall(pattern_5, expression)
        elif len(matches) != len(time_ranges):
            matches = re.findall(pattern_6, expression)
        matches=[s for s in matches if s.strip()]
        for i in range(min(len(time_ranges), len(matches))):
            time_range = time_ranges[i]
            start_hour = int(time_range[0])
            start_minute = int(time_range[1])
            end_hour = int(time_range[2])
            end_minute = int(time_range[3])
            piecewise.add_segment(
                lambda x: start_hour < x < end_hour, lambda x: matches[i]
            )
            price_list.append(matches[i])
            time_list.append(end_hour)
    else:
        matches = re.findall(pattern_1, expression)
        matches=[s for s in matches if s.strip()]
        if len(matches) == 0:
            random_float = round(random.uniform(0.8, 1.2), 2)
            matches.append(random_float)
        piecewise.add_segment(lambda x: 0 < x < 24, lambda x: matches[0])
    if len(piecewise.segments) == 0:
        random_float = round(random.uniform(0.8, 1.2), 2)
        matches.append(random_float)
        piecewise.add_segment(lambda x: 0 < x < 24, lambda x: matches[0])
    return piecewise


def con_f(bound):
    p = random.random()
    f = []
    for _ in range(24):
        if p < 0.8:
            f.append(random.randint(0, bound))
        else:
            f.append(random.randint(bound, bound + 3))
    return f


def ran_r(bound):
    p = random.random()
    if p < 0.7:
        return 0
    else:
        return random.randint(1, bound)


class PiecewiseFunction:
    def __init__(self):
        self.segments = []

    def add_segment(self, condition, function):
        self.segments.append((condition, function))

    def evaluate(self, x):
        if len(self.segments) != 0:
            for condition, function in self.segments:
                if condition(x):
                    return function(x)
            return 1
        else:
            return 1
