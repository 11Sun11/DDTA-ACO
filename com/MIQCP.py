from gurobipy import *
from ACO.Ant import *
# 取消对等待时间的计算
# 快充的充电价格为1,慢充0.5
# 功率快充固定40,慢充7


def MIQCP(args):
    G=args.net.G
    net=args.net
    G_B=args.net.G_B
    Re=args.req
    EV=args.vehicle
    CS_list=args.net.CS_list
    nodes = list(G.nodes())
    for u, v in G.edges():
        edge_info = G.get_edge_data(u, v)
        if "com_E" in edge_info:
            com_E = edge_info["com_E"]
            tr = edge_info["tr"]
        else:
            com_E = net.cal_energy(u, v)
            tr=G[u][v]["length"]/(Re.v*1000)
            G[u][v]["com_E"] = com_E
            G[u][v]["tr"] = tr
    edges = set()
    edges_length = dict()
    edges_comE = dict()
    for u, v in G.edges():
        edges.add((u, v))  # 添加正方向
        edges.add((v, u))  # 添加反方向
        edges_length[(u, v)] = G[u][v]["length"]
        edges_length[(v, u)] = G[u][v]["length"]
        edges_comE[(u, v)] = G[u][v]["com_E"]
        edges_comE[(v, u)] = G[u][v]["com_E"]
    model = Model("LD-CRP prroblem")
    # 决策变量，xij表示i->j的边是否被选择，Qj表示充了多少电
    x = model.addVars(edges, vtype=GRB.INTEGER, lb=0, ub=1, name="x")
    # 到达时间的计算
    arrival_time = model.addVars(nodes, vtype=GRB.CONTINUOUS, name="arrival_time")
    int_arr_time = model.addVars(nodes, vtype=GRB.INTEGER, name="int_arr_time")
    alpha = model.addVars(CS_list, vtype=GRB.CONTINUOUS, name="alpha")
    # 充电方式的选择
    Q_slow = model.addVars(CS_list, vtype=GRB.INTEGER, lb=0, ub=1, name="Q_slow")
    Q_fast = model.addVars(CS_list, vtype=GRB.INTEGER, lb=0, ub=1, name="Q_fast")
    Q_replace = model.addVars(CS_list, vtype=GRB.INTEGER, lb=0, ub=1, name="Q_replace")
    # 在每个节点上的汽车电量
    Q_amount = model.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="Q_amount")
    # 等待时间的协变量
    # wait = model.addVars(CS_list, vtype=GRB.CONTINUOUS, lb=0, name="wait")
    # 充电时间
    tc = model.addVars(CS_list, vtype=GRB.CONTINUOUS, lb=0, name="tc")
    # 充电开销
    cost_cs = model.addVars(CS_list, vtype=GRB.CONTINUOUS, lb=0, name="cost_cs")
    for node in nodes:
        model.addConstr(int_arr_time[node] <= arrival_time[node], name="leg")
        model.addConstr(int_arr_time[node] >= arrival_time[node] - 1, name="beg")
        model.addConstr(Q_amount[node] >= 0, name=f"non_negative_{node}")
        model.addConstr(Q_amount[node] <= 1, name=f"non_surpass_{node}")
        if node == Re.s:
            model.addConstr(Q_amount[node] == Re.init / EV.Q, name="begin_Q")
            model.addConstr(
                quicksum(x[i] for i in edges if i[0] == node) == 1,
                name=f"flow_start_{node}",
            )
        elif node == Re.des:  # 终点
            model.addConstr(
                quicksum(x[i] for i in edges if i[1] == node) == 1,
                name=f"flow_end_{node}",
            )
        else:
            model.addConstr(
                quicksum(x[i] for i in edges if i[0] == node)
                - quicksum(x[i] for i in edges if i[1] == node)
                == 0,
                name=f"flow_balance_{node}",
            )
    model.setObjective(
        quicksum(cost_cs[n] for n in CS_list) + arrival_time[Re.des],
        GRB.MINIMIZE,
    )
    for cs in CS_list:
        # 等待时间的限制
        model.addConstr(Q_amount[cs] + alpha[cs] <= 1, name="beg")

        model.addConstr(
            Q_fast[cs] <= 1,
            name=f"opt_{cs}",
        )
        model.addConstr(
            cost_cs[cs]
            == tc[cs] * (Q_fast[cs]),
            name=f"cost_cs{cs}",
        )
        model.addConstr(
            tc[cs]
            == alpha[cs] * EV.Q / 40 * Q_fast[cs],
            name=f"opt_{cs}",
        )
    # 添加到达时间的约束
    for i in edges:
        # 添加开销的约束
        if i[0] in CS_list:
            model.addConstr(
                Q_amount[i[1]]
                <= Q_amount[i[0]] + alpha[i[0]] - edges_comE[i] / EV.Q + (1 - x[i]),
                name=f"Q_amount_update_{i[0]}",
            )
            model.addConstr(
                arrival_time[i[1]]
                >= arrival_time[i[0]]
                + tc[i[0]]
                + edges_length[i] / Re.v
                - (1 - x[i]),
                name=f"time_update_{i[0]}",
            )
        else:
            model.addConstr(
                Q_amount[i[1]] <= Q_amount[i[0]] - edges_comE[i] / EV.Q + (1 - x[i]),
                name=f"Q_amount_update_{i}",
            )
            model.addConstr(
                arrival_time[i[1]]
                >= arrival_time[i[0]] + edges_length[i] / Re.v - (1 - x[i]) ,
                name=f"time_update_{i}",
            )
    model.setParam("Method", 1)
    model.optimize()

    path = [Re.s]
    cur = Re.s
    while True:
        Found_next = False
        for edge in edges:
            if edge[0] == cur and x[edge].X > 0.5:
                path.append(edge[1])
                cur = edge[1]
                Found_next = True
                break
        if not Found_next:
            break
    CP = []
    sol=Ant(args.req.init,net.node_to_index[Re.s],net.node_to_index[Re.deses])
    sol.ori_path=path
    for node in path:
        if node in CS_list:
            opt = 0
            if Q_fast[node].X == 1:
                opt = 2
            CP.append(CO(opt, alpha[node].X))
    return sol
