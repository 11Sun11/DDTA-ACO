'''
Author: Xin-Can Wang
Date: 2024-12-21 19:40:45
LastEditors: Xin-Can Wang
LastEditTime: 2025-01-11 13:24:24
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/com/EVPRP.py
'''
from queue import Queue
import heapq
from ACO.Ant import*
class node_s:
    def __init__(self,s,alpha,t,pre=None):
        self.s=s
        self.alpha=round(alpha,1)
        self.t=t
        self.pre=pre
    def __lt__(self, other):
        return self.t < other.t

    def __eq__(self, other):
        return self.t == other.t
class node_cp:
    def __init__(self,s,alpha,t,cp):
        self.s=s
        if alpha is not None:
            self.alpha=round(alpha,2)
        else:
            self.alpha=alpha
        self.t=t
        self.cp=cp
    def __lt__(self, other):
        return self.t < other.t

    def __eq__(self, other):
        return self.t == other.t
class node_out:
    def __init__(self,n,m,t_a,alpha_pre,alpha_a,alpha_d,t_d,pre=None):
        self.d=n
        self.a=m
        if alpha_a is not None:
            self.alpha_a=round(alpha_a,2)
        else:
            self.alpha_a=alpha_a
        self.alpha_pre=alpha_pre
        self.alpha_a=alpha_a
        # if alpha_d is not None:
        #     self.alpha_d=round(alpha_d,2)
        # else:
        #     self.alpha_d=alpha_d
        self.alpha_d=alpha_d
        self.t_a=t_a
        self.t_d=t_d
        self.pre=pre

def EVPRP(args):
    global weight_t
    weight_t=dict()
    weight_E=dict()
    s=args.req.s
    d=args.req.des
    req=args.req
    net=args.net
    G=args.net.G_B
    G_T=args.net.G
    v=args.req.v
    Q=args.vehicle.Q
    vehicle=args.vehicle
    pri_queue=[]
    out_list=dict()
    ub=10000000
    
    node_pre=node_out(s,None,None,None,None,req.init,req.t0,None)
    cur=node_s(s,req.init,req.t0,node_pre)
    LCB=[1,0.5]
    V=dict()
    T_dict=dict()
    end_Node=None
    heapq.heappush(pri_queue,cur)
    while pri_queue:
        node_list=[]
        task=heapq.heappop(pri_queue)
        nodei=task.s
        t_cur=task.t
        alpha_pre=task.alpha
        node_pre=task.pre
        flag=True
        for nodej in G_T.successors(nodei):
            signal=False
            if nodei in weight_t.keys():
                if nodej in weight_t[nodei].keys():
                    E=weight_E[nodei][nodej]
                    t=weight_t[nodei][nodej]
                    signal=True
            else:
                weight_t[nodei]=dict()
                weight_E[nodei]=dict()
            if not signal:
                E=net.cal_energy(nodei,nodej)
                _,E=net.com_E(nodei,nodej)
                if torch.is_tensor(E):
                    E=E.cpu().item()
                t=G_T[nodei][nodej]["length"]/(v*1000)
                weight_t[nodei][nodej]=t
                weight_E[nodei][nodej]=E
            t_d=t_a=t_cur+t
            # alpha_a记录到达该节点的SoC
            alpha_a=alpha_pre-E/Q
            alpha_d=alpha_a
            if G_T.nodes[nodej]["iscs"]==1:
                for b in LCB:
                    if b< alpha_a:
                        break
                    t_ch= vehicle.nolinear_charging(
                                b-alpha_a,
                                alpha_a,
                                G_T.nodes[nodej]["cs"].P[1]
                            )
                    t_d+=t_ch
                    if t_d>ub:
                        flag=False
                        break
                    alpha_d=b
                    # 离开上一个节点的SoC，到达该节点的SoC,以及离开该节点的SoC
                    node_o=node_out(nodej,nodei,t_a,alpha_pre,alpha_a,alpha_d,t_d,node_pre)
                    node_list.append(node_o)
                    # out_list.append(node_o)
                    if nodej not in out_list.keys():
                        out_list[nodej]=dict()
                    if round(alpha_d,2) not in out_list[nodej]:
                        out_list[nodej][round(alpha_d,2)]=[]
                    out_list[nodej][round(alpha_d,2)].append(node_o) 
                    alpha_str=format(alpha_d,".1f") 
                    key=f"{nodej}_{alpha_str}"
                    if nodej!=req.des and (key not in T_dict.keys()):
                        T_dict[key]=t_d
                        heapq.heappush(pri_queue,node_s(nodej,alpha_d,t_d,node_o))         
            else:
                if alpha_a>0:
                    alpha_d=alpha_a
                    node_o=node_out(nodej,nodei,t_a,alpha_pre,alpha_a,alpha_a,t_d,node_pre)
                    # out_list.append(node_o)
                    if nodej not in out_list.keys():
                        out_list[nodej]=dict()
                    if round(alpha_d,2) not in out_list[nodej]:
                        out_list[nodej][round(alpha_d,2)]=[]
                        # out_list[nodej][round(alpha_d,2)]=node_o
                    out_list[nodej][round(alpha_d,2)].append(node_o)
                    alpha_str=format(alpha_d,".1f")
                    key=f"{nodej}_{alpha_str}"
                    if nodej!=req.des and (key not in T_dict.keys()):
                        T_dict[key]=t_d
                        heapq.heappush(pri_queue,node_s(nodej,alpha_d,t_d,node_o))
            
            if nodej==req.des:
                if ub>t_d:
                    ub=t_d
                    end_Node=node_out(nodej,nodei,t_a,alpha_pre,alpha_a,alpha_a,t_d,node_pre)
            
            if alpha_d>0:
                alpha_str=format(alpha_d,".1f")
                key=f"{nodej}_{alpha_str}"
                if  key in T_dict.keys():
                    if T_dict[key]>t_d:
                        for node in pri_queue:
                            if node.s==nodej and node.alpha==b:
                                node.t=t_d
            if flag == False:
                break
    sol= recover(end_Node,args)  
    return sol
    # return ub,out_list
def BackwardPass(args,in_list,ub):
    req=args.req
    CPs=[] 
    Q=Queue()
    Q.put(node_cp(req.des,None,ub,[(req.des,None)]))
    while not Q.empty():
        cur=Q.get()
        rows=[]
        # for node in in_list:
        #     if node.d==cur.s and node.t_a<=ub:
            # if node.d==cur.s and node.t_a<=ub:
            # if node.d==cur.s and node.t_a<=ub:
        if cur.alpha is None:
            for alphak in in_list[cur.s].keys():
                for node in in_list[cur.s][alphak]: 
                    if node.t_a<=ub:
                        rows.append(node)
        else:
            alphak=round(cur.alpha,2)
            if cur.s in in_list:
                if alphak in in_list[cur.s]:
                    for node in in_list[cur.s][alphak]: 
                        if node.t_a<=ub:
                            rows.append(node)
        for r in rows:
            m=r.a
            # 到达这个节点的alpha值
            alpha_a=r.alpha_pre
            ub_m=ub-weight_t[m][cur.s]-(r.t_d-r.t_a)
            p_m=cur.cp
            p_m.append((m,r.alpha_d-r.alpha_a))
            if m==req.s:
                CPs.append(p_m)
            else:
                Q.put(node_cp(m,alpha_a,ub_m,p_m))
    return CPs
def convert_sol(CPs,args):
    sols=[]
    for cp in CPs:
        cur=args.req.s
        sol=Ant(args.req.init,args.req.s,args.req.des)
        for c in cp:
            co=CO(1,c[1])
            next=c[0]
            flag=True
            if nx.has_path(args.net.G,cur,next):
                shortest_path=nx.shortest_path(args.net.G,cur,next)
                sol.ori_path+=shortest_path
                co.index=len(sol.ori_path)-1
                sol.CO.append(co)
            else:
                flag=False
                break
        if flag:
            sols.append(sol)
    return sols
def EVPRP_main(args):
    return EVPRP(args)
    # ub,out_list=EVPRP(args)
    # CPs=BackwardPass(args,out_list,ub)
    # sols=convert_sol(CPs,args)
    # return sols
def recover(node,args):
    path=[]
    COs=[]
    while node:
        path.append(node.d)
        if node.alpha_a is not None and node.alpha_a!=node.alpha_d:
            COs.append(CO(1,node.alpha_d-node.alpha_a))
        node=node.pre
    path=path[::-1]
    COs=COs[::-1]
    sol=Ant(args.req.init,args.req.s,args.req.des)
    sol.ori_path=path
    sol.CO=COs
    return [sol]



            

        


            



                    
                    


    
    