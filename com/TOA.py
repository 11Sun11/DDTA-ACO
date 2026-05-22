'''
Author: Xin-Can Wang
Date: 2024-12-21 14:08:39
LastEditors: Xin-Can Wang
LastEditTime: 2024-12-29 16:06:27
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/com/TOA.py
'''

import math
from ACO.Ant import*

def TOA(path,args):
    G=args.net.G
    G_B=args.net.G_B
    net=args.net
    Re=args.req 
    EV=args.vehicle
    pop = []
    for l in path:
        SoC = Re.init
        if len(l) == 1:
            sol=Ant(args.req.init,args.req.s,args.req.des)
            sol.ori_path=l
            sol.CO=[]
            pop.append(sol)
            continue
        CP = dict()
        energy = []
        socs = []
        last_c = -1
        for i in range(len(l) - 1):
            comE = net.cal_energy(l[i], l[i + 1],v=Re.v)
            comE /= EV.Q
            energy.append(comE)
            socs.append(SoC)
            # 发现电量不够，在最近的上一个充电站充电
            if (SoC - comE) < 0:
                index = -1
                for c in range(len(l[: i + 1]) - 1, -1, -1):
                    if G.nodes[l[: i + 1][c]]["iscs"] == 1:
                        station = l[: i + 1][c]
                        index = c
                        break
                if index == -1:
                    break
                if station in CP:
                    break
                opt = 0
                for opt in [1, 3, 2]:
                    if G.nodes[station]["cs"].n[opt] > 0:
                        break
                CP[station] = [index, 1 - socs[index], opt]
                # 已经充过一次电
                if last_c != -1:
                    if socs[index] > 0.3:
                        CP[l[last_c]] = [last_c, 0.8 - socs[last_c], opt]
                        for k in range(last_c + 1, index + 1):
                            socs[k] -= 0.2
                    else:
                        CP[l[last_c]] = [last_c, 1 - socs[last_c], opt]

                for k in range(index + 1, i + 1):
                    socs[k] += 1 - socs[index]
                SoC = 1
                last_c = index
            SoC -= comE
        socs.append(SoC)
        if last_c != -1:
            if l[last_c] in CP:
                opt = 0
                for opt in [1, 3, 2]:
                    if G.nodes[l[last_c]]["cs"].n[opt] > 0:
                        break
                if SoC > 0.3:
                    CP[l[last_c]] = [last_c, 0.8 - socs[last_c], opt]
                    for k in range(last_c + 1, index + 1):
                        socs[k] -= 0.2
                else:
                    CP[l[last_c]] = [last_c, 1 - socs[last_c], opt]
                    # 先假设充满电
        # if not CP:
        #     sorted_keys = sorted(CP, key=CP.get)
        #     CP = [gene(CP[key][2], CP[key][1]) for key in sorted_keys]
        # else:
        #     CP = []
        CPP = []
        for cs in l:
            if G.nodes[cs]["iscs"] == 1:
                if cs in CP:
                    CPP.append(CO(CP[cs][1], math.floor(CP[cs][1] * 10) / 10))
                else:
                    CPP.append(CO(0, 0.0))
        sol=Ant(args.req.init,args.req.s,args.req.des)
        sol.ori_path=l
        sol.CO=CPP
        pop.append(sol)
    return pop
