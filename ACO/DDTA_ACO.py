'''
Author: Xin-Can Wang
Date: 2024-10-19 15:07:50
LastEditors: Xin-Can Wang
LastEditTime: 2025-08-03 14:54:59
Description: ACO优化器
FilePath: /wangxincan/code/ACO-MulT-TTE/ACO/Dy_ACOV4.py
'''

import time
import copy
import gc
from ACO.Ant import*
import torch
from tqdm import tqdm
from utils.util import *
from ACO.utility import *
from torch.utils.data.dataloader import DataLoader
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import random

global energy_rec
global com_time
com_time=[[],[]]
class ACO_optimizer():
    def __init__(self,para,choice=1,method="_basic_aco_diversity"):
        # 记录有多少只蚂蚁走过这条路径
        self.short_inc=False
        self.iter=0
        # 信息素的因子
        self.args=para
        self.diff_guide=True
        #启发式信息的因子
        self.alpha=para._alpha
        self.beta=para._beta
        #信息素挥发因子
        self.pho=para.pho
        self.surr=True
        # 信息素矩阵
        # 双层网络结构，包括底/顶层路网，充电站列表
        self.net=para.net
        self.req=para.req
        # 蚂蚁数量
        self.ants_num=para.ants_num
        # 最大迭代次数
        self.max_iter=para.max_iter
        # q0表示直接选择概率最大的下一点的概率
        self.q0=para.q0
        self.qc=para.qc
        self.p0=para.p0
        self.vehicle=para.vehicle
        self.CS_list=self.net.CS_list
        self.model=para.model
        self.update=0
        self.ant_GS=True
        self.CP_GS=True
        self.abl_MES_one=True
        # self.pre_gen_ins()
        self.early_stop=para.early_stop
        # 索引零表示平均速度，索引1表示权重，长度
        self.archive_size=self.ants_num
        self.nodes_list=self.net.nodes_list
        self.node_to_index=self.net.node_to_index
        self.index_to_node=self.net.index_to_node
        self.CS_to_index=self.net.CS_to_index
        self.index_to_CS=self.net.index_to_CS
        self.update_repair=True
        # self.CO_taui=self.CO_tau=self.init_COtau()
        self.elite_abl=2
        # self.pre_gen_ins()
        if choice==1:
            self.taui=self.tau = self.init_protau()
            self.m_et=self.init_h_m_et()
        else:
            self.recover_h_tau()
        self.elite=True
        self.iter_best=[]
        self.E_iter_best=[]
        self.E_iter_best_fit=[]
        self.E_iter_best_time=[]
        self.E_iter_best_cost=[]
        self.simple=False
        self.ants=[]
        self.archive_ants=[]
        self.archive_fits=[]
        self.archive_dis=np.full((self.archive_size, self.archive_size), np.inf)
        self.best_ant=None
        self.best_fit=np.inf
        self.his_best_fit=[]
        self.his_best_time=[]
        self.his_best_cost=[]
        self.repair=True
        # self.MEM_bestPath=dict()
        self.pre_v=to_var(torch.tensor([0,0], dtype=torch.float32),self.args.device)
        # self.init(para)
        # 记录算法每一次的时间记录
        # 分为四个部分，local search, build ants(构建蚂蚁),
        # 0:构建蚂蚁，1：eval, 2:time_eval, 3:local search, 4:repair1, 5:repair2,6:MEM
        self.time_rec=[[] for _ in range(7)]
        self.mode="mm"

    def init(self,args):
        self.beta=1-self.alpha
        self.LS_pro= [[0.5 for j in range(5)] for i in range(3)]
        self.mode="mm"
        self.E_iter_best=[]
        self.E_iter_best_fit=[]
        self.E_iter_best_time=[]
        self.E_iter_best_cost=[]
        self.time_rec=[[] for i in range(7)]
        # args.req.s,args.req.des=self.index_to_node[],int(example[1])
        self.node_to_index=self.net.node_to_index
        self.index_to_node=self.net.index_to_node
        self.CS_to_index=self.net.CS_to_index
        self.index_to_CS=self.net.index_to_CS
        self.pre_v=to_var(torch.tensor([0,0], dtype=torch.float32),self.args.device)
        # 用sigmod函数进行转换
        self.LS_pro= [[j for j in range(5)] for i in range(2)]
        self.iter_best=[]
        self.ants=[]
        self.best_ant=None
        self.best_fit=np.inf
        self.his_best_fit=[]
        self.his_best_time=[]
        self.his_best_cost=[]
        # self.set_mem=True
        # self.MEM_bestPath=dict()
        del self.archive_ants
        self.archive_ants=[]
        self.archive_fits=[]
        self.archive_dis=np.full((self.archive_size, self.archive_size), np.inf)
        self.mem_exceeds=np.zeros(self.tau.shape)
        self.diff_guide=False
        # self.CO_taui=self.CO_tau=self.init_COtau()
        self.recover_h_tau()
        self.init_h_ped()
        gc.collect()
        
    def dec_update(self,ant,start=None,end=None):
        if start is None and end is None:
            path=ant.path[:]
        elif end is None:
            path=ant.path[start:]
        elif start is None:
            path=ant.path[end]
        else:
            path=ant.path[start:end]
        if len(path)==0:
            return
        time_steps=np.arange(len(path))
        decay_sequence = np.exp(-0.1 * time_steps)
        cur=path[0]
        for i,next in enumerate(path[1:]):
            update_value=self.tau[cur][next]-(self.pho/self.ants_num)*(1-decay_sequence[i])
            if update_value< self.min_tau:
                if self.mem_exceeds[cur][next]>-3:
                    update_value=self.min_tau
                    self.mem_exceeds[cur][next]-=1
                elif update_value<0:
                    update_value=self.min_tau/100
            elif update_value>self.max_tau:
                if self.mem_exceeds[cur][next]<3:
                    update_value=self.max_tau
                    self.mem_exceeds[cur][next]+=1
                elif update_value>1:
                    update_value=1

            self.tau[cur][next]=update_value
            cur=next
        mask_upper= (self.tau > self.max_tau)
        indices_mask_lower = np.where((self.tau != 0) & (self.tau < self.min_tau) & (self.mem_exceeds <= -3))
        self.tau[indices_mask_lower] = self.min_tau
        self.tau[mask_upper] = self.max_tau
    # 对找到的可行路径进行信息增强
    def inc_update(self,path,start=0,end=-1,decay=False):
        path=path[start:end]
        time_steps=len(path)
        decay_sequence = np.exp(-0.1 * time_steps)
        decay_sequence/=np.sum(decay_sequence)
        cur=path[0]
        for next in path[1:]:
            self.tau[cur][next]*=2
            cur=next
        self.tau[(self.tau != 0) & (self.tau < self.min_tau)] = self.min_tau
        self.tau[self.tau > self.max_tau] = self.max_tau
    @staticmethod
    def Conversion(x,dict1):
        if isinstance(x,list):
            node=[]
            for i in x:
                node.append(dict1[i])

            return node
        return dict1[x]
    def elite_search(self):
        elite_num=[int(self.ants_num/30) for i in range(3)]
        elite_off=[] 
        pro=self.get_ls_pro()
        pro=softmax(pro)

        if self.elite_abl==2:
            choice = random.choices([0,1], weights=pro, k=1)[0]
            # choice = random.choices([0,1,2], weights=pro, k=1)[0]
        else:
            choice=self.elite_abl
        self.init_ant_cs(self.iter_best)
        if len(self.iter_best)<0.1*self.ants_num:
            LS_group=self.archive_ants
        else:
            LS_group=self.three_select(elite_num)
        if choice==0:
            self.ants=self.LS_point(LS_group)
        elif choice==1:
            self.ants=self.LS_short(LS_group)
        if len(self.ants)==0:
            return
        begin=time.time()
        if self.surr:
            if self.CP_GS:
                fit,times,costs=self.surr_eval_with_repair()
                fit=fit.cpu().numpy()
                
                times=times.cpu().numpy()
                costs=costs.cpu().numpy()
            else:
                _,fit=self.eval_no_repair()
                fit=fit.cpu().numpy()
        else:
            self.ants,fit=self.static_eval()
            fit=fit.cpu().numpy()
        end=time.time()
        self.time_rec[1].append(end-begin)
        if len(fit)==0:
            return 
        
        best_index=np.argmin(fit)
        cur_best_ant=self.ants[best_index]
        cur_best_fit=fit[best_index]
        cur_best_time=times[best_index]
        cur_best_cost=costs[best_index]
        reward=self.best_fit-cur_best_fit
        if cur_best_fit<self.his_best_fit[-1]:
            self.E_iter_best.append(cur_best_ant)
            self.E_iter_best_fit.append(cur_best_fit)
            self.E_iter_best_time.append(cur_best_time)
            self.E_iter_best_cost.append(cur_best_cost)
        else:
            self.E_iter_best.append(self.iter_best[-1])
            self.E_iter_best_fit.append(self.his_best_fit[-1])
            self.E_iter_best_time.append(self.his_best_time[-1])
            self.E_iter_best_cost.append(self.his_best_cost[-1])
        self.LS_pro[choice]=self.update_LS_pro(self.LS_pro[choice],reward)
        return elite_off
    #计算每个算子的当前选择概率
    def get_ls_pro(self):
        pro=[sum(i) for i in self.LS_pro]
        minn=min(pro)
        if min(pro)<0:
            minn+=0.1
            for p in pro:
                p+=minn
        return normalize_probabilities(pro)
    #更新每个算子的概率
    def update_LS_pro(self,ls_pro,new_value):
        ls_pro[:-1]=ls_pro[1:]
        ls_pro[-1]=new_value
        return ls_pro
   
    def get_index(self,ant,p1):
        G=self.net.G
        path=ACO_optimizer.Conversion(ant.path,self.index_to_node)
        count=0
        for p in path[:p1]:
            if G.nodes[p]["iscs"]==1:
                count+=1
        return count

    def init_ant_cs(self,ants):
        G=self.net.G
        cs_list=[]
        cs_index_list=[]
        for ant in ants:
            path=ACO_optimizer.Conversion(ant.path,self.index_to_node)
            for i,p in enumerate(path):
                if G.nodes[p]["iscs"]==1:
                    cs_list.append(p)
            ant.cs_list=cs_list
            ant.cs_index_list=cs_index_list
        return 
        
    def recover_h_tau(self):
        self.energy_info,self.max_e,self.min_e,self.mean_e=self.recover_np_file("energy_info")
        self.time_info,self.max_t,self.min_t,self.mean_t=self.recover_np_file("time_info")
        self.tau,self.max_tau,self.min_tau,self.mean_tau=self.recover_np_file("tau")
        # self.init_protau()
        self.m_et,self.max_m,self.min_m,self.mean_m=self.recover_np_file("m_et")
        has_nan = np.isnan(self.m_et).any()
        if has_nan:
            print("数组中是否存在 NaN 值:", has_nan)
        # self.h=self.req.com_f(self.time_info,self.energy_info,np.zeros(self.time_info.shape))
        self.taui=np.copy(self.tau)
        return
    # 
    #迭代主流程
    def _basic_aco_diversity5(self):
        un_imporve=0
        self.shorest_inc()
        self.iter=-1
        self.best_source=[]
        for self.iter in range(self.max_iter):
            self.better_ants=[]
            begin=time.time()
            self.build_ant(self.ants_num,"")
            end=time.time()
            self.time_rec[0].append(end-begin)
            if self.iter==0:
                self.ants.append(self.shorest_ant)
            for i in range(len(self.ants)):
                self.ants[i].ori_path=ACO_optimizer.Conversion(self.ants[i].path,self.index_to_node)
            self.ants=self.unique_ants(self.ants)
            
            begin=time.time()
            if self.surr:
                if self.simple:
                    fit=self.eval_simple()
                elif self.CP_GS:
                    fit,times,costs=self.surr_eval_with_repair()
                else:
                    _,fit=self.eval_no_repair()
            else:
                self.ants,fit=self.static_eval()
                # fit=fit.cpu().numpy()
            end=time.time()
            com_time[1].append(end-begin)
            self.time_rec[1].append(end-begin)
            if len(fit)==0:
                continue
            fit=fit.cpu().numpy()
            times=times.cpu().numpy()
            costs=costs.cpu().numpy()
            best_index=np.argmin(fit)
           
            
            # better_fit=max(fit[best_index]+0.1,self.best_fit)
            better_fit=self.best_fit*1.1
            if fit[best_index]<better_fit:
                better_index=(fit<better_fit)&(fit!=self.best_fit)
                # if len(fit[better_index])!=0:
                self.better_ants=[self.ants[i] for i,index in enumerate(better_index) if index==True]
                begin=time.time()
                # if self.set_mem:
                self.add_MEM(self.better_ants,fit[better_index])
                end=time.time()
                self.time_rec[6].append(end-begin)
           
            
            cur_best_ant=self.ants[best_index]
            self.iter_best.append(cur_best_ant)
            self.his_best_fit.append(fit[best_index])
            self.his_best_time.append(times[best_index])
            self.his_best_cost.append(costs[best_index])
            if self.best_ant is None or fit[best_index]<self.best_fit:
                self.best_ant=cur_best_ant
                self.best_fit=fit[best_index]
                self.best_time=times[best_index]
                self.best_cost=costs[best_index]
                un_imporve=0
            elif self.best_fit<1000:
                un_imporve+=1
            if un_imporve>=0.7*self.early_stop:
                self.tau=self.taui
            if un_imporve>=self.early_stop:
                break
            if self.best_fit<1000:
                if self.update==0:
                    self.gupdate_tau()
                else:
                    self.gupdate_tau2()
            if self.elite:
                begin=time.time()
                self.elite_search()
                end=time.time()
                self.time_rec[3].append(end-begin)
            # self.req.v=self.pre_v[0]/60
            print(f"Diversity:the best fit{self.best_fit}:{self.str_ant(self.iter_best[-1])}")

            self.ants=[]
        print("________________________________________________________________________")
        # write(f'modeule_time_part_rec.csv',[sum(i) for i in self.time_rec])
        # write(f'Conver_diversity_partinit1.csv',self.his_best_fit)
        if self.elite and self.repair and self.surr:
            write(f'modeule_time_rec.csv_{self.args.data}',[sum(i) for i in self.time_rec])
        else:
            write(f'modeule_time_rec_{self.args.data}_r_{self.repair}_e_{self.elite}_s_{self.surr}_{self.args.w}.csv',[sum(i) for i in self.time_rec])
        # write(f"com/Fitn_optimizers_{args.data}.csv",best_fit_list)
        # write(f'Conver_diversity_init1.csv',self.his_best_fit)
        # write(f'modeule_time_part_rec2.csv',[sum(i) for i in self.time_rec])
        # write(f'Conver_diversity_partinit2.csv',self.his_best_fit)
        if not self.surr:
            self.ants=[self.best_ant]
            _,fit=self.eval_no_repair()
            real_best_fit=fit.cpu().numpy()[0]
            write(f'surr/E_diff_best_fit_{self.surr}_{self.args.data}.csv',[real_best_fit,self.best_fit,abs(real_best_fit-self.best_fit),real_best_fit-self.best_fit])
            self.best_fit=real_best_fit
        if self.elite and len(self.E_iter_best_fit)!=0 and self.surr:
            min_index, min_value = min(enumerate(self.E_iter_best_fit), key=lambda x: x[1])
            if self.best_fit>min_value:
                self.best_fit=min_value
                self.best_ant=self.E_iter_best[min_index]
                self.best_cost=self.E_iter_best_time[min_index]
                self.best_time=self.E_iter_best_cost[min_index]
        if not self.surr:
            # self.repair=False
            new_ant,self.real_fit=self.eval_no_repair(self.E_iter_best)
            # self.E_iter_best_fit=self.E_iter_best_fit.cpu().numpy()
            if new_ant is not None and len(new_ant)!=0:
            # for f in self.real_fit:
            #     if isinstance(f, torch.Tensor):
            #         f=f.item()
                if isinstance(self.real_fit,torch.Tensor):
                    self.real_fit=self.real_fit.cpu().numpy()
            write(f'surr/E_Conver_diversity_{self.surr}_real_{self.args.data}.csv',self.real_fit)
            sub_fit=[abs(i-j) for i,j in zip(self.E_iter_best_fit,self.real_fit)]
            write(f'surr/E_Conver_diff_fit_{self.surr}_{self.args.data}.csv',sub_fit)
        # write(f'opt_path_len_{self.args.data}.csv',[len(self.best_ant.path)])
        return self.best_ant,self.best_fit,self.best_time,self.best_cost
    
    def add_MEM(self,ants,fits):
        if len(ants)==0:
            return
        archive_num=len(self.archive_ants)
        # 先选择比较优秀的解，在选择优秀解中距离最大的解
        if archive_num==0:
            best_index, best_fit = min(enumerate(fits), key=lambda x: x[1])
            self.archive_ants.append(copy.deepcopy(ants[best_index]))
            self.archive_fits.append(best_fit)
            return
        else:
            to_mem_dis=self.dis_ants(ants,self.archive_ants)
            ants_dis=self.dis_ants(ants)
            min_mem_dis=np.min(to_mem_dis, axis=1)
            sol_indices = list(np.argsort(min_mem_dis)[::-1])
            if archive_num<self.ants_num:
                index= np.argmax(min_mem_dis)
                self.archive_ants.append(copy.deepcopy(ants[index]))
                self.archive_fits.append(fits[index])
                self.archive_dis[archive_num, : archive_num] = to_mem_dis[index]
                self.archive_dis[: archive_num, archive_num] = to_mem_dis[index]
                return 
        for i in range(len(ants)):
            min_index = np.argmin(self.archive_dis)
            # 计算最小值在展平矩阵中的行索引和列索引
            row_index, col_index = np.unravel_index(min_index, self.archive_dis.shape)
            if min_mem_dis[sol_indices[i]]<=self.archive_dis[row_index][col_index]:
                break
            replace = max([row_index, col_index], key=lambda x: self.archive_fits[x])
            if self.archive_fits[replace]<fits[sol_indices[i]] and random.random()<0.5:
                continue
            else:
                self.archive_ants[replace] = copy.deepcopy(ants[sol_indices[i]])
                replace = max([row_index, col_index], key=lambda x: self.archive_fits[x])
                self.archive_dis[:archive_num,replace]=to_mem_dis[sol_indices[i]]
                self.archive_dis[replace,:archive_num]=to_mem_dis[sol_indices[i]]
                self.archive_fits[replace]=fits[sol_indices[i]]
                to_mem_dis[:, replace] = ants_dis[:, sol_indices[i]]
                min_mem_dis=np.min(to_mem_dis, axis=1)
       
        return 
    # 按照指定的信息素矩阵构建指定数量的蚂蚁
 
    # 基于特定规则来使用生成充电方案
    def build_ant(self,ants_num,tau):
        pre_co=None
        unfeasible=0
        count=0 
        while count<  ants_num: 
            count+=1
            ant=Ant(self.args.req.init,self.node_to_index[self.req.s],self.node_to_index[self.req.des])
            cur=ant.path[-1]
            last_cs_index=0
            while ant.path[-1]!=self.node_to_index[self.req.des]:
                next_index=self.select_next_index(ant,tau)
                if next_index is None:
                    if self.update_repair and ant.path[-1]!=self.node_to_index[self.req.des] and self.repair:
                        # self.dec_update(ant,max(-3,-len(ant.path)))
                        self.dec_update(ant)
                    break
                if self.index_to_node[next_index] in self.net.CS_list:
                    co=self.rule_CO(self.Conversion(ant.path[last_cs_index:next_index],self.index_to_node))
                    # co.pre=pre_co
                    # pre_co=co
                    ant.move_to_next_index(next_index,co)
                else:
                    ant.move_to_next_index(next_index)
                # 更新本地信息素
                if self.mode!="mm":
                    self.lupdate_tau(cur,next_index,tau)
            if ant.path[-1]==self.node_to_index[self.req.des]:
                self.ants.append(ant)
            elif self.repair and self.ant_GS:
                begin=time.time()
                # if self.update_repair:
                #     self.dec_update(ant)
                ant=self.repair_path(ant)
                end=time.time()
                self.time_rec[4].append(end-begin)
                if ant is not None:
                    self.ants.append(ant)
                unfeasible+=1
    #生成对应的充电规划
    def rule_CO(self,path):
        if len(path)==0:
            return CO(2,0)
        cur=path[0]
        total_E=0
        if self.pre_v[1]==0:
            v=20
        else:
            v=self.pre_v[0]/60
        for next in path[1:]:
            e=self.net.cal_energy(cur,next, v=v)
            total_E+=e
            cur=next
        co=CO(2,max(total_E/self.vehicle.Q,0))
        return co
    # 针对充电操作，初始化信息素矩阵
    # 获得路径对应的消耗总能量
    def get_energy_time(self,path):
        lens=len(path)
        energy=0
        G=self.net.G
        v=20
        time=0
        for i in range(lens-1):
            node=path[i]
            next=path[i+1]
            # 在time时刻经过该路段所需要的时间
            # 基于通行时间计算得到的能量消耗
            time_i=G[node][next]["length"]/v
            energy+=self.net.cal_energy(node,next,t_l=time_i)
            time+=time_i
        return time,energy

    # 计算两点之间的能量
#不同的信息素初始化方式

    # 初始化信息素矩阵
    
    # 根据位置和目的地初始化信息素矩阵
    def init_h_ped(self):
        G=self.net.G
        number_of_nodes=len(self.nodes_list)
        self.pos_info=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        self.h=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        for i,nodei in enumerate(self.nodes_list):
            for j,nodej in enumerate(self.nodes_list):
                if nodej in G.successors(nodei):
                    self.pos_info[i][j]=self.net.dir_dis(nodej,self.req.des)-self.net.dir_dis(nodei,self.req.des)
        self.pos_info,self.max_p,self.min_p,self.mean_p= self.norimalize_negative(self.pos_info)
        mask=self.m_et!=0
        has_negative = np.any(self.m_et < 0)
        if has_negative:
            print("has ngative m_et")
        if np.any(self.pos_info < 0):
            print("negative min_p")
        self.h[mask]=self.pos_info[mask]/self.m_et[mask]
        self.h,*_= self.norimalize(self.h)
        # self.write_np_file("pos_info")
        # np.savetxt('h_pos.csv', self.h, delimiter=',', fmt='%.5f')
        return self.h
    
    # 根据终点位置初始化信息素矩阵，位置增益/时间和能量开支的平均值
    def init_h_m_et(self):
        G=self.net.G
        number_of_nodes=G.number_of_nodes()
        self.TE_mem=dict()
        # 值为一个列表，第一个元素存储消耗时间，第二个维度存储
        self.h=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        self.time_info=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        self.energy_info=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        self.pos_info=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        for i,nodei in enumerate(self.nodes_list):
            for j,nodej in enumerate(self.nodes_list):
                if nodej in G.successors(nodei):
                    path=nx.shortest_path(self.net.G,nodei,nodej)
                    G[nodei][nodej]["h_time"],G[nodei][nodej]["h_energy"]=self.get_energy_time(path)
                    self.time_info[i][j]=G[nodei][nodej]["h_time"]
                    self.energy_info[i][j]=G[nodei][nodej]["h_energy"]
        self.time_info,self.max_t,self.min_t,self.mean_t= self.norimalize(self.time_info)
        self.energy_info,self.max_e,self.min_e,self.mean_e= self.norimalize(self.energy_info)
        self.m_et=(self.time_info+self.energy_info)/2
        self.m_et,self.max_m,self.min_m,self.mean_m= self.norimalize(self.m_et)
        self.write_np_file("time_info")
        self.write_np_file("m_et")
        self.write_np_file("energy_info")
        return self.m_et
    
    # 初始化能量和时间开支的平均值
    def str_ant(self,ant):
        strr=""
        k=0
        for node in ant.ori_path:
            str_node="("+'{:06d}'.format(node)
            if self.net.G.nodes[node]["iscs"]==1:
                str_node+=","+str(ant.CO[k].opt)+","+f"{ant.CO[k].alpha:.2f}"
            str_node+=")"
            strr+=str_node
        return strr
    
    def shorest_inc(self):
        path=nx.shortest_path(self.net.G,self.req.s,self.req.des)
        ant=Ant(self.args.req.init,self.node_to_index[self.req.s],self.node_to_index[self.req.des]) 
        COs=self.gen_CO_by_path(path,0)
        ant.CO=COs
        ant.ori_path=path
        path=ACO_optimizer.Conversion(path,self.node_to_index)
        ant.path=path
        # self.inc_update(path)
        self.shorest_ant=ant
        self.eval_no_repair(self.shorest_ant)

    def path_inc(self):
        self.path=list(nx.all_simple_paths(self.net.G,self.req.s,self.req.des))
        for path in self.path:
            self.inc_update(path)
        return 
    
    # 任意两点之间的启发式信息获取方式：将通行时间和消耗能量做平均，首先进行归一化，之后做平均
    # 两点间的所有路径中计算得到的最小值倒数作为启发式函数的信息
    def write_scale(self,X_A_Set):
        with open("length_sumlength.csv", 'a') as f:
            np.savetxt(f, X_A_Set[:,1:3], delimiter=',', fmt='%d', header='', comments='')
        with open("x1y1_x2y2.csv", 'a') as f:
            np.savetxt(f, X_A_Set[:,6:10], delimiter=',', fmt='%d', header='', comments='') 
    # 按照权重平均的方法更新权重矩阵   
        
    # 收集蚂蚁构建的解决方案
    def collate_func(self,sols):
        time=[0 for _ in range(len(sols))]
        lens = []
        X_s_Set=[]
        X_A_Set=[]
        lengths_Set=[]
        for sol in sols:
            X_s,X_A,lenn,sumoflength=self.net.GetXq(sol)
            lengths_Set.append(sumoflength)
            X_s_Set.append(X_s)
            lens.append(lenn)
            X_A_Set+=X_A
        lens=np.asarray(lens, dtype=np.int16)
        # con_links = np.concatenate(X_A, dtype='object')
        # X_s_Set=np.concatenate(X_s_Set,)
        X_A_Set=np.array(X_A_Set,dtype='object')
        self.write_scale(X_A_Set)
        X_A_Set[:,1:3]=self.net.length_scaler.transform(X_A_Set[:,1:3])
        X_A_Set[:,6:10]=self.net.xy_scaler.transform(X_A_Set[:,6:10])
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(
        (*mask.shape, 1 + 2 + 3 + 4), dtype=np.float32
        )  # 最后一个数是segment embedding维度，需要依据不同的维度更换
        padded[mask] = X_A_Set
        rawlinks = np.full(
            mask.shape, fill_value=self.args.data_config["edges"] + 1, dtype=np.int16
        )
        rawlinks[mask] = np.concatenate(X_s_Set)
        mask_label_tmp = []
        sub_input_tmp = []
        for k in X_s_Set:
            tmp1, tmp2 = self.random_mask(k, rate=self.args.mask_rate)
            mask_label_tmp.append(tmp1)
            sub_input_tmp.append(tmp2)
        mask_label = np.full(mask.shape, dtype=np.int16, fill_value=-100)
        mask_label[mask] = np.concatenate(mask_label_tmp)
        linkindex = np.full(
        mask.shape, fill_value=self.args.data_config["edges"] + 1, dtype=np.int16
    )
        linkindex[mask] = np.concatenate(sub_input_tmp)
        mask_encoder = np.zeros(mask.shape, dtype=np.int16)
        mask_encoder[mask] = np.concatenate([[1] * k for k in lens])
        return {
        "links": torch.FloatTensor(padded),  # 路段的特征
        "lens": torch.LongTensor(lens),  # 每一个轨迹序列中路段的个数
        "mask_label": torch.LongTensor(mask_label),  # 掩码路段序列
        "linkindex": torch.LongTensor(linkindex),
        "rawlinks": torch.LongTensor(rawlinks),  # 路段序列-轨迹中的路段序列
        "encoder_attention_mask": 
            torch.LongTensor(mask_encoder),
        "lengthSet":torch.LongTensor(lengths_Set)
          # 为长度不一的路段生成的掩码
    }, time

    #   return {'links':torch.FloatTensor(padded), 'lens':torch.LongTensor(lens), 'inds': inds, 'mask_label': torch.LongTensor(mask_label),
            # "linkindex":torch.LongTensor(linkindex), 'rawlinks': torch.LongTensor(rawlinks),'encoder_attention_mask': torch.LongTensor(mask_encoder)}, time
    def random_mask(self,tokens: np.array, rate: float):
        replaces = np.where(np.random.random(len(tokens)) <= rate)[0]
        replaces = np.array(replaces, dtype=int)
        # 存在索引超出了int16的最大正整数范围
        labels = np.full(len(tokens), dtype=np.int16, fill_value=-100)
        tokens = np.array(tokens.copy())
        labels[replaces] = tokens[replaces]
        tokens[replaces] = np.asarray(
            [self.args.data_config["edges"] + 1] * len(replaces)
        )  # 此处直接赋值会改变dataset原始值，应该考虑采用深拷贝复制一份新数组再更改
        return labels, tokens
    
    # 初始化信息素矩阵,按照1/距离
    # 初始化信息素矩阵，按照pos距离和里程的比值
    def init_protau(self):
        G=self.net.G
        number_of_nodes=G.number_of_nodes()
        self.tau=np.full((number_of_nodes,number_of_nodes),fill_value=0.0, dtype=float)
        for i,nodei in enumerate(self.nodes_list):
            for j,nodej in zip(ACO_optimizer.Conversion(list(G.successors(nodei)),self.node_to_index),list(G.successors(nodei))):
                self.tau[i][j]=abs(self.net.dir_dis(nodei,nodej))/(G[nodei][nodej]["length"]+0.0001)
                if self.tau[i][j]<0:
                    pass
                # print(self.tau[i][j])
        if np.any(self.tau<0):
            print("init negative tau")
        self.tau,self.max_tau,self.min_tau,self.mean_tau=self.norimalize(self.tau)
        if np.any(self.tau<0):
            print("init negative tau")
        self.write_np_file("tau",[self.max_tau,self.min_tau,self.mean_tau])
        self.taui=np.copy(self.tau)
        return self.tau
    # 从节点序列向边序列的转换
    def node_to_edge(self,ant):
        cur=ant.path[0]
        edge_path=[]
        for next in ant.path:
            edge_path.append((cur,next))
            cur=next
        return set(edge_path)

    def write_np_file(self,name,first_row=None):
        if first_row is None:
            pre_fix=["max_","min_","mean_"]
            first_row=[]
            for pre in pre_fix:
                first_row.append(getattr(self,pre+name[0]))
        stats_row = np.array([first_row])
        # 将统计值和原矩阵合并
        matrix=getattr(self, name, None)
        if stats_row.shape[1] < matrix.shape[1]:
            padding_width = matrix.shape[1] - stats_row.shape[1]
        stats_row = np.pad(stats_row, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)
        combined_data = np.vstack((stats_row,  getattr(self, name, None)))
        # 将合并后的数据写入 CSV 文件，不加 header
        np.savetxt(f'{self.args.info_path}{name}.csv', combined_data, delimiter=',', fmt='%.5f')

        # 从 CSV 文件读取数据
        
    def recover_np_file(self,name):
        # 提取统计信息
        loaded_data = np.loadtxt(f'{self.args.info_path}{name}.csv', delimiter=',')
        overall_max = loaded_data[0, 0]  # 整个矩阵的最大值
        overall_min = loaded_data[0, 1]  # 整个矩阵的最小值
        overall_mean = loaded_data[0, 2]  # 整个矩阵的均值
        # 恢复原始矩阵
        original_matrix = loaded_data[1:, :]  # 从第二行开始是原始数据
        return original_matrix,overall_max,overall_min,overall_mean
    # 初始化路径选择的信息素矩阵
    # 需要仔细考虑的事情，充电操作的信息素更新方式
    #初始化启发式信息矩阵
   
    
                    
    # 信息素的本地更新
    def lupdate_tau(self,cur,next_index,tau=None):
        if tau=="" or tau is None:
            tau=self.tau
        else:
            tau=getattr(self,f"{tau}_tau")
        update_value=max((1-self.pho)*self.tau[cur][next_index]+self.pho*self.taui[cur][next_index],self.min_tau)
        if update_value< self.min_tau:
            if self.mem_exceeds[cur][next_index]>-3:
                update_value=self.min_tau
            elif update_value<0:
                update_value=self.min_tau/100
        if self.mode=="mm":
            self.tau[cur][next_index]=update_value
            self.tau[cur][next_index]=min(self.tau[cur][next_index],update_value)
    # 信息素的全局更新 
    def gupdate_tau(self):
        # 按照一定概率选择至今最优和迭代最优进行信息素的更新
        if random.random()<self.q0:
            best_path=self.best_ant.path
            cur=best_path[0]
        else:
            best_path=self.iter_best[-1].path
            cur=best_path[-1]
        # self.tau=(1-self.pho)*self.tau
        for next in best_path[1:]:
            update_value=(1-self.pho)*self.tau[cur][next]+self.pho/self.best_fit
            self.tau[cur][next]=update_value
            cur=next
        if self.mode=="mm":
            self.tau[(self.tau != 0) & (self.tau < self.min_tau)] = self.min_tau
            self.tau[self.tau > self.max_tau] = self.max_tau  
        return 
    def gupdate_tau2(self):
        # 按照一定概率选择至今最优和迭代最优进行信息素的更新
        if random.random()<self.q0:
            best_path=self.best_ant.path
            cur=best_path[0]
        else:
            best_path=self.iter_best[-1].path
            cur=best_path[-1]
        for next in best_path[1:]:
            update_value=(1-self.pho)*self.tau[cur][next]+self.pho/math.exp(-0.5*self.best_fit)
            self.tau[cur][next]=update_value
            cur=next
        if self.mode=="mm":
            self.tau[(self.tau != 0) & (self.tau < self.min_tau)] = self.min_tau
            self.tau[self.tau > self.max_tau] = self.max_tau  
        return 
    # 对于拥堵路段的识别和避免
      # 使用代理模型进行评估

    def surr_eval(self,sol,update_v=True):
        if not isinstance(sol,list):
            sol=[sol]
        self.model.eval()
        loader = DataLoader(Datadict(sol), batch_size=50,
                                    collate_fn=lambda x: self.collate_func(x),
                                    shuffle=False, pin_memory=True)
        tqdm_loader = tqdm(loader,mininterval=3, disable=True)
        # 遍历数据加载器，获取输入特征和真实数据
        length_Set=to_var(torch.tensor([]),self.args.device)
        time_Set = to_var(torch.tensor([]),self.args.device)
        for step, (feature, truth_data) in enumerate(tqdm_loader):
            with torch.no_grad():
                feature = to_var(feature, self.args.device)
                length_Set=torch.cat((length_Set,feature["lengthSet"]))
                time,_=self.model(feature,self.args)
                time_Set=torch.cat((time_Set,time))
        return time_Set
    def pre_gen_ins(self):
        args=self.args
        for sd,example in zip(args.sd_ins[1:],args.ins[1:]):
            args.s,args.des=int(sd[0]),int(sd[1])
            args.req.s,args.req.des=int(sd[0]),int(sd[1])
            args.req.par_min(args.net.G_B,args.vehicle)
            args.target_time=int(example[2])
            args.week=int(example[3])
            args.day=int(example[4])
            args.req.t0=args.time=int(example[5])
            self.shorest_inc()
            _,time=self.eval_no_repair(self.shorest_ant)
            self.args.req.t_min=time  
            write(f"ins_{args.data}.csv",[self.args.s,self.args.des,self.args.time,self.args.week,self.args.day,self.args.req.t_min.item(),self.args.req.cost_min],self.args.info_path)
    def static_eval(self,ants=None):
        if ants is None:
            ants=self.ants
        G_T = self.net.G
        G_B = self.net.G_B
        Re = self.args.req
        EV = self.vehicle
        minn=self.req.min
        maxx=self.req.max
        device=self.args.device
        fit = []
        cost_list = []
        time_list = []
        Q=self.vehicle.Q
        remove_list=[]
        for i in range(len(ants)):
            if ants[i].ori_path[-1]!=self.req.des or ants[i].ori_path[0]!=self.req.s:
                self.dec_update(ants[i])
                remove_list.append(i)
        ants=[value for i, value in enumerate(ants) if i not in remove_list]
        i=0
        if len(ants)==0:
            return [],[]
        Penalty=[0 for _ in range(len(ants))]
        Penalty=torch.tensor(Penalty,device=device, dtype=torch.float32)
        for i in range(len(ants)):
            p = ants[i].ori_path
            cp = ants[i].CO
            t = Re.t0
            time = 0
            cost = 0
            tl = 0
            alpha = Re.init
            q = 0
            flag = True
            for j in range(len(p) - 1):
                tem = p[j]
                d = p[j + 1]
                if G_T.nodes[tem]["iscs"] == 1:  
                    temp = cp[q - 1]        
                    if temp.alpha==0:
                        opt=0
                    else:
                        opt=temp.opt
                    cs = p[temp.index]
                    if (temp.opt != 0) & (G_T.nodes[tem]["cs"].O[temp.opt] != 0):
                        
                        wait = 0
                        if (
                            G_T.nodes[tem]["cs"].f[int(temp.opt)][int(time)%24]
                            > G_T.nodes[tem]["cs"].n[int(temp.opt)]
                        ):
                            wait += (
                                G_T.nodes[tem]["cs"].f[int(temp.opt)][int(time)%24]
                                - G_T.nodes[tem]["cs"].n[int(temp.opt)]
                            ) * 0.5
                        if G_T.nodes[cs]["cs"].O[temp.opt] != 0:
                            if temp.opt != 3:
                                if (temp.alpha + alpha)  >= 1:
                                    flag = False
                                    break
                                tc = EV.nolinear_charging(
                                    alpha,
                                    alpha + temp.alpha,
                                    G_T.nodes[cs]["cs"].P[int(temp.opt)],
                                )
                                cost+= (
                                    tc
                                    * G_T.nodes[cs]["cs"]
                                    .p[int(temp.opt)]
                                    .evaluate(t)
                                    * G_T.nodes[cs]["cs"].P[int(temp.opt)]
                                )
                                time += tc
                            else:
                                tc = 0.15
                                cost += 40 + (1.5 * (1 - alpha)*Q)
                                alpha += 1
                                time += tc
                        q+=1
                        if tl > Re.t_l:
                            Penalty[i] += (tl - Re.t_l)/24
                        tl = 0
                        # 不符合要求时，执行修改
                if G_T.has_edge(tem, d):
                    edge_info = G_T.get_edge_data(tem, d)
                    if "com_E" in edge_info:
                        com_E = edge_info["com_E"]
                        tr = edge_info["tr"]
                    else:
                        com_E = self.net.cal_energy(tem, d)
                        G_T[tem][d]["com_E"] = com_E
                        G_T[tem][d]["tr"]=G_T[tem][d]["length"]/(1000*72)
                        tr=G_T[tem][d]["tr"]
                    alpha -= com_E/Q
                    if alpha < 0:
                        flag = False
                        break
                    time += tr
                    tl += tr
                else:
                    flag = False
                    break
            if alpha<0 or (not flag):
                Penalty[i]+=10000
                time_list.append(1000000)
                cost_list.append(1000000)
                flag = False
            elif alpha<minn:
                Penalty[i]+=(minn-alpha)/self.req.min_alpha
            if alpha>maxx:
                Penalty[i]+=(alpha-maxx)/self.req.min_alpha
            # if (alpha < 0) or (not flag):
            #     time_list.append(1000000)
            #     cost_list.append(1000000)
            #     flag = False
            if flag:
                time_list.append(time*60)
                cost_list.append(cost+(self.req.init-alpha)*0.8)
        time_list=torch.tensor(time_list,device=device, dtype=torch.float32)
        cost_list=torch.tensor(cost_list,device=device, dtype=torch.float32)
        fit = Re.com_f(time_list, cost_list,Penalty)
            # print(SoCList)
        for f in fit:
            if isinstance(f, torch.Tensor):
                f=f.item()
        return ants,fit  
    def surr_eval_with_repair(self):
        surr_time=0
        repair_time=0
        device=self.args.device
        tl=self.args.t_l*60
        G=self.net.G
        eval_set=[]
        alphas=[self.req.init for _ in range(len(self.ants))]
        Q=self.vehicle.Q
        minn=self.req.min
        maxx=self.req.max
        remove_list=[]
        for i in range(len(self.ants)):
            if (self.ants[i].ori_path[-1]!=self.req.des or self.ants[i].ori_path[0]!=self.req.s) :
                if self.update_repair:
                    self.dec_update(self.ants[i])
                remove_list.append(i)
        self.ants=[value for i, value in enumerate(self.ants) if i not in remove_list]
        i=0
        if len(self.ants)==0:
            return []
        Penalty=[0 for _ in range(len(self.ants))]
        Penalty=torch.tensor(Penalty,device=device, dtype=torch.float32)
        fit=[0 for _ in range(len(self.ants))]
        time_Cost=[self.req.t0 for _ in range(len(self.ants))]
        time_Cost=torch.tensor(time_Cost,device=device, dtype=torch.float32)
        total_cost=[0 for _ in range(len(self.ants))]
        total_cost=torch.tensor(total_cost, device=device, dtype=torch.float32)
        while True:
            eval_set=[]
            path_set=[]
            # 分别记录每一轮j个蚂蚁在各自的站点上的开支，充电时间，充电等待时间
            cost=[0 for _ in range(len(self.ants))]
            tc=[0 for _ in range(len(self.ants))]
            tc=torch.tensor(tc,device=device, dtype=torch.float32)
            tw=[0 for _ in range(len(self.ants))]
            for j in range(len(self.ants)):
                # 跳过已经判断为不可行的蚂蚁
                if Penalty[j]>=10000:
                    continue
                # 重新初始化等待时间和充电时间为0
                ant=self.ants[j]
                path=ant.ori_path
                if i==0 and len(ant.CO)==0:
                    tuple_path=path
                elif i>=len(ant.CO):
                    continue
                else:
                    co=ant.CO[i]
                    tem=path[co.index]
                    if co.alpha<=0 or co.opt==0:
                        opt=0
                        
                        co.alpha=0
                    else:
                        # opt=int(co.opt+1)
                        opt=int(co.opt)
                        if opt==3:
                            co.alpha=min(1,1-alphas[j])
                        # self.lupdate_co(co,self.CS_to_index[tem])
                    try_alpha=alphas[j]+co.alpha
                    # 尝试按照充电计划充电，并按照
                    begin=time.time()
                    if self.repair:
                        if try_alpha>maxx:
                            if try_alpha>1:
                                temp=min((try_alpha-1),co.alpha)
                                co.alpha-=temp
                                try_alpha-=temp
                            if random.random()<self.req.p0 and try_alpha>maxx:
                                co.alpha-=min((try_alpha-maxx),co.alpha)
                            if co.alpha<0:
                                co.alpha=0
                
                    end=time.time()
                    repair_time+=end-begin        
                    if (
                            G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                            > G.nodes[tem]["cs"].n[opt]
                        ):
                            tw[j] = (
                                G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                                - G.nodes[tem]["cs"].n[opt]
                            ) * 0.5*60
                    if opt==1 or opt==2:
                        tc[j]= self.vehicle.nolinear_charging(
                                co.alpha,
                                alphas[j],
                                G.nodes[tem]["cs"].P[opt]
                            )
                        tca=tc[j].item()/60
                        cost[j] = (
                                    tca
                                    * float(G.nodes[tem]["cs"]
                                    .p[int(opt)]
                                    .evaluate(((time_Cost[j]+tw[j])/60)%24))
                                    * G.nodes[tem]["cs"].P[int(co.opt)]
                                )
                    elif opt==3:
                        tc[j]=15
                        cost[j]=40 + 1.5 * co.alpha*self.vehicle.Q
                    if i==0:
                        tuple_path=path[:co.index+1]
                    else:
                        tuple_path=path[ant.CO[i-1].index:co.index+1]
                # tuple_path=self.Conversion(tuple_path,self.index_to_node)
                flag=False
                unit=eval_unit(tuple_path,j,((time_Cost[j]+tc[j]+tw[j])%1440))
                path_set.append(unit)
                if not flag:
                    eval_set.append(unit)
            if len(path_set)==0:
                break
            if len(eval_set)!=0:
                begin=time.time()
                com_t=self.surr_eval(eval_set)
                end=time.time()
                surr_time+=end-begin
                com_t = torch.squeeze(com_t)
                # 检查最小值
                com_t[com_t < 0] = 200000
                com_t[torch.isinf(com_t)] = 20000
           
            # 在当前站点判断到达下一节点的能量是否足够，如果不够，那么补充能量（补充后的能量不能超过最大容量，但要大于所需要的能量）
            # 以一定的概率p执行可行范围内的随机修改或者最小修改
           
                if com_t.shape == torch.Size([]):
                    com_t=com_t.unsqueeze(0)
            for q,unit in enumerate(path_set):
                j=unit.index
                t=com_t[q]
                if t>tl:
                    Penalty[j]+=(t-tl)/1440
                # 获取路段对应的蚂蚁下标
                E=self.get_e_by_t(unit.path,t)
                ant=self.ants[j]
                path=ant.ori_path
                # 存储每一段路径的消耗能量
                # ant.E_set.append(E)
                try_alpha=alphas[j]-E/Q
                if i<len(ant.CO):
                    co=ant.CO[i]
                
                    try_alpha+=co.alpha
                # 包含了opt==3时E>Q的ke 
                if E>Q :
                    # 该蚂蚁探索出的路径不可行，打标记(极大惩罚值)
                    if i==0:
                        start=0
                    else:
                        start=ant.CO[i-1].index
                    if self.update_repair:
                        self.dec_update(self.ants[j],start,end=co.index)
                    # self.dec_update(self.ants[j],end=co.index)
                    Penalty[j]+=10000
                elif try_alpha<minn and self.repair:
                    begin=time.time()
                    # 当小于用户的预定值时，以一定概率执行修改
                    if i<len(ant.CO):
                        
                        while opt==0:
                            opt=self.vehicle.O[random.randint(1,3)]
                            co.opt=opt
                            co.alpha=0
                    # 充电选择为0，1，2
                        if opt<3:
                            repair_temp=0
                            # 做出修改，重新计算充电时间
                            upper_threshold=(1-(alphas[j]+co.alpha))
                            repair_temp=floor_dec(random.uniform(0,upper_threshold))
                            if repair_temp+try_alpha<0:
                                repair_temp=-try_alpha
                            end=time.time()
                            repair_time+=end-begin
                            cs=path[co.index]
                            tc[j]+= self.vehicle.nolinear_charging(
                                repair_temp,
                                alphas[j]+co.alpha,
                                G.nodes[cs]["cs"].P[int(opt)],
                            )
                            tca=tc[j].item()/60
                            cost[j] = (
                                tca
                                * float(G.nodes[cs]["cs"]
                                .p[int(opt)]
                                .evaluate(((time_Cost[j]+tw[j])/60)%24))
                                * G.nodes[cs]["cs"].P[int(opt)]
                            )
                            co.alpha+=repair_temp
                            # 重新评估
                            tuple_path=unit.path
                            begin=time.time()
                            t_new=self.surr_eval(eval_unit(tuple_path,j,(time_Cost[j]+tc[j]+tw[j])%1440))
                            end=time.time()
                            surr_time+=end-begin
                            t_new=torch.squeeze(t_new)
                            if torch.isinf(t_new):
                                t_new=20000
                            if t_new<0:
                                t_new=-t_new
                            if t>tl:
                                Penalty[j]-=(t-tl)/1440
                            if t_new>tl:
                                Penalty[j]+=(t_new-tl)/1440
                            # 更新SoC，时间
                            t=t_new
                            E=self.get_e_by_t(unit.path,t)
                elif try_alpha<0:
                    Penalty[j]+=10000
                time_Cost[j]+=t+tc[j]+tw[j]
                alphas[j]-=E/Q
                if i<len(ant.CO):
                    alphas[j]+=co.alpha
                # 当E<0时，补充能量至电动汽车装满电
                alphas[j]=min(alphas[j],1)
                total_cost[j]+=cost[j]
            for j in range(len(self.ants)):
                if alphas[j]<minn:
                    Penalty[j]+=(minn-alphas[j])/self.req.min_alpha 
                if alphas[j]>maxx:
                    Penalty[j]+=(alphas[j]-maxx)/self.req.min_alpha           
            # 更新每一个路径对应的时间表
            i=i+1
        # 验证每只蚂蚁的可行性
        for j in range(len(self.ants)):
            total_cost[j]+=(self.req.init-alphas[j])*0.8
            total_cost[j]=max(total_cost[j],0)
            time_Cost[j]-=self.req.t0
        fit,time_n,cost_n=self.req.com_f(time_Cost,total_cost,Penalty)
        self.time_rec[5].append(repair_time)
        self.time_rec[2].append(surr_time)
        return fit,time_n,cost_n
    
    #没有采用线性模型，对于开销只采用近似的方式进行评估
    def eval_simple(self):
        surr_time=0
        repair_time=0
        device=self.args.device
        tl=self.args.t_l*60
        G=self.net.G
        eval_set=[]
        alphas=[self.req.init for _ in range(len(self.ants))]
        Q=self.vehicle.Q
        minn=self.req.min
        maxx=self.req.max
        remove_list=[]
        for i in range(len(self.ants)):
            if self.ants[i].ori_path[-1]!=self.req.des or self.ants[i].ori_path[0]!=self.req.s:
                if self.update_repair:
                    self.dec_update(self.ants[i])
                remove_list.append(i)
        self.ants=[value for i, value in enumerate(self.ants) if i not in remove_list]
        i=0
        
        if len(self.ants)==0:
            return []
        Penalty=[0 for _ in range(len(self.ants))]
        Penalty=torch.tensor(Penalty,device=device, dtype=torch.float32)
        fit=[0 for _ in range(len(self.ants))]
        time_Cost=[self.req.t0 for _ in range(len(self.ants))]
        time_Cost=torch.tensor(time_Cost,device=device, dtype=torch.float32)
        total_cost=[0 for _ in range(len(self.ants))]
        total_cost=torch.tensor(total_cost, device=device, dtype=torch.float32)
        while True:
            eval_set=[]
            path_set=[]
            # 分别记录每一轮j个蚂蚁在各自的站点上的开支，充电时间，充电等待时间
            cost=[0 for _ in range(len(self.ants))]
            tc=[0 for _ in range(len(self.ants))]
            tc=torch.tensor(tc,device=device, dtype=torch.float32)
            tw=[0 for _ in range(len(self.ants))]
            for j in range(len(self.ants)):
                # 跳过已经判断为不可行的蚂蚁
                if Penalty[j]>=10000:
                    continue
                # 重新初始化等待时间和充电时间为0
                ant=self.ants[j]
                path=ant.ori_path
                if i==0 and len(ant.CO)==0:
                    tuple_path=path
                elif i>=len(ant.CO):
                    continue
                else:
                    co=ant.CO[i]
                    tem=path[co.index]
                    if co.alpha<=0:
                        opt=0
                        
                        co.alpha=0
                    else:
                        # opt=int(co.opt+1)
                        opt=co.opt
                        if opt==3:
                            co.alpha=min(1,1-alphas[j])
                        # self.lupdate_co(co,self.CS_to_index[tem])
                    try_alpha=alphas[j]+co.alpha
                    # 尝试按照充电计划充电，并按照
                    begin=time.time()
                    if self.repair:
                        if try_alpha>maxx:
                            if try_alpha>1:
                                temp=min((try_alpha-1),co.alpha)
                                co.alpha-=temp
                                try_alpha-=temp
                            if random.random()<self.req.p0 and try_alpha>maxx:
                                co.alpha-=min((try_alpha-maxx),co.alpha)
                            if co.alpha<0:
                                co.alpha=0
                    end=time.time()
                    repair_time+=end-begin        
                    if (
                            G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                            > G.nodes[tem]["cs"].n[opt]
                        ):
                            tw[j] = (
                                G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                                - G.nodes[tem]["cs"].n[opt]
                            ) * 0.5*60
                    if opt!=0:
                        cost[j]= co.alpha*Q/40
                        tc[j]=cost[j]*60
                    if i==0:
                        tuple_path=path[:co.index+1]
                    else:
                        tuple_path=path[ant.CO[i-1].index:co.index+1]
                flag=False
                unit=eval_unit(tuple_path,j,((time_Cost[j]+tc[j]+tw[j])%1440))
                path_set.append(unit)
                if not flag:
                    eval_set.append(unit)
            if len(path_set)==0:
                break
            if len(eval_set)!=0:
                begin=time.time()
                com_t=self.surr_eval(eval_set)
                end=time.time()
                surr_time+=end-begin
                com_t = torch.squeeze(com_t)
                # 检查最小值
                com_t[com_t < 0] = 200000
                com_t[torch.isinf(com_t)] = 20000
            # 在当前站点判断到达下一节点的能量是否足够，如果不够，那么补充能量（补充后的能量不能超过最大容量，但要大于所需要的能量）
            # 以一定的概率p执行可行范围内的随机修改或者最小修改
           
                if com_t.shape == torch.Size([]):
                    com_t=com_t.unsqueeze(0)
            for q,unit in enumerate(path_set):
                j=unit.index
                t=com_t[q]
                if t>tl:
                    Penalty[j]+=(t-tl)/1440
                # 获取路段对应的蚂蚁下标
                E=self.get_e_by_t(unit.path,t)
                ant=self.ants[j]
                path=ant.ori_path
                # 存储每一段路径的消耗能量
                # ant.E_set.append(E)
                try_alpha=alphas[j]-E/Q
                if i<len(ant.CO):
                    co=ant.CO[i]
                    try_alpha+=co.alpha
                # 包含了opt==3时E>Q的ke 
                if E>Q :
                    # 该蚂蚁探索出的路径不可行，打标记(极大惩罚值)
                    if i==0:
                        start=0
                    else:
                        start=ant.CO[i-1].index
                    self.dec_update(self.ants[j],start,end=co.index)
                    Penalty[j]+=10000
                elif try_alpha<minn and self.repair:
                    begin=time.time()
                    # 当小于用户的预定值时，以一定概率执行修改
                    if i<len(ant.CO):
                        
                        if opt==0:
                            opt=2
                            co.opt=opt
                            co.alpha=0
                    # 充电选择为0，1，2
                        if opt<3:
                            repair_temp=0
                            # 做出修改，重新计算充电时间
                            upper_threshold=(1-(alphas[j]+co.alpha))
                            repair_temp=floor_dec(random.uniform(0,upper_threshold))
                            if repair_temp+try_alpha<0:
                                repair_temp=-try_alpha
                            end=time.time()
                            repair_time+=end-begin
                            cs=path[co.index]
                            cost[j]+= repair_temp*Q/40
                            tc[j]=cost[j]*60
                            # if cost[j]<0:
                            #     print("this is negative")
                            co.alpha+=repair_temp
                            # 重新评估
                            tuple_path=unit.path
                            begin=time.time()
                            t_new=self.surr_eval(eval_unit(tuple_path,j,(time_Cost[j]+tc[j]+tw[j])%1440))
                            end=time.time()
                            surr_time+=end-begin
                            t_new=torch.squeeze(t_new)
                            if torch.isinf(t_new):
                                t_new=20000
                            if t_new<0:
                                t_new=-t_new
                            if t>tl:
                                Penalty[j]-=(t-tl)/1440
                            if t_new>tl:
                                Penalty[j]+=(t_new-tl)/1440
                            # 更新SoC，时间
                            t=t_new
                            E=self.get_e_by_t(unit.path,t)
                elif try_alpha<0:
                    Penalty[j]+=10000
                time_Cost[j]+=t+tc[j]+tw[j]
                alphas[j]-=E/Q
                if i<len(ant.CO):
                    alphas[j]+=co.alpha
                # 当E<0时，补充能量至电动汽车装满电
                alphas[j]=min(alphas[j],1)
                total_cost[j]+=cost[j]
            for j in range(len(self.ants)):
                if alphas[j]<minn:
                    Penalty[j]+=(minn-alphas[j])/self.req.min_alpha 
                if alphas[j]>maxx:
                    Penalty[j]+=(alphas[j]-maxx)/self.req.min_alpha           
            # 更新每一个路径对应的时间表
            i=i+1
        # 验证每只蚂蚁的可行性
        for j in range(len(self.ants)):
            total_cost[j]+=(self.req.init-alphas[j])*0.8
            total_cost[j]=max(total_cost[j],0)
            time_Cost[j]-=self.req.t0
        fit=self.req.com_f(time_Cost,total_cost,Penalty)
        self.time_rec[5].append(repair_time)
        self.time_rec[2].append(surr_time)
        return fit
    #在不对解进行修复性修改的情况下直接进行评估
    def eval_no_repair(self,ants=None):
        if ants is None:
            ants=self.ants
        if not isinstance(ants,list):
            ants=[ants]
        surr_time=0
        repair_time=0
        device=self.args.device
        G=self.net.G
        t_l=self.req.t_l*60
        eval_set=[]
        alphas=[self.req.init for _ in range(len(ants))]
        Q=self.vehicle.Q
        minn=self.req.min
        maxx=self.req.max
        remove_list=[]
        for i in range(len(ants)):
            if len(ants[i].ori_path)==0:
                remove_list.append(i)
            elif ants[i].ori_path[-1]!=self.req.des or ants[i].ori_path[0]!=self.req.s:
                remove_list.append(i)
        ants=[value for i, value in enumerate(ants) if i not in remove_list]
        i=0
        if len(ants)==0:
            return None,[1000]
        Penalty=[0 for _ in range(len(ants))]
        Penalty=torch.tensor(Penalty,device=device, dtype=torch.float32)
        fit=[0 for _ in range(len(ants))]
        time_Cost=[self.req.t0 for _ in range(len(ants))]
        time_Cost=torch.tensor(time_Cost,device=device, dtype=torch.float32)
        total_cost=[0 for _ in range(len(ants))]
        total_cost=torch.tensor(total_cost, device=device, dtype=torch.float32)
        while True:
            eval_set=[]
            path_set=[]
            # 分别记录每一轮j个蚂蚁在各自的站点上的开支，充电时间，充电等待时间
            cost=[0 for _ in range(len(ants))]
            tc=[0 for _ in range(len(ants))]
            tc=torch.tensor(tc,device=device, dtype=torch.float32)
            tw=[0 for _ in range(len(ants))]
            for j in range(len(ants)):
                # 跳过已经判断为不可行的蚂蚁
                if Penalty[j]>=10000:
                    continue
                # 重新初始化等待时间和充电时间为0
                ant=ants[j]
                path=ant.ori_path
                if i==0 and len(ant.CO)==0:
                    tuple_path=path
                elif i>=len(ant.CO):
                    continue
                else:
                    co=ant.CO[i]
                    tem=path[co.index]
                    if co.alpha==0:
                        opt=0
                    else:
                        opt=int(co.opt)
                        if opt==3:
                            co.alpha=min(1,1-alphas[j])
                        # self.lupdate_co(co,self.CS_to_index[tem])
                    # 尝试按照充电计划充电，并按照 
                    if (
                            G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                            > G.nodes[tem]["cs"].n[opt]
                        ):
                            tw[j] = (
                                G.nodes[tem]["cs"].f[opt][int(time_Cost[j]/60)%24]
                                - G.nodes[tem]["cs"].n[opt]
                            ) * 0.5*60
                    if opt==1 or opt==2:
                        tc[j]= self.vehicle.nolinear_charging(
                                co.alpha,
                                alphas[j],
                                G.nodes[tem]["cs"].P[opt]
                            )
                        tca=tc[j].item()/60
                        cost[j] = (
                                    tca
                                    * float(G.nodes[tem]["cs"]
                                    .p[int(opt)]
                                    .evaluate(((time_Cost[j]+tw[j])/60)%24))
                                    * G.nodes[tem]["cs"].P[int(co.opt)]
                                )
                    elif opt==3:
                        tc[j]=15
                        cost[j]=40 + 1.5 * co.alpha*self.vehicle.Q
                    if i==0:
                        tuple_path=path[:co.index+1]
                    else:
                        tuple_path=path[ant.CO[i-1].index:co.index+1]
                    # tuple_path=self.Conversion(tuple_path,self.index_to_node)
                flag=False
                unit=eval_unit(tuple_path,j,((time_Cost[j]+tc[j]+tw[j])%1440))
                path_set.append(unit)
                if not flag:
                    eval_set.append(unit)
            if len(path_set)==0:
                break
            if len(eval_set)!=0:
                begain=time.time()
                com_t=self.surr_eval(eval_set,False)
                end=time.time()
                surr_time+=end-begain
                com_t = torch.squeeze(com_t)
                # 检查最小值
                # if min_value < 0:
                #     print("张量中有负值")
                com_t[com_t < 0] = 200000
                com_t[torch.isinf(com_t)] = 20000
                if com_t.shape == torch.Size([]):
                    com_t=com_t.unsqueeze(0)
            for q,unit in enumerate(path_set):
                j=unit.index
                t=com_t[q]
                if t>t_l:
                    Penalty[j]+=(t-t_l)/1440
                # 获取路段对应的蚂蚁下标
                E=self.get_e_by_t(unit.path,t)
                ant=ants[j]
                path=ant.ori_path
                # 存储每一段路径的消耗能量
                # ant.E_set.append(E)
                
                # 包含了opt==3时E>Q的ke 
                if E>Q :
                    # 该蚂蚁探索出的路径不可行，打标记(极大惩罚值)
                   Penalty[j]+=10000
                time_Cost[j]+=t+tc[j]+tw[j]
                if i <len(ant.CO):
                    co=ant.CO[i]
                    alphas[j]+=co.alpha-E/Q
                # 当E<0时，补充能量至电动汽车装满电
                alphas[j]=min(alphas[j],1)
            
                total_cost[j]+=cost[j]
                       
            # 更新每一个路径对应的时间表
            i=i+1
            for j in range(len(ants)):
                if alphas[j]<minn:
                    Penalty[j]+=(minn-alphas[j])/self.req.min_alpha 
                elif alphas[j]>maxx:
                    Penalty[j]+=(alphas[j]-maxx)/self.req.min_alpha     
            # for unit,t,e in zip(eval_set,com_t,cost):
            #     time[unit.index]+=t
            #     cost[unit.index]+=e
        # 验证每只蚂蚁的可行性
        for j in range(len(ants)):
            total_cost[j]+=(self.req.init-alphas[j])*0.8
            total_cost[j]=max(total_cost[j],0)
            time_Cost[j]-=self.req.t0
        fit=self.req.com_f(time_Cost,total_cost,Penalty)
        return ants,fit
    
    # 根据旅行时间来计算消耗的能量
    def get_e_by_t(self,paths,t_total):
        total_dis=0
        G=self.net.subG
        cur=paths[0]
        for next in paths[1:]:
            total_dis+=G[cur][next]["length"]
            cur=next
        t_l=[self.req.t0]
        cur=paths[0]
        E_total=0
        for next in paths[1:]:
            t_l.append(t_l[-1]+t_total*(G[cur][next]["length"])/total_dis)
            E_total+=self.net.cal_energy(cur,next,t_l=t_l[-1]-t_l[-2])
            cur=next
        return E_total

        # 对drive——time,构建（s,d,t）对的集合
    def select_CO(self,cur,name=""):
        CO_tau=getattr(self,f"{name}CO_tau")
        cur=self.CS_to_index[cur]
        if np.random.random()<self.q0:
            opt,alpha = np.unravel_index(np.argmax(CO_tau[cur]), CO_tau[cur].shape)
        else:
            opt,alpha=ACO_optimizer.stochastic_acceptCO(CO_tau[cur])
        
        alpha*=0.05
        cur_co=CO(opt,alpha)
        if opt<=1:
            self.lupdate_co(cur_co,cur,CO_tau)
        return CO(opt,alpha)
    # 选择下一个节点
    # @staticmethod
    def select_next_index(self,ant,name=""):
        if name is None or name =="" :
            tau=self.tau
        else:
            tau=getattr(self, f"{name}_tau")
        G=self.net.G
        cur=ant.path[-1]
        neighbors=ACO_optimizer.Conversion(list(G.successors(self.index_to_node[cur])),self.node_to_index)
        # 确保不产生环路
        select_neighbor = [item for item in neighbors if item not in ant.path]
        if len(select_neighbor)==0 :
            return None
        if np.any(tau[cur][select_neighbor] < 0):
            print("negative tau")
        elif np.any(self.h[cur][select_neighbor] < 0):
            print("negative h")
        tau_tpi=tau[cur][select_neighbor]
        transition_prob = np.power(tau_tpi,self.alpha)*np.power(self.h[cur][select_neighbor], self.beta)
        if np.sum(transition_prob)==0:
            return None
        transition_prob = transition_prob / np.sum(transition_prob)
        if np.random.random()<self.q0:
            index=np.argmax(transition_prob)
            next_index=select_neighbor[index]
        else:
            next_index=ACO_optimizer.stochastic_accept(select_neighbor,transition_prob)
        return next_index
    # 基于时间信息素矩阵选择下一个节点
    # 基于选择可能性随机选择解决方案
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
        norm_transition_prob = transition_prob/sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]
    @staticmethod        
    def stochastic_acceptCO(transition_prob):
        """
        轮盘赌
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        row,col = transition_prob.shape
        N=row*col
        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob
        # select: O(1)
        while True:
            rnumber=int(N * random.random())
            # randomly select an individual with uniform probability
            c = rnumber%col
            r = int(rnumber/col)
            if random.random() <= norm_transition_prob[r][c]:
                return r,c 
    #对解进行去重
    def unique_ants(self,ants):
        dis=self.dis_ants(ants)
        upper_triangle_indices = np.triu_indices_from(dis, k=1)
        # 获取上三角非零元素的索引
        zero_dis_pairs = [(i, j) for i, j in zip(*upper_triangle_indices) if dis[i, j] == 0]
        to_remove=set()
        for _,j in zero_dis_pairs:
            to_remove.add(j)
        # 使用 enumerate() 实现
        ants= [ant for i, ant in enumerate(ants) if i not in to_remove]
        return ants

    def dis_ants(self,ants1,ants2=None):
        if ants2 is None:
            return self.dis_single_ants(ants1)
        width=len(ants1)
        highth=len(ants2)
        dis = np.zeros((width, highth))
        for i,ant1 in enumerate(ants1):
            for j,ant2 in enumerate(ants2):
                dis[i][j]=ant1.dis_ant(ant2,self.net.G)
        return dis
    def dis_single_ants(self,ants):
        width=len(ants)
        highth=len(ants)
        dis = np.zeros((width, highth))
        for i,ant1 in enumerate(ants):
            j=i
            for ant2 in ants[i:]:
                dis[i][j]=ant1.dis_ant(ant2,self.net.G)
                dis[j][i]=dis[i][j]
                j+=1
        return dis

    # 对不可行方案进行修复
    def repair_path(self,ant):
        if isinstance(ant,list):
            ant=ant[0]
        G=self.net.G
        d=self.req.des
        if not hasattr(ant,'ori_path') or len(ant.ori_path)==0:
            ant.ori_path=ACO_optimizer.Conversion(ant.path,self.index_to_node)
        new_s_index=random.randint(0,len(ant.ori_path)-1)
        new_s=ant.ori_path[new_s_index]
        trys=0
        while (not nx.has_path(G, new_s, d)) and trys<3 and new_s_index>0:
            new_s_index=random.randint(0,new_s_index)
            new_s=ant.ori_path[new_s_index]
            trys+=1
        if nx.has_path(G, new_s, d):
            path=nx.shortest_path(G,new_s,d)
            con_path=ACO_optimizer.Conversion(path,self.node_to_index)
            COs=self.gen_CO_by_path(path,new_s_index)
            ant.path=ant.path[:new_s_index]+con_path
            ant.ori_path=ant.ori_path[:new_s_index]+path
            pre_index=self.CO_number(ant.ori_path[:new_s_index],self.net.G)
            ant.CO=ant.CO[:pre_index]+COs
            return ant
        return None
        
    # 根据路径随机生成充电方案
    def gen_CO_by_path(self,path,pre_index):
        COs=[]
        G=self.net.G
        for i,node in enumerate(path):
            if G.nodes[node]["iscs"]==1:
                # 生成一个数据点
                data = np.random.normal(0.3, 0.1)
                # 限制在 0 到 0.8 之间
                data_clipped = np.clip(data, 0, 0.8)
                co=CO(1,data_clipped)
                co.index=i+pre_index
                COs.append(co)
        return COs
    # 当还没有找到可行域，应该增加随机
    # 刚刚找到可行解，应该以适应度优先
    # 算法后期，应该以多样性
    # 趋于收敛，应当增加随机（检测种群的平均距离以及最优解是否变化）
    # 传入每种解决方案的数量并给出
    def three_select(self,num_list):
        # 根据适应度选
        fit_index = sorted(range(len(self.archive_fits)), key=lambda x: self.archive_fits[x])
        fit_ant=[self.archive_ants[i] for i in fit_index[:num_list[0]]]
        other_ants=[self.archive_ants[i] for i in range(len(self.archive_ants)) if i not in fit_index[:num_list[0]]]
        if len(other_ants)==0:
            if len(self.ants)==0:
                return fit_ant
            elif len(self.ants)<(num_list[2]+num_list[1]):
                return self.ants+fit_ant
            else:
                random_ant=random.sample(self.ants, num_list[2]+num_list[1])
                return fit_ant+random_ant
        random_ant=random.sample(self.ants, num_list[2])
        
        dis=self.dis_ants(other_ants,self.archive_ants)
        min_dis = np.min(dis, axis=1)
        sorted_indices = np.argsort(min_dis)
        diversity_ant=[other_ants[i] for i in sorted_indices[:num_list[1]]]
        
        return fit_ant+diversity_ant+random_ant

    # 本地搜索算子，点变异
    # local-search strategy:充电计划的乱序
    # 乱序本地搜索
    def LS_short(self,ants,num=5):
        if not isinstance(ants,list):
            ants=[ants]
        new_ants=[]
        for ant in ants:
            for i in range(num):
                # 深度复制以防修改原蚂蚁
                new_ant=copy.deepcopy(ant)
                if len(ant.CO)<2:
                    continue
                choices=random.sample(list(range(len(ant.ori_path))),2)
                lower,upper=sorted(choices)
                s=new_ant.ori_path[lower]
                d=new_ant.ori_path[upper]
                lenn=0
                if nx.has_path(self.net.G,s,d):
                    shorest_path=nx.shortest_path(self.net.G,s,d)
                    new_ant.ori_path=new_ant.ori_path[:lower]+shorest_path+new_ant.ori_path[upper+1:]
                    for nv in new_ant.ori_path:
                        if self.net.G.nodes[nv]["iscs"]==1:
                            lenn+=1
                        if lenn>len(new_ant.CO):
                            new_ant.CO.append(CO(1,truncated_gauss(0.4,0.3)))
                if lenn<len(new_ant.CO):
                    new_ant.CO=new_ant.CO[:lenn]
                new_ant.update_co_index(self.net.G)
                new_ant.path=self.Conversion(ant.ori_path,self.node_to_index)
                new_ants.append(new_ant)
        return new_ants

    # 点变异式本地搜索
    def LS_point(self,ants,num=5):
        if not isinstance(ants,list):
            ants=[ants]
        new_ants=[]
        G=self.net.G
        for ant in ants:
            # 深度复制以防修改原蚂蚁
            for _ in range(num):
                new_ant=copy.deepcopy(ant)
                if len(ant.CO)==0:
                    continue
                elif len(ant.CO)==1:
                    p=0
                else:
                    p=random.randint(0,len(ant.CO)-1)
                new_ant.CO[p].alpha=truncated_gauss(0.4, 0.3)
                new_ants.append(new_ant)
        return new_ants
    
    # 将充电计划划分为两段 
    def split_co(self,ant,split_node):
        G=self.net.G
        before_co=[]
        after_co=[]
        path=ant.ori_path
        k=0
        for i,p in enumerate(path):
          
            if G.nodes[p]["iscs"]==1:
                if i<split_node:
                    before_co.append(ant.CO[k])

                else:
                    after_co.append(ant.CO[k])
                k+=1
        return before_co,after_co
    # 将充电计划按照给定的两个点划分成三段
    def split_co2(self,ant,split_nodes):
        G=self.net.G
        before_co=[]
        mid_co=[]
        after_co=[]
        path=ant.ori_path
        k=0
        for i,p in enumerate(path):
            if G.nodes[p]["iscs"]==1:
                if i<split_nodes[0]:
                    before_co.append(ant.CO[k])
                elif i<split_nodes[1]:
                    mid_co.append(ant.CO[k])
                else:
                    after_co.append(ant.CO[k])
                k+=1
        return before_co,mid_co,after_co
    
    # 数据处理块，归一化，异常值处理，主要针对信息素矩阵
    # 对含有负数的矩阵的截断处理，极小负数设置为0
    def negative_rect(self,matrix):
        non_zero_values = pd.Series(matrix[matrix != 0])
        # 计算 Q1 和 Q3
        Q1 = non_zero_values.quantile(0.25)
        Q3 = non_zero_values.quantile(0.75)
        IQR = Q3 - Q1
        # 定义极端值的上下限
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        matrix[matrix<lower_bound]=0
        mask = matrix != 0
        non_zero_values = pd.Series(matrix[mask])
        non_zero_values_clipped = non_zero_values.clip(lower=lower_bound, upper=upper_bound)
        if lower_bound<0:
            inc=1.5*abs(lower_bound)
            non_zero_values_clipped+=inc
            lower_bound+=inc
            upper_bound+=inc
        matrix[mask] = non_zero_values_clipped
        average=np.mean(non_zero_values_clipped)
        return matrix,lower_bound,average,upper_bound,mask
    
    # 利用IQR处理矩阵中的极端值-极大值和极小值
    def IQR_rect(self,matrix):
        non_zero_values = pd.Series(matrix[matrix != 0])
        # 计算 Q1 和 Q3
        Q1 = non_zero_values.quantile(0.25)
        Q3 = non_zero_values.quantile(0.75)
        IQR = Q3 - Q1
        # 定义极端值的上下限
        lower_bound = max(Q1 - 1.5 * IQR,np.min(non_zero_values))
        upper_bound = min(Q3 + 1.5 * IQR,np.max(non_zero_values))
        # 替换极端值为上下限
        # 在非零值中进行替换
        non_zero_values_clipped = non_zero_values.clip(lower=lower_bound, upper=upper_bound)
        mask = matrix != 0
        matrix[mask] = non_zero_values_clipped
        average=np.mean(non_zero_values_clipped)
        return matrix,lower_bound,average,upper_bound,mask
    
    def norimalize(self,matrix):
        matrix_IQR,lower_bound,average,upper_bound,mask=self.IQR_rect(matrix)
        matrix[mask] = (matrix_IQR[mask] - lower_bound/5) / (upper_bound - lower_bound/5)
        upper_bound=np.max(matrix[mask])
        lower_bound=np.min(matrix[mask])
        average=np.mean(matrix[mask])
        return matrix,upper_bound,lower_bound,average
    def norimalize_negative(self,matrix):
        matrix_IQR,lower_bound,average,upper_bound,mask=self.negative_rect(matrix)
        matrix[mask] = (matrix_IQR[mask] - lower_bound/5) / (upper_bound - lower_bound/5)
        upper_bound=np.max(matrix[mask])
        lower_bound=np.min(matrix[mask])
        average=np.mean(matrix[mask])
        return matrix,upper_bound,lower_bound,average
    # 统计路径中CS的数量
    def CO_number(self,path,G):
        number=0
        for i,node in enumerate(path):
            if G.nodes[node]["iscs"]==1:
                number+=1
        return number



            








       
        



