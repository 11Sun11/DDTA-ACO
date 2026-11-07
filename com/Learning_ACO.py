'''
Author: Xin-Can Wang
Date: 2024-12-25 10:52:28
LastEditors: Xin-Can Wang
LastEditTime: 2025-02-07 14:45:33
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/com/Learning_ACO.py
'''
from models.MLP import*
from tqdm import tqdm
import time
from utils.util import *
from ACO.utility import *
from ACO.Ant import*
from ACO.DDTA_ACO import*
import torch.nn.functional as F
class Learning_ACO:
    def __init__(self,para):
        self.iter=0
        # 信息素的因子
        self.args=para
        #启发式信息的因子
        self.alpha=para._alpha
        self.beta=para._beta
        #信息素挥发因子
        self.pho=para.pho
        # 信息素矩阵
        # 双层网络结构，包括底/顶层路网，充电站列表
        self.net=para.net
        # 蚂蚁数量
        # self.ants_num=para.ants_num
        self.ants_num=100
        # 最大迭代次数
        self.max_iter=para.max_iter
        # q0表示直接选择概率最大的下一点的概率
        self.q0=para.q0
        self.p0=para.p0
        self.vehicle=para.vehicle
        self.req=para.req
        self.CS_list=self.net.CS_list
        self.model=self.recover_model()
        # 索引零表示平均速度，索引1表示权重，长度
        self.nodes_list=self.net.nodes_list
        self.node_to_index=self.net.node_to_index
        self.index_to_node=self.net.index_to_node
        self.CS_to_index=self.net.CS_to_index
        self.index_to_CS=self.net.index_to_CS
        self.iter_best=[]
        self.ants=[]
        self.best_ant=None
        self.best_fit=np.inf
        self.his_best_fit=[]
        self.init_h()
        # self.init(para)
        # 记录算法每一次的时间记录
        # 分为四个部分，local search, build ants(构建蚂蚁),
        # 0:构建蚂蚁，1：eval, 2:time_eval, 3:local search, 4:repair1, 5:repair2,6:MEM
        self.time_rec=[[] for i in range(7)]
        self.tau=None
        self.taui=None
        self.h=None
    def init(self):
        self.tau=self.tau
        self.his_best_fit=[]
        self.ants=[]
        self.best_ant=None
        self.best_fit=np.inf
        self.init_h()
    def recover_model(self):
        model =  MLP(56, 100, 50, 4)
        # 加载模型权重
        model.load_state_dict(torch.load(f'data/save_models/LACO_MLP/mlp_model_{self.args.data}.pth'))
        model.to(self.args.device)
        # 如果需要，将模型设置为评估模式
        # 现在你可以使用加载的模型进行推断
        # 例如，使用随机输入进行测试
        return model
    def evaluation(self):
        G_T = self.net.G
        G_B = self.net.G_B
        Re = self.args.req
        EV = self.vehicle
        fit = []
        minn=self.req.min
        maxx=self.req.max
        cost_list = []
        time_list = []
        Q=self.vehicle.Q
        remove_list=[]
        for i in range(len(self.ants)):
            if self.ants[i].ori_path[-1]!=self.req.des or self.ants[i].ori_path[0]!=self.req.s:
                self.dec_update(self.ants[i])
                remove_list.append(i)
        self.ants=[value for i, value in enumerate(self.ants) if i not in remove_list]
        i=0
        if len(self.ants)==0:
            return []
        Penalty=[0 for _ in range(len(self.ants))]
        for i in range(len(self.ants)):
            p = self.ants[i].ori_path
            cp = self.ants[i].CO
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
                    cs = p[temp.index]
                    if (temp.opt != 0) & (G_T.nodes[tem]["cs"].O[temp.opt] != 0):
                        tl = 0
                        wait = 0
                        if (
                            G_T.nodes[tem]["cs"].f[int(temp.opt)][int(t/60)]
                            > G_T.nodes[tem]["cs"].n[int(temp.opt)]
                        ):
                            wait += (
                                G_T.nodes[tem]["cs"].f[int(temp.opt)][int(t/60)]
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
                        Penalty[i] += (tl - Re.t_l)/Re.t_l
                        # 不符合要求时，执行修改
                if G_T.has_edge(tem, d):
                    edge_info = G_T.get_edge_data(tem, p[j + 1])
                    if "com_E" in edge_info:
                        com_E = edge_info["com_E"]
                        tr = edge_info["tr"]
                    else:
                        com_E = self.net.cal_energy(tem, d)
                        tr=G_T[tem][d]["length"]/(Re.v*1000)
                        G_T[tem][p[j + 1]]["com_E"] = com_E
                        G_T[tem][p[j + 1]]["tr"] = tr
                    alpha -= com_E/Q
                    if alpha < 0:
                        flag = False
                        break
                    if alpha<minn:
                        Penalty[i]+=(minn-alpha)/self.req.min_alpha 
                    time += tr
                    tl += tr
                else:
                    flag = False
                    break
            if (alpha < 0) or (not flag):
                time_list.append(1000)
                cost_list.append(1000)
                flag = False
            if flag:
                time_list.append(time*60)
                cost_list.append(cost+0.8*(self.req.init-alpha))
            fit = Re.com_f(time_list, cost_list,Penalty)
            # print(SoCList)
        return fit 
    def init_h(self):
        G=self.net.G
        number_of_nodes=G.number_of_nodes()
        self.TE_mem=dict()
        # 值为一个列表，第一个元素存储消耗时间，第二个维度存储
        self.h=np.full((number_of_nodes,number_of_nodes),fill_value=0.0)
        for i,nodei in enumerate(self.nodes_list):
            for j,nodej in enumerate(self.nodes_list):
                if nodej in G.successors(nodei):
                    self.h[i][j]=1/(G[nodei][nodej]["length"]/1000)
        non_zero_index=(self.h!=0)
        maxx=np.max(self.h)
        self.h[non_zero_index]/=maxx
        return self.h
    def _base_aco(self):
        # if "mode"=="random":
        #     self.taui=self.tau = self.init_tau()
        #     self.h=self.init_hq()
        # else:
        #     self.recover_h_tau()
        #     self.init_protau()
        # self.init_h_ped()
        fit_list=[]
        for self.iter in tqdm(range(self.max_iter)):
            begain=time.time()
            self.build_ant2(self.ants_num)
            end=time.time()
            self.time_rec[0].append(end-begain)
            for i in range(len(self.ants)):
                self.ants[i].ori_path=ACO_optimizer.Conversion(self.ants[i].path,self.index_to_node)
            begain=time.time()
            if len(self.ants)!=0:
                self.select_ants()
            fit=self.evaluation()
            if len(fit)==0:
                continue
            # 转换为 NumPy 矩阵
            # fit = fit.cpu().numpy()
            end=time.time()
            self.time_rec[1].append(end-begain)
            if len(fit)==0:
                continue
            best_index=np.argmin(fit)
            cur_best_ant=self.ants[best_index]
            self.iter_best.append(cur_best_ant)
            self.his_best_fit.append(fit[best_index])
            if self.best_ant is None or fit[best_index]<self.best_fit:
                self.best_ant=cur_best_ant
                self.best_fit=fit[best_index]
            self.update_base_aco(fit)
            print(f"base_aco:the best fit{self.best_fit}")
            self.ants=[]
   
        write(f'LACO/modeule_time_base.csv',[sum(i) for i in self.time_rec])
        write(f'LACO/Conver_diversity_base.csv',self.his_best_fit)
        return self.best_ant,self.best_fit
    def build_ant2(self,ants_num):
        count=0 
        while count<ants_num: 
            count+=1
            ant=Ant(self.args.req.init,self.node_to_index[self.req.s],self.node_to_index[self.req.des])
            last_cs_index=0
            while ant.path[-1]!=self.node_to_index[self.req.des]:
                next_index=self.select_next_index(ant)
                if next_index is None:
                    break
                if self.index_to_node[next_index] in self.net.CS_list:
                    co=self.rule_CO(ACO_optimizer.Conversion(ant.path[last_cs_index:next_index],self.index_to_node))
                    ant.move_to_next_index(next_index,co)
                else:
                    ant.move_to_next_index(next_index)
                # 更新本地信息素
            if ant.path[-1]==self.node_to_index[self.req.des]:
                self.ants.append(ant)
    def rule_CO(self,path):
        if len(path)==0:
            return CO(1,0)
        cur=path[0]
        total_E=0
        v=60
        for next in path[1:]:
            e=self.net.cal_energy(cur,next, v=v)
            total_E+=e
            cur=next
        co=CO(1,total_E/self.vehicle.Q)
        return co
    def select_next_index(self,ant):
        tau=self.tau
        G=self.net.G
        cur=ant.path[-1]
        neighbors=ACO_optimizer.Conversion(list(G.successors(self.index_to_node[cur])),self.node_to_index)
        # 确保不产生环路
        select_neighbor = [item for item in neighbors if item not in ant.path]
        if len(select_neighbor)==0 :
            return None
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
    def update_base_aco(self,fit):
        self.tau=(1-self.pho)*self.tau
        for f,ant in zip(fit,self.ants):
            path=ant.path
            cur=path[0]
            for next in path[1:]:
                self.tau[cur][next]+=1/f
        return self.tau
    def select_ants(self):
        unit_set=[]
        feasible_ant=[]
        for i,ant in enumerate(self.ants):
            unit=eval_unit(ant.ori_path,i,self.req.t0)
            unit_set.append(unit)
        self.model.eval()
        loader = DataLoader(Datadict(unit_set), batch_size=50,
                                    collate_fn=lambda x: self.collate_func(x),
                                    shuffle=False, pin_memory=True)
        tqdm_loader = tqdm(loader,mininterval=3, disable=True)
        # 遍历数据加载器，获取输入特征和真实数据
        Status_Set = to_var(torch.tensor([]),self.args.device)
        for step, (feature, truth_data) in enumerate(tqdm_loader):
            with torch.no_grad():
                feature = to_var(feature, self.args.device)
                status=self.model(feature)
                Status_Set=torch.cat((Status_Set,status))
        pr_fea=[]
        all_status=[]
        epsilon=0.1

        for i,multi_status in enumerate(Status_Set):   
            split_mult=[1 for _ in range(4)]    
            multi_status=F.softmax(multi_status,dim=1)    
            for k,status in enumerate(multi_status):      
                for j,s in enumerate(status):
                    split_mult[j]*=(s+epsilon)
            split_mult=[i.item() for i in split_mult]
            all_status.append(split_mult)
                # sum_pr*=(s+epsilon)
        sum_pro=np.sum(all_status)
        for i,status in enumerate(Status_Set):
            split_mult=all_status[i]
            pro=(split_mult[2]+split_mult[3])/sum_pro
            if pro<epsilon:
                feasible_ant.append(self.ants[i])
        self.ants=feasible_ant
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
        # self.write_scale(X_A_Set)
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
        


