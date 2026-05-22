'''
Author: Xin-Can Wang
Date: 2024-12-25 15:33:56
LastEditors: Xin-Can Wang
LastEditTime: 2025-01-21 19:06:23
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/com/Learning_SA.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.prepare import *
from utils.util import *
from ACO.Ant import*
from utils.util import *
from ACO.utility import *
from ACO.DDTA_ACO import*
# 假设有一些路段特征和对应的时间开销
# 特征：路段长度、交通状况等
# 这里使用随机数据作为示例

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(367, 10)
        self.timeembed = nn.Embedding(1441, 20)
        self.gpsrep = nn.Linear(4, 16)
        self.linear = nn.Linear(56, 1)  # 输入特征数为2，输出为1

    def forward(self, inputs):
        feature = inputs
        highwayrep= self.highwayembed(feature[ :, 0].long())
        weekrep = self.weekembed(feature[ :, 3].long()) #8
        daterep = self.dateembed(feature[ :, 4].long())  # 367
        timerep = self.timeembed(feature[ :, 5].long()) #1444
        # print(feature[:, :, 6:10])
        gpsrep = self.gpsrep(feature[ :, 6:10])
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        x=torch.cat([feature[..., 1:3], highwayrep, gpsrep,datetimerep], dim=-1)
        # x = x.mean(dim=1)
        return self.linear(x)
def train_LinearRegression(args):
    # 创建模型实例
    model = LinearRegressionModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降
    info_all=load_datadoct_pre(args)
    train_data=np.load(f'./code/ACO-MulT-TTE/data/{args.data}/train.npy', allow_pickle=True)
    num_epochs = 20  # 训练轮数
    # test_data=np.load('./code/ACO-MulT-TTE/data/chengdu/status/data_test.npy', allow_pickle=True)
    # features,truth_data=MulT_TTE_collate_func(train_data, args, info_all)
    # 标准化特征
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # 将标签进行独热编码
    # 创建数据集和数据加载器
    data_loader = DataLoader(Datadict(train_data), batch_sampler=BatchSampler(train_data, args.data_config['batch_size']),
                                       collate_fn=lambda x:coll_func_linear(x, info_all),
                                       pin_memory=True)
   
  
    num_epochs = 1000
    model = model.to(args.device)
    for epoch in range(num_epochs):
        model.train()
        for feature,truth in data_loader:
            # 获取当前批次
            # 前向传播
            truth = to_var(truth, args.device)
            feature = to_var(feature, args.device)
            outputs = model(feature)
            loss = criterion(outputs, truth)

            # 后向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数
        # 确保目标形状为 (batch_size, 1)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), f'data/save_models/LSA_LR/linear_regression_model_{args.data}.pth')
    # 测试模型
    # model.eval()
    # with torch.no_grad():
    #     sample_data = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
    #     predicted = model(sample_data)
    #     print(f'Predicted time cost for features [0.5, 0.3]: {predicted.item():.4f}')
class Learning_SA:
    def __init__(self,args):
        self.net=args.net
        self.model = LinearRegressionModel()
        # 加载保存的状态字典
        self.model.load_state_dict(torch.load(f'data/save_models/LSA_LR/linear_regression_model_{args.data}.pth'))
        # 设置模型为评估模式（如果有 dropout 或 batch normalization 等层）
        self.model = self.model.to(args.device)
        self.model.eval()
        self.args=args
    def SA(self,optimizator:ACO_optimizer):
        fit_list=[]
        args=self.args
        net=args.net
        path=nx.shortest_path(net.G,args.req.s,args.req.des)
        n=len(path)
        init_ant=Ant(args.req.init,net.node_to_index[args.req.s],args.net.node_to_index[args.req.des]) 
        COs=optimizator.gen_CO_by_path(path,0)
        init_ant.CO=COs
        init_ant.ori_path=path
        path=ACO_optimizer.Conversion(path,net.node_to_index)
        init_ant.path=path
        new_fit=[] 
        new_ants=[]
        init_ant,ori_fit=self.eval(init_ant)
        new_ants1=optimizator.LS_shuffle(init_ant)
        new_ants2=optimizator.LS_point(init_ant)
        new_ants3=optimizator.repair_path(init_ant)
        new_ants3=[new_ants3]
        for ant in [new_ants1,new_ants2,new_ants3]:
            if len(ant)!=0:
                new_ants+=ant
        if len(new_ants)!=0:
            new_ants,new_fit=self.eval(new_ants)
            Delta_T=[abs(fit-ori_fit) for fit in new_fit]
            Delta_T=[a for a in Delta_T if a >=0.001]
        else:
            return None,10000
        if len(Delta_T)==0:
            return None,10000
        T_s=min(Delta_T)
        T_f=max(Delta_T)
        Ti=T_s
        N=100
        X_i=init_ant
        X_best=init_ant
        T_best=T_s
        fit_best=ori_fit
        for iter in list(range(N)):
            pro=random.random()
            ls=False
            new_ants=None
            if pro<0.3:
                new_ants=optimizator.LS_shuffle(init_ant,1)
            elif pro<0.6:
                new_ants=optimizator.LS_point(init_ant,1)
            if new_ants is None or len(new_ants)==0:
                new_ants=optimizator.repair_path(init_ant)
            new_ants,fiti=self.eval(new_ants)
            beta_i=(T_s-T_f)/(N+math.sqrt(iter)*n*T_s*T_f)
            Ti=Ti/(1+beta_i*Ti)
            Delta=min(fiti)-ori_fit
            if Delta<=0 :
                init_ant=new_ants
            else:
                try:
                    if abs(Delta/Ti)>10000:
                        pi=1
                    else:
                        pi=math.exp(-Delta/Ti)
                except:
                    print(f"Delta:{Delta} Ti:{Ti}")
                    pi=1
                if random.random()<=pi:
                    init_ant=new_ants
                    ori_fit=fiti
                    if fiti<fit_best:
                        fit_best=ori_fit
                        X_best=new_ants
                        T_best=Ti
            fit_list.append(ori_fit)
            if abs(Ti-T_f)<=0.05:
                break
        write(f"LSA/Conver_diversity_False_{args.data}.csv",fit_list)
        return X_best, fit_best
    def eval(self,ants):
        if not isinstance(ants,list):
            ants=[ants]
        args=self.args
        req=args.req
        device=args.device
        tl=args.t_l
        net=args.net
        G=net.G
        eval_set=[]
        alphas=[req.init for _ in range(len(ants))]
        minn=req.min
        maxx=req.max
        vehicle=args.vehicle
        Q=vehicle.Q
        remove_list=[]
        for i in range(len(ants)):
            if ants[i].ori_path[-1]!=req.des or ants[i].ori_path[0]!=req.s:
                remove_list.append(i)
        ants=[value for i, value in enumerate(ants) if i not in remove_list]
        # for ant in self.ants:
        #     print(f"{self.assert_index(ant.ori_path,ant.CO,self.net.G)} eval1")

        i=0
        if len(ants)==0:
            return [],[]
        Penalty=[0 for _ in range(len(ants))]
        Penalty=torch.tensor(Penalty,device=device, dtype=torch.float32)
        fit=[0 for _ in range(len(ants))]
        time_Cost=[req.t0 for _ in range(len(ants))]
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
                if i>=len(ant.CO):
                    continue
                co=ant.CO[i]
                # if i==0:
                #     print(f"{self.assert_index(path,ant.CO,self.net.G)} eval")
                tem=path[co.index]
                if co.alpha==0:
                    opt=0
                else:
                    opt=int(co.opt+1)
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
                    tc[j]= vehicle.nolinear_charging(
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
                    cost[j]=40 + 1.5 * co.alpha*vehicle.Q
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
                com_t=self.surr_eval(eval_set,self.args)
                
                # 检查最小值
                # if min_value < 0:
                #     print("张量中有负值")
                com_t=torch.tensor(com_t)
                com_t[com_t < 0] = 200000
                com_t[torch.isinf(com_t)] = 20000
            
            # print(com_t.shape) 
            # com_t=com_t.cpu().numpy()
            # 在当前站点判断到达下一节点的能量是否足够，如果不够，那么补充能量（补充后的能量不能超过最大容量，但要大于所需要的能量）
            # 以一定的概率p执行可行范围内的随机修改或者最小修改
            
                if com_t.shape == torch.Size([]):
                    com_t=com_t.unsqueeze(0)
            for q,unit in enumerate(path_set):
                j=unit.index
                t=com_t[q]
                # 获取路段对应的蚂蚁下标
                E=args.net.get_e_by_t(unit.path,t)/1000
                ant=ants[j]
                path=ant.ori_path
                # 存储每一段路径的消耗能量
                # ant.E_set.append(E)
                co=ant.CO[i]
                # 包含了opt==3时E>Q的ke 
                if E>Q :
                    # 该蚂蚁探索出的路径不可行，打标记(极大惩罚值)
                    Penalty[j]+=10000
                time_Cost[j]+=t+tc[j]+tw[j]
                alphas[j]+=co.alpha-E/Q
                # 当E<0时，补充能量至电动汽车装满电
                alphas[j]=min(alphas[j],1)
                if alphas[j]<minn:
                    Penalty[j]+=(minn-alphas[j])/req.min_alpha 
                total_cost[j]+=cost[j]
                        
            # 更新每一个路径对应的时间表
            i=i+1
            # for unit,t,e in zip(eval_set,com_t,cost):
            #     time[unit.index]+=t
            #     cost[unit.index]+=e
        # 验证每只蚂蚁的可行性
        fit=req.com_f(time_Cost,total_cost,Penalty)
        return ants,fit
    
    def col_func(self,sols,info_all):
        time=torch.Tensor([0 for _ in range(len(sols))])
        edgeinfo, nodeinfo, scaler, scaler2 = info_all
        infos=[]
        recflect=[]
        for ind, l in enumerate(sols):
            # 路段的链接ID
            length=0
            date=l.t
            recflect.append(l.index)
            for x in l.path:
                info = edgeinfo[x]
                length += info[1]
            for x in l.path:
                info = edgeinfo[x]
                infot = []
                infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
                infot.append(info[1])
                infot.append(length)
                infot += list(date)
                try:
                    infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
                except:
                    print(info)
                infos.append(infot)
                # highway length sumoflength date3 gps4
        new_data=torch.FloatTensor(np.asarray(infos))
        return new_data,time,recflect
    def coll_func(self,sols):
        time=torch.Tensor([0 for _ in range(len(sols))])
        infos=[]
        recflect=[]
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
        for ind, l in enumerate(sols):
            # 路段的链接ID
            length=0
            date=[self.args.req.week,self.args.req.date,l.t]
            
            for t in range(len(l.path)-1):
                cur=l.path[t]
                next=l.path[t+1]
                length += self.net.G[cur][next]["length"]
            for t in range(len(l.path)-1):

                cur=l.path[t]
                next=l.path[t+1]
                GB_path=self.net.G[cur][next]["path"]
                cur_b=GB_path[0]
                for next_b in GB_path[1:]:
                    recflect.append(ind)
                    infot = []
                    infot.append(highway[self.net.G_B[cur_b][next_b]['highway']] if self.net.G_B[cur_b][next_b]['highway'] in highway.keys() else 0)
                    infot.append(self.net.G_B[cur_b][next_b]["length"])
                    infot.append(length)
                    infot += list(date)
                    infot.append(self.net.G_B.nodes[cur_b]["x"])
                    infot.append(self.net.G_B.nodes[cur_b]["y"])
                    infot.append(self.net.G_B.nodes[next_b]["x"])
                    infot.append(self.net.G_B.nodes[next_b]["y"])
                    infos.append(infot)
                    cur_b=next_b
                    # highway length sumoflength date3 gps4
        new_data=torch.FloatTensor(infos)
        return new_data,recflect
                
    def surr_eval(self,units,args):
        model=self.model
        loader = DataLoader(Datadict(units), batch_size=50,
                                        collate_fn=lambda x: self.coll_func(x),
                                        shuffle=False, pin_memory=True)
        tqdm_loader = tqdm(loader,mininterval=3, disable=True)
        # 遍历数据加载器，获取输入特征和真实数据
        time_Set = [0 for _ in range(len(units))]
        for step, (feature, reflect) in enumerate(tqdm_loader):
            with torch.no_grad():
                feature = to_var(feature, args.device)
                time=model(feature)
                for i,t in enumerate(time):
                    time_Set[reflect[i]]+=t
        return time_Set
    def repair_path(self,ant):
        G=self.net.G
        d=self.req.des
        if not hasattr(ant,'ori_path') or len(ant.ori_path)==0:
            ant.ori_path=ACO_optimizer.Conversion(ant.path,self.index_to_node)
        new_s_index=random.randint(0,len(ant.ori_path)-1)
        new_s=ant.ori_path[new_s_index]
        trys=0
        while (not nx.has_path(G, new_s, d)) and trys<3 and new_s_index>0:
            new_s_index=random.ranint(0,new_s_index)
            new_s=ant.ori_path[new_s_index]
            trys+=1
        if nx.has_path(G, new_s, d):
            path=nx.shortest_path(G,new_s,d)
            COs=self.gen_CO_by_path(path,new_s_index)
            ant.path=ant.path[:new_s_index]+ACO_optimizer.Conversion(path,self.node_to_index)
            ant.ori_path=ant.ori_path[:new_s_index]+path
            pre_index=self.CO_number(ant.ori_path[:new_s_index],self.net.G)
            ant.CO=ant.CO[:pre_index]+COs
            # self.assert_index(ACO_optimizer.Conversion(ant.path,self.index_to_node),ant.CO,self.net.G)
            return ant
        return None
def local_11(G,ant1,ant2):
    path_i=ant1.ori_path
    path_j=ant2.ori_path
    A=random.choice(path_i)
    for i in range(len(path_j)):
        C = path_j[i]
        for j in range(i + 1, len(path_j)):
            D = path_j[j]
            # 获取从 C 到 D 的所有路径
            all_paths = list(nx.all_simple_paths(G, source=C, target=D))
            # 检查每条路径是否经过 A
            for path in all_paths:
                if A in path:
                    new_ant=Ant()
                    path_j[i:j]=path
    return False  # 未找到满足条件的 C 和 D
def find_path_with_A(G, path_j, A):
    for i in range(len(path_j)):
        C = path_j[i]
        for j in range(i + 1, len(path_j)):
            D = path_j[j]
            # 获取从 C 到 D 的所有路径
            all_paths = list(nx.all_simple_paths(G, source=C, target=D))
            # 检查每条路径是否经过 A
            for path in all_paths:
                if A in path:
                    return True  # 找到满足条件的 C 和 D
    return False  # 未找到满足条件的 C 和 D