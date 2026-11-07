import sys
import os
import argparse
from ACO.utility import *
from ACO.Net import *
from ACO.DDTA_ACO import *
from surr_train.training_main import *
from ACO.Draw import *
# 防止陷入局部最优,从信息素入手,检测每一行中信息素的分布状态
from sklearn.preprocessing import StandardScaler
from com.RH import*
from com.TOA import*
from com.Learning_ACO import*
from com.Learning_SA import*
from com.SP import*
from com.EVPRP import*
from com.MIQCP import*
import time
# export TF_ENABLE_ONEDNN_OPTS=0


def ACO_main_repair(args):
    # 初始化汽车参数
    vehicle=EV(args)
    args.vehicle=vehicle
    # 初始化用户需求
    req = Re(args)
    load_datadoct_pre(args)
    args.req=req
    ants_num_list=list(range(20, 80, 10))
    p0_list = q0_list = pho_list = hr_list = np.arange(0.1, 1, 0.2).tolist()
    # 初始化路网
    net=Net(args)
    args.net=net
    args.model=get_model(args)
    print("build model over!")
    optimizer="_basic_aco_diversity5"
    init_method="init"
    optimizers=[]
    optimizers=ACO_optimizer(args,1,optimizer)
    # optimizers.pre_gen_ins()
    get_ex(args)
    # 记得补一个0
    # for args.w in range(21,40):
    for args.w in range(20):
        set_ins(args, args.w)
        # 优化器
        com_fit=[]
        com_runt=[]
        # for optimizers.early_stop in early_stop_list:
        # for optimizers.q0 in q0_list:
        # for optimizers.ants_num in ants_num_list:
        # for optimizers.repair in [True,False]:
        # total_fit=[0 for _ in range(4)]
        # total_runT=[0 for _ in range(4)]
        # iter_list=[0,1,2]
        iter_list=[True,False]
        total_fit=[[] for _ in range(len(iter_list))]
        total_runT=[[] for _ in range(len(iter_list))]
        total_time=[[] for _ in range(len(iter_list))]
        total_cost=[[] for _ in range(len(iter_list))]
        lenn=20
        for _ in range(lenn):
            runT=[]
            Fitn=[]
            time_list=[]
            cost_list=[]
            # for i,optimizers.ant_GS in enumerate([False]):
            # for i,optimizers.CP_GS in enumerate([False]):
            # for i,optimizers.elite_abl in enumerate(iter_list):
            for i,optimizers.elite in enumerate(iter_list):
            # for i, optimizers.update in enumerate([0,1]): 
            # for i,optimizers.repair in enumerate([False]):
            # for i,optimizers.pho in enumerate(pho_list):
            # for i,optimizers.surr in enumerate([False]):  
            # for i,optimizers.update_repair in enumerate([False]):  
            # for i,optimizers.abl_MES_one in enumerate([0,1,2]):  
                # optimizers[i].init(args)
                # optimizers.beta=1-optimizers.alpha
                method = getattr(optimizers, init_method)
                method(args)
                method = getattr(optimizers, optimizer)
                if callable(method):
                    begin=time.time()
                    result=method()  # 执行方法
                    end=time.time()
                    total_fit[i].append(result[1])
                    Fitn.append(result[1])
                    total_time[i].append(result[2])
                    Fitn.append(result[2])
                    total_cost[i].append(result[3])
                    Fitn.append(result[3])
                    total_runT[i].append(end-begin)
                    runT.append(end-begin)
                    # write(f'Repair/Conver_diversity_{optimizers.repair}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'Repair/E_Conver_diversity_{optimizers.repair}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'Repair/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'Pho/Conver_diversity_{optimizers.pho}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'Pho/E_Conver_diversity_{optimizers.pho}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'Pho/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'surr/Conver_diversity_{optimizers.surr}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'surr/E_Conver_diversity_{optimizers.surr}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'surr/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'update_repair/Conver_diversity_{optimizers.surr}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'update_repair/E_Conver_diversity_{optimizers.surr}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'update_repair/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'Elite/Conver_diversity_{optimizers.elite}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'Elite/E_Conver_diversity_{optimizers.elite}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'Elite/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'Ant_GS/Conver_diversity_{optimizers.ant_GS}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'Ant_GS/E_Conver_diversity_{optimizers.ant_GS}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'Ant_GS/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'CP_GS/Conver_diversity_{optimizers.CP_GS}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'CP_GS/E_Conver_diversity_{optimizers.CP_GS}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'CP_GS/iters_{args.data}.csv',[optimizers.iter])
                    # write(f'abl_MES/Conver_diversity_{optimizers.elite_abl}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'abl_MES/E_Conver_diversity_{optimizers.elite_abl}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'abl_MES/iters_{optimizers.elite_abl}_{args.data}.csv',[optimizers.iter])
                    # write(f'abl_MES_one/Conver_diversity_{optimizers.abl_MES_one}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'abl_MES_one/E_Conver_diversity_{optimizers.abl_MES_one}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'abl_MES_one/iters_{args.data}_{optimizers.abl_MES_one}.csv',[optimizers.iter])
                    # write(f'Update/Conver_diversity_{optimizers.update}_{args.data}.csv',optimizers.his_best_fit)
                    # write(f'Update/E_Conver_diversity_{optimizers.update}_{args.data}.csv',optimizers.E_iter_best_fit)
                    # write(f'early_stop/Conver_diversity_{optimizers.early_stop}.csv',optimizers.his_best_fit)
                    # write(f'ants_num/Conver_diversity_{optimizers.ants_num}.csv',optimizers.his_best_fit)
                    # write(f'alpha/Conver_diversity_{format(optimizers.alpha, ".2f")}_{{format(optimizers.beta, ".2f")}}.csv',optimizers.his_best_fit)
                    # write(f'alpha/E_Conver_diversity_{format(optimizers.alpha, ".2f")}_{{format(optimizers.beta, ".2f")}.csv',optimizers.E_iter_best_fit)
                    # write(f'alpha_beta/Conver_diversity_{format(optimizers.alpha, ".2f")}_{{format(optimizers.beta, ".2f")}}.csv',optimizers.his_best_fit)
                    # write(f'alpha_beta/E_Conver_diversity_{format(optimizers.alpha, ".2f")}_{{format(optimizers.beta, ".2f")}.csv',optimizers.E_iter_best_fit)
                    # write(f'q0/Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.his_best_fit)
                    # write(f'q0/E_Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.E_iter_best_fit)
                    write(f'Supple_Elite/Conver_diversity_{optimizers.elite}_{args.data}_split_cost_{args.w}.csv',optimizers.his_best_cost)
                    write(f'Supple_Elite/E_Conver_diversity_{optimizers.elite}_{args.data}_split_cost_{args.w}.csv',optimizers.E_iter_best_cost)
                    write(f'Supple_Elite/Conver_diversity_{optimizers.elite}_{args.data}_split_time_{args.w}.csv',optimizers.his_best_time)
                    write(f'Supple_Elite/E_Conver_diversity_{optimizers.elite}_{args.data}_split_time_{args.w}.csv',optimizers.E_iter_best_time)
                    write(f'Supple_Elite/Conver_diversity_{optimizers.elite}_{args.data}_split_{args.w}.csv',optimizers.his_best_fit)
                    write(f'Supple_Elite/E_Conver_diversity_{optimizers.elite}_{args.data}_split_{args.w}.csv',optimizers.E_iter_best_fit)
                # write("runT_part_optimizers.csv",runT)
                # write("Fitn_part_optimizers.csv",Fitn)
            # write(f"Pho/multi_runT_{args.data}.csv",runT)
            # write(f"Pho/multi_Fitn_{args.data}.csv",Fitn)
            write(f"Supple_Elite/multi_runT_{args.data}_{args.w}.csv",runT)
            write(f"Supple_Elite/multi_Fitn_{args.data}_{args.w}.csv",Fitn)
            write(f"Supple_Elite/multi_time_{args.data}_{args.w}.csv",time_list)
            write(f"Supple_Elite/multi_cost_{args.data}_{args.w}.csv",cost_list)
            # write(f"Elite/multi_runT_{args.data}.csv",runT)
            # write(f"Elite/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"Ant_GS/multi_runT_{args.data}.csv",runT)
            # write(f"Ant_GS/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"CP_GS/multi_runT_{args.data}.csv",runT)
            # write(f"CP_GS/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"abl_MES/multi_runT_{args.data}.csv",runT)
            # write(f"abl_MES/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"abl_MES_one/multi_runT_{args.data}.csv",runT)
            # write(f"abl_MES_one/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"Elite/multi_runT_{args.data}.csv",runT)
            # write(f"Elite/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"MEM/multi_runT_{args.data}.csv",runT)
            # write(f"MEM/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"surr/multi_runT_{args.data}.csv",runT)
            # write(f"surr/multi_Fitn_{args.data}.csv",Fitn)
            # write(f"update_repair/multi_runT_{args.data}.csv",runT)
            # write(f"update_repair/multi_Fitn_{args.data}.csv",Fitn)
        com_fit=[sum(t)/len(t) for t in total_fit]
        com_runt=[sum(t)/len(t) for t in total_runT]
        com_time=[sum(t)/len(t) for t in total_time]
        com_cost=[sum(t)/len(t) for t in total_cost]
        # write(f"ins_t_res_{args.data}.csv",[args.s,args.des,args.time,args.week,args.day,args.req.t_min.item(),args.req.cost_min,com_fit[0],com_runt[0]])
            # write("alpha_beta/runT_optimizers.csv",runT)
            # write("alpha_beta/Fitn_optimizers.csv",Fitn)
        # write("runT_part_optimizers2.csv",runT)
        # 参数调查-pho，信息素蒸发
        # write("pho/Fitn_optimizers.csv",Fitn)
        # write("pho/runT_optimizers.csv",runT)
        # 参数调查-
        # write(f"KCM/runT_optimizers_{name}.csv",com_fit)
        # write(f"KCM/Fitn_optimizers_{name}.csv",com_runt)
        # write(f"Repair/runT_optimizers_{optimizers.repair}_{args.data}.csv",com_fit)
        # write(f"Repair/Fitn_optimizers_{optimizers.repair}_{args.data}.csv",com_runt)
        # write(f"surr/runT_optimizers_{optimizers.surr}_{args.data}.csv",com_fit)
        # write(f"surr/Fitn_optimizers_{optimizers.surr}_{args.data}.csv",com_runt)
        # write(f"update_repair/runT_optimizers_{optimizers.surr}_{args.data}.csv",com_fit)
        # write(f"update_repair/Fitn_optimizers_{optimizers.surr}_{args.data}.csv",com_runt)
        # write(f"MEM/runT_optimizers_{optimizers.repair}_{args.data}.csv",com_fit)
        # write(f"MEM/Fitn_optimizers_{optimizers.repair}_{args.data}.csv",com_runt)
        # write("hr/Fitn_optimizers.csv",Fitn)
        # write("hr/Fitn_optimizers.csv",Fitn)
        # write("hr/runT_optimizers.csv",runT)
        # 参数调查-蚂蚁数量
        # write("ants_num/Fitn_optimizers.csv",Fitn)
        # write("ants_num/runT_optimizers.csv",runT)
        # 参数调查-选择迭代最优or全局最优进行信息素更新
        # write("q0/Fitn_optimizers.csv",Fitn)
        # write("q0/runT_optimizers.csv",runT)
        # 参数调查-选择下一节点的概率
        # write("p0/Fitn_optimizers.csv",Fitn)
        # write("p0/runT_optimizers.csv",runT)
        # 参数调查-Fitn
        # write("early_stop/Fitn_optimizers.csv",Fitn)
        # write("early_stop/runT_optimizers.csv",runT)
        # 参数调查-Elite
        # write(f"Elite/Fitn_optimizers_{args.data}.csv",com_fit)
        # write(f"Elite/runT_optimizers_{args.data}.csv",com_runt)
        # write("CP_GS/Fitn_optimizers.csv",com_fit)
        # write("CP_GS/runT_optimizers.csv",com_runt)
        # write(f"abl_MES/runT_optimizers_{args.data}.csv",com_runt)
        # write(f"abl_MES/Fitn_optimizers_{args.data}.csv",com_fit)
        # write(f"abl_MES_one/runT_optimizers_{args.data}.csv",com_runt)
        # write(f"abl_MES_one/Fitn_optimizers_{args.data}.csv",com_fit)
        # write("Ant_GS/Fitn_optimizers.csv",com_fit)
        # write("Ant_GS/runT_optimizers.csv",com_runt)
        # 参数调查-Repair
        # write(f"Pho/Fitn_optimizers_{args.data}.csv",com_fit)
        # write(f"Pho/runT_optimizers_{args.data}.csv",com_runt)
        write(f"Supple_Elite/multi_runT_{args.data}.csv",com_runt)
        write(f"Supple_Elite/multi_Fitn_{args.data}.csv",com_fit)
        write(f"Supple_Elite/multi_time_{args.data}.csv",com_time)
        write(f"Supple_Elite/multi_cost_{args.data}.csv",com_cost)
        # write("Fitn_part_optimizers.csv",Fitn)
        # write("runT_optimizers_diffpart.csv",runT)
        # write("Fitn_optimizers_diffpart.csv",Fitn) 
        # 参数调查-short-inc
        # write("short_inc/Fitn_optimizers.csv",Fitn)
        # write("short_inc/runT_optimizers.csv",runT)
        # write("alpha/Fitn_optimizers.csv",Fitn)
        # write("alpha/runT_optimizers.csv",runT)
        #消融实验-自适应
        # write("adapt/Fitn_optimizers.csv",Fitn)
        # write("adapt/runT_optimizers.csv",runT)
        # time_list.append(runT)
        # iter_list.append(optimizers[i].iter)
        # Convergence_curve(optimizers[i].his_best_fit,name=str(i))    
        # print(f"best_ant: {optimizers[i].best_ant}  best_fit{optimizers[i].best_fit}")
        # if i%10==0:
        #     write('output.csv',iter_list)
        #     iter_list=[]
        #     write('time_rec.csv',time_list)
        #     time_list=[]
        # 调查蚁群算法种类对算法的影响
        # write("aco_type/Fitn_optimizers.csv",Fitn)
        # write("aco_type/runT_optimizers.csv",runT)
    return optimizers.best_ant,optimizers.best_fit
def com_ACO(args):
    vehicle=EV(args)
    args.vehicle=vehicle
    # 初始化用户需求
    req = Re(args)
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    args.req=req
    net = Net(args)
    args.net=net
    args.model=get_model(args)
    # gen_instances(args)
    # net.gen_ins()
    # 45-65
    # args.sd_ins=get_instance(args,"sd")
    # args.ins=get_instance(args,"example")
    print("build model over!")
    optimizer="_basic_aco_diversity5"
    # init_method=["init","part_init","part_init2"]
    init_method="init"
    laco=Learning_ACO(args)
    lsa=Learning_SA(args)
    optimizers=[]
    optimizers=ACO_optimizer(args,0,optimizer)
    best_ant_list=[]
    best_fit_list=[]
    get_ex(args)
    # 记得补一个0
    for args.w in  range(20,41):
        if args.w==26 and args.data=="porto":
            continue
        set_ins(args, args.w)
        runT_list=[]
        best_fit_list=[]
        Succ_list=[]
        # 优化器
        method = getattr(optimizers, init_method)
        optimizers.beta=1-optimizers.alpha
        optimizers.mode="mm"
        begin=time.time()
        paths=[nx.shortest_path(args.net.G, source=args.req.s, target=args.req.des)]
        end=time.time()
        gen_time=end-begin
        begin=time.time()
        sols=EVPRP_main(args)
        if len(sols)==0:
            best_fit_list.append(10000)
        else:
            for ant in sols:
                ant.recover(args.net)
            sol,fits=optimizers.eval_no_repair(sols)
            best_value=min(fits)
            if isinstance(best_value,torch.Tensor):
                best_value=best_value.cpu().item()
            best_fit_list.append(best_value)
        end=time.time()
        runT_list.append(end-begin)
        sol=RH(args, paths, [1,3,2])
        if len(sol)==0:
            best_fit_list.append(10000)
        else:
            for ant in sol:
                ant.recover(args.net)
            sol,fits=optimizers.eval_no_repair(sol)
            best_value=min(fits)
            if isinstance(best_value,torch.Tensor):
                best_value=best_value.cpu().item()
            best_fit_list.append(best_value)
        end=time.time()
        runT_list.append(end-begin+gen_time)
        begin=time.time()
        sol=TOA(paths,args)
        if len(sol)==0:
            best_fit_list.append(10000)
        else:
            for ant in sol:
                ant.recover(args.net)
            sol,fits=optimizers.eval_no_repair(sol)
            best_value=min(fits)
            if isinstance(best_value,torch.Tensor):
                best_value=best_value.cpu().item()
            best_fit_list.append(best_value)
        end=time.time()
        runT_list.append(end-begin+gen_time)
        begin=time.time()
        sol=SCSHT_main(args)
        if len(sol)==0:
            best_fit_list.append(10000)
        else:
            for ant in sol:
                ant.recover(args.net)
            sol,fits=optimizers.eval_no_repair(sol)
            best_value=min(fits)
            if isinstance(best_value,torch.Tensor):
                best_value=best_value.cpu().item()
            best_fit_list.append(best_value)
        end=time.time()
        runT_list.append(end-begin) 
        unfeasible=0
        result_set=[]
        time_set=0
        for _ in range(20):
            
            laco.tau=optimizers.taui
            # laco.h=optimizers.h
            laco.init()
            begin=time.time()
            sol, fit_best=laco._base_aco()
            if sol is None:
                best_value=10000
                unfeasible+=1
            else:
                if not isinstance(sol,list):
                    sol=[sol]
                for ant in sol:
                    ant.recover(args.net)
                sol,fits=optimizers.eval_no_repair(sol)
                if len(fits)==0:
                    best_value=10000
                    unfeasible+=1
                else:
                    best_value=min(fits)
                    if isinstance(best_value,torch.Tensor):
                        best_value=best_value.cpu().item()  
                    if best_value>=1000:
                        unfeasible+=1
                    else:
                        result_set.append(best_value)
            end=time.time()
            
            time_set+=end-begin
        write(f"com/laco_Fitn_{args.data}.csv",result_set)
        if unfeasible!=20:
            best_fit_list.append(sum(result_set)/len(result_set))
            runT_list.append(time_set/20)
            Succ_list.append(1-unfeasible/20)
        else:
            best_fit_list.append(10000)
            runT_list.append(time_set/20)
            Succ_list.append(0)
        unfeasible=0
        result_set=[]
        time_set=0
        for _ in range(20):
         
            begin=time.time()
            sol, fit_best=lsa.SA(optimizers)
            if sol is None:
                best_value=10000
                unfeasible+=1
            else:
                if not isinstance(sol,list):
                    sol=[sol]
                for ant in sol:
                    ant.recover(args.net)
                sol,fits=optimizers.eval_no_repair(sol)
                if len(fits)==0:
                    best_value=10000
                    unfeasible+=1
                else:
                    best_value=min(fits)
                if isinstance(best_value,torch.Tensor):
                    best_value=best_value.cpu().item()
                if best_value==10000:
                    unfeasible+=1
                else:
                    result_set.append(best_value)    
            end=time.time()
            time_set+=end-begin
        if unfeasible!=20:
            best_fit_list.append(sum(result_set)/len(result_set))
            runT_list.append(time_set/20)
            Succ_list.append(1-unfeasible/20)
        else:
            best_fit_list.append(10000)
            runT_list.append(time_set/20)
            Succ_list.append(0)
        write(f"com/lsa_Fitn_{args.data}.csv",result_set)
            # args.net.Visualize_sol(optimizers.best_ant,optimizers.traff_tau,optimizers.iter_best)
            # write(f'q0/Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.his_best_fit)
            # write(f'q0/E_Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.E_iter_best_fit)
        # write(f"com/Fitn_optimizers_{args.data}.csv",best_fit_list)
        # write(f"com/runT_optimizers_{args.data}.csv",runT_list)
        write(f"com/Succ_optimizers_{args.data}.csv",Succ_list)
            # args.net.Visualize_sol(optimizers.best_ant,optimizers.traff_tau,optimizers.iter_best)
            # write(f'q0/Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.his_best_fit)
            # write(f'q0/E_Conver_diversity_{format(optimizers.q0, ".2f")}.csv',optimizers.E_iter_best_fit)
        write(f"com/Fitn_optimizers_other_{args.data}.csv",best_fit_list)
        write(f"com/runT_optimizers_other_{args.data}.csv",runT_list)
    return optimizers.best_ant,optimizers.best_fit

def ran_gen(args):
    args.week=random.randint(0,6)
    args.day=random.randint(0,356)
    args.s,args.des=random.sample(args.net.POI_list,2)
    args.req.par_min(args.net.G_B,args.vehicle)
def gen_instance1(args):
    # 随机生成100个实例
    net=args.net
    ins=[]
    for i in range(100):
        week=random.randint(0,7)
        day=random.randint(0,356)
        s,des=random.sample(net.POI_list,2)
        paths = find_all_paths(net.G,s, des)
        if paths is not None and len(paths)>3:
            ins.append(s,des,week,day)
    with open("instances.csv","w") as file:
        file.write(ins)
if __name__ == '__main__':
    # regen_instance()
    parser = argparse.ArgumentParser()
    # TTE Model
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='MulT_TTE', choices=['MulT_TTE'])
    parser.add_argument('-M', '--mode', type=str, default='train', help='input the process mode', choices=['train', 'resume', 'test'])
    # parser.add_argument('-d', '--dataset', type=str, default='porto_MulT_TTE', help='input the dataset name', choices=['chengdu_MulT_TTE','porto_MulT_TTE'])
    # parser.add_argument('-da', '--data', type=str, default='porto', help='input the dataset name', choices=['chengdu','porto'])
    parser.add_argument('-d', '--dataset', type=str, default='chengdu_MulT_TTE', help='input the dataset name', choices=['chengdu_MulT_TTE','porto_MulT_TTE'])
    parser.add_argument('-da', '--data', type=str, default='chengdu', help='input the dataset name', choices=['chengdu','porto'])
    parser.add_argument('-i', '--identify', type=str, help='input the specific identification information', default='')
    parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cuda:0")
    # parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cpu")
    parser.add_argument('-o', '--optim', type=str, help='input the chosen optimization function', default="Adam", choices=['Adam'])
    parser.add_argument('-c', '--loss', type=str, help='input the chosen loss function', default="smoothL1", choices=['rmse','mse', 'mape', 'mae', 'smoothL1'])
    parser.add_argument('-cl', '--loss_val', type=float, help='input the specific parameter for smoothL1',  default=300.)
    parser.add_argument('-e', '--epochs', type=int, help='input the max epochs',default=50)
    parser.add_argument('-b', '--beta', type=float, help='input the learning preference between MSG and TTE (the bigger the value, the more preference for TTE.)',default=0.7)
    parser.add_argument('-l', '--lr', type=float, help='input the initial learning rate',default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, help='input the weight decay of optimization',default=0.00001)
    parser.add_argument('-p', '--patience', type=int, help='input the max iteration times of early stop',default=10)
    parser.add_argument('-r', '--mask_rate', type=float, help='input the mask rate of segments in a trajectory', default=0.4)
    # ACO的参数
    parser.add_argument('-hr', '--hr', type=float, help='The influence ratio of heuristic information on the transfer matrix', default=0.4)
    parser.add_argument('-pr', '--pr', type=float, help='The influence ratio of pheromones on the transfer matrix', default=0.4)
    parser.add_argument('-pho', '--pho', type=float, help='Pheromone Volatilization Factor', default=0.1)
    parser.add_argument('-mI', '--max_iter', type=int, help='Maximum number of iterations in ant colony optimization', default=100)
    parser.add_argument('-ants_num', '--ants_num', type=int, help='Maximum number of ants', default=50)
    parser.add_argument('-_alpha', '--_alpha', type=float, help='the probablity of directly select', default=0.8)
    parser.add_argument('-_beta', '--_beta', type=float, help='the probablity of directly select', default=0.2)
    parser.add_argument('-q0', '--q0', type=float, help='the probablity of select best', default=0.8)
    parser.add_argument('-qc', '--qc', type=float, help='the probablity of select best charging option', default=0.5)
    parser.add_argument('-p0', '--p0', type=float, help='the probablity of update by the iterbest and nowbest', default=0.5)
    # 汽车的参数
    parser.add_argument('-Em', '--Emax', type=int, help='The maximum energy of vehicle', default=100)
    parser.add_argument('-P', '--P_a', type=int, help='Auxiliary equipment power', default=450)
    parser.add_argument('-O', '--O', type=list, help='The supported charging technique of vehicle', default=[0,1,2,3])
    parser.add_argument('-Ma', '--Mass', type=float, help='The mass of vehicle', default=1400)
    parser.add_argument('-A', '--A', type=float, help='The Front surface area of vehicle', default=3.912)
    parser.add_argument('-C_d', '--C_d', type=float, help='The friction factor of vehicle', default=0.7)
    parser.add_argument('-C_r', '--C_r', type=float, help='The friction factor of vehicle', default=0.01)
    parser.add_argument('-Q', '--Q', type=float, help='The battery capacity of vehicle', default=100)
    parser.add_argument('-eff1', '--eff1', type=float, help='The frist efficiency parameters', default=0.997)
    parser.add_argument('-eff2', '--eff2', type=float, help='The second efficiency parameters', default=0.808)
    parser.add_argument('-eff3', '--eff3', type=float, help='The third efficiency parameters', default=0.153)
    parser.add_argument('-eta_add', '--eta_add', type=float, help='The consumption conversion efficiency', default=0.76)
    parser.add_argument('-eta_re', '--eta_re', type=float, help='The recovery conversion efficiency', default=1.27)

    # 用户需求的参数
    parser.add_argument('-s', '--s', type=int, help='The source of travel', default=0)
    parser.add_argument('-des', '--des', type=int, help='The destination of travel', default=0)
    parser.add_argument('-W', '--W', type=list,  help='The Weight of objective', default=[0.5, 0.5])
    parser.add_argument('-t0', '--t0', type=float, help='The departure time of the trip', default=700)
    parser.add_argument('-minn', '--minn', type=list, help='The minimum SoC value', default=0.2)
    parser.add_argument('-maxx', '--maxx', type=list, help='The maximum SoC value', default=0.8)
    parser.add_argument('-aux', '--aux', type=list, help='Flag to enable auxiliary equipment', default=1)
    parser.add_argument('-t_l', '--t_l', type=float, help='The maximum continuous driving time', default=4)
    parser.add_argument('-init', '--init', type=int, help='The initial energy of vehicle', default=0.6)
    parser.add_argument('-date', '--date', type=int, help='The date of day', default=355)
    parser.add_argument('-week', '--week', type=int, help='The week of day', default=5)
    parser.add_argument('-v', '--v', type=float, help='Expected speed', default=20)
    parser.add_argument('-ph', '--ph', type=float, help='Expected speed', default=1.225)

    # 
    parser.add_argument('-insf', '--insf', type=str, default='data/ins.csv', help='input the dataset name')

    # 路网的参数
    parser.add_argument('-CS_rate', '--CS_rate', type=float, default=0.5, help='The CS rate of the network')

    parser.add_argument('-interval', '--interval', type=int, default=5, help='The interval to construct the upper-layer network')
    parser.add_argument('-early_stop', '--early_stop', type=int, default=10, help='The interval to construct the upper-layer network')
    parser.add_argument('-data_path', '--data_path', type=str, default='./code/DDTA-ACO/data/', help='Data files')
    parser.add_argument('-ins_path', '--ins_path', type=str, default='./code/DDTA-ACO/Draw/data/ran_ins.csv', help='Data files')
    parser.add_argument('-write_path', '--write_path', type=str, default='./code/DDTA-ACO/Draw/data/', help='Data files')
    # parser.add_argument('-ins_path', '--ins_path', type=str, default='./code/DDTA-ACO/data/porto/Net/ran_ins.csv', help='Data files')
    parser.add_argument('-edge_path', '--edge_path', type=str, default='./code/DDTA-ACO/data/chengdu/network-chengdu/edges.csv', help='Node files')
    parser.add_argument('-node_path', '--node_path', type=str, default='./code/DDTA-ACO/data/chengdu/network-chengdu/nodes.csv', help='The files of edges')
    # parser.add_argument('-edge_path', '--edge_path', type=str, default='./code/DDTA-ACO/data/porto/network-porto/edges.csv', help='Node files')
    # parser.add_argument('-node_path', '--node_path', type=str, default='./code/DDTA-ACO/data/porto/network-porto/nodes.csv', help='The files of edges')
    parser.add_argument('-CS_Set', '--CS_Set', type=list, default='[]', help='Configuration parameters for charging stations')
    parser.add_argument('-res_path', '--res_path', type=str, default='./code/DDTA-ACO/Draw/data/', help='Configuration parameters for charging stations')

    global args
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    print(args.model)
    print(args.dataset)  
    args.res_path=args.res_path+args.data+"/res"
    args.CS_Set = read_CS("./code/DDTA-ACO/data/output_深圳.xlsx")
    ACO_main_repair(args)
    sys.exit(0)

def initial_scaler(args):
    if "porto" in args.dataset:
        # 手动设置数据标准化器的均值和标准化
        scaler = StandardScaler()
        scaler.fit([[0, 0]])
        scaler.mean_ = [107.497195, 3010.37456]
        scaler.scale_ = [131.102877, 2750.78118]
        scaler2 = StandardScaler()
        scaler2.fit([[0, 0, 0, 0]])
        scaler2.mean_ = [-8.62247695, 41.15923239, -8.62256569, 41.15929004]
        scaler2.scale_ = [0.02520552, 0.01236445, 0.02526226, 0.01242564]
    elif "chengdu" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0,0]])
        scaler.mean_ = [188.285260, 3969.52982]
        scaler.scale_ = [206.040346, 3658.76429]
        scaler2 = StandardScaler()
        scaler2.fit([[0,0,0,0]])
        scaler2.mean_ = [104.06379941,  30.65844312, 104.06381633,  30.65845601]
        scaler2.scale_ = [0.03480474, 0.02717924, 0.03484908, 0.02719959]
    return scaler,scaler2
                  