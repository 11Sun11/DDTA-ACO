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
from utils.prepare import*
import time
from models.MLP import*
from com.Learning_SA import*
def ACO_main(args):
    pro=[0.8,0.1,0.1]
    # train_LinearRegression(args)
    # load_datadoct_pre(args)
    # run_MLP(args)
    # get_length(args) 
    # new_data=gen_status_data(args)
    # split_dataset(new_data,pro)
    # 初始化汽车参数
    vehicle=EV(args)
    args.vehicle=vehicle
    # 初始化用户需求
    req = Re(args)
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # with open(args.insf,"r") as file:
    #     reader = csv.reader(file)
    #     data = [row for row in reader]
    #     args.week,args.day,args.s,args.des=data[0]
    args.req=req
    # stop_iter_list=list(range(4, 14, 2))
    # ants_num_list=list(range(30, 80, 10))
    # p0_list=q0_list=pho_list=hr_list=list(range(0.2, 0.8, 0.2))
    # 初始化路网
    net=Net(args)
    # net.gen_ins()
    args.sd_ins=get_instance(args,"sd")
    args.ins=get_instance(args,"example")
    random.shuffle(args.ins)
    args.net=net
    args.model=get_model(args)
    print("build model over!")
    # gen_instance(args)
    # print("begin build model!")
    iter_list=[]
    time_list=[]
    # optimizer_method=["_basic_aco_diversity5","_basic_aco_diversity4"]
    optimizer_method=["_basic_aco_diversity5","_basic_aco_diversity5","_basic_aco_diversity5"]
    # init_method=["init","part_init","part_init2"]
    init_method="init"
    # gen_instances(args)
    optimizers=[]
    # for args.elite in [True,False]
    # for args.repair in [True,False]
    # for args.q0 in q0_list:
    # for args.ants_num in ants_num_list:
    # for args.early_stop in early_stop_list:
    # for args.hr in q0_list:
        # args.pr=1-args.hr
    # for args.q0 in q0_list:
    # for args.q0 in q0_list:
    for optimizer in optimizer_method:
        optimizers.append(ACO_optimizer(args,0,optimizer))
    for sd,example in zip(args.sd_ins,args.ins):
        runT=[]
        Fitn=[]
        args.t_set=False
        args.s,args.des=int(sd[0]),int(sd[1])
        args.req.s,args.req.des=int(sd[0]),int(sd[1])
        # args.req.s=args.net.edge_list[args.s][0]
        # args.req.des=args.net.edge_list[args.des][1]
        # if args.net.revise_by_SD(args)==-1 or (not nx.has_path(args.net.G_B, args.req.s, args.req.des)):
        #      continue
        args.req.par_min(args.net.G_B,args.vehicle)
        args.target_time=int(example[2])
        args.week=int(example[3])
        args.day=int(example[4])
        args.req.t0=args.time=int(example[5])
        # 优化器
        for i,optimizer in enumerate(optimizer_method):
        # for self.elite in [True,False]:
            # optimizers[i].init(args)
            # optimizers[i].part_init(args)
            # optimizers[i].part_init2(args)
            method = getattr(optimizers[i], init_method)
            method(args)
            method = getattr(optimizers[i], optimizer)
            if callable(method):
                begin=time.time()
                result=method()  # 执行方法
                end=time.time()
                Fitn.append(result[1])
                runT.append(end-begin)
                args.net.Visualize_sol(optimizers[i].best_ant,optimizers[i].traff_tau,optimizers[i].iter_best)
                write(f'modeule_time_diffpart_rec{i}.csv',[sum(i)/runT[-1] for i in optimizers[i].time_rec])
                write(f'com_time_diff.csv',[sum(i) for i in com_time])
                write(f'Conver_diversity_diffpartinit{i}.csv',optimizers[i].his_best_fit)
        # write("runT_part_optimizers.csv",runT)
        # write("Fitn_part_optimizers.csv",Fitn)
        # write("runT_part_optimizers2.csv",runT)
        # 参数调查-pho，信息素蒸发
        # write("pho/Fitn_optimizers.csv",Fitn)
        # write("pho/runT_optimizers.csv",runT)
        # 参数调查-
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
        # write("early-stop/Fitn_optimizers.csv",Fitn)
        # write("early-stop/runT_optimizers.csv",runT)
        # 参数调查-Elite
        # write("Elite/Fitn_optimizers.csv",Fitn)
        # write("Elite/runT_optimizers.csv",runT)
        # 参数调查-Repair
        # write("Repair/Fitn_optimizers.csv",Fitn)
        # write("Repair/runT_optimizers.csv",runT)
        # write("Fitn_part_optimizers.csv",Fitn)
        write("runT_optimizers_diffpart.csv",runT)
        write("Fitn_optimizers_diffpart.csv",Fitn)
        # 参数调查-short-inc
        
        # time_list.append(runT)
        # iter_list.append(optimizers[i].iter)
        # Convergence_curve(optimizers[i].his_best_fit,name=str(i))    
        # print(f"best_ant: {optimizers[i].best_ant}  best_fit{optimizers[i].best_fit}")
        # if i%10==0:
        #     write('output.csv',iter_list)
        #     iter_list=[]
        #     write('time_rec.csv',time_list)
        #     time_list=[]
    return optimizers[i].best_ant,optimizers[i].best_fit

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
    parser = argparse.ArgumentParser()
    # TTE Model
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='MulT_TTE', choices=['MulT_TTE'])
    parser.add_argument('-M', '--mode', type=str, default='train', help='input the process mode', choices=['train', 'resume', 'test'])
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
    parser.add_argument('-_alpha', '--_alpha', type=float, help='the probablity of directly select', default=0.5)
    parser.add_argument('-_beta', '--_beta', type=float, help='the probablity of directly select', default=0.5)
    parser.add_argument('-q0', '--q0', type=float, help='the probablity of select best', default=0.5)
    parser.add_argument('-qc', '--qc', type=float, help='the probablity of select best charging option', default=0.5)
    parser.add_argument('-p0', '--p0', type=float, help='the probablity of update by the iterbest and nowbest', default=0.5)
    # 汽车的参数
    parser.add_argument('-Em', '--Emax', type=int, help='The maximum energy of vehicle', default=100)
    parser.add_argument('-P', '--P_a', type=int, help='Auxiliary equipment power', default=45)
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
    parser.add_argument('-insf', '--insf', type=str, default='ins.csv', help='input the dataset name')

    # 路网的参数
    parser.add_argument('-CS_rate', '--CS_rate', type=float, default=0.5, help='The CS rate of the network')

    parser.add_argument('-interval', '--interval', type=int, default=5, help='The interval to construct the upper-layer network')
    parser.add_argument('-early_stop', '--early_stop', type=int, default=5, help='The interval to construct the upper-layer network')
    parser.add_argument('-data_path', '--data_path', type=str, default='./code/ACO-MulT-TTE/data/', help='Data files')
    parser.add_argument('-ins_path', '--ins_path', type=str, default='./code/ACO-MulT-TTE/Draw/data/ran_ins.csv', help='Data files')
    parser.add_argument('-edge_path', '--edge_path', type=str, default='./code/ACO-MulT-TTE/data/chengdu/network-chengdu/edges.csv', help='Node files')
    parser.add_argument('-node_path', '--node_path', type=str, default='./code/ACO-MulT-TTE/data/chengdu/network-chengdu/nodes.csv', help='The files of edges')
    parser.add_argument('-CS_Set', '--CS_Set', type=list, default='[]', help='Configuration parameters for charging stations')
    parser.add_argument('-res_path', '--res_path', type=str, default='./code/ACO-MulT-TTE/Draw/data/', help='Configuration parameters for charging stations')

    global args
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    print(args.model)
    print(args.dataset)  
    args.res_path=args.res_path+args.data+"/res"
    args.CS_Set = read_CS("./code/ACO-MulT-TTE/output_深圳.xlsx")
    ACO_main(args)
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
