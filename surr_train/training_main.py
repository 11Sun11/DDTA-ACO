import os
import sys
import shutil
from tqdm import tqdm
import json
import torch
from torch import optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from surr_train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.metric import calculate_metrics
from utils.util import to_var
from utils.util import StandardScaler2
import time
def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)
        outputs, _ = model(features, args)
        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([pre2, inds]))


def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    load_datadoct_pre(args)
    data_loaders, scaler = load_datadict(args)
    args.scaler = scaler
    # 用于对时间的预测输出做归一化处理
    model = create_model(args)
    loss_func = create_loss(args)
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    model = model.to(args.device)
    print(f'loss function: {args.loss}')
    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    print(model)
    # 训练
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)
        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError()
        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args)
    elif args.mode == 'resume':
        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()
        final_model = torch.load(os.path.join(model_folder, 'final_model.pkl'), map_location=args.device)
        start_epoch = final_model['epoch']
        model.load_state_dict(final_model['model_state_dict'], strict=False)
        optimizer.load_state_dict(final_model['optimizer_state_dict'])
        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args, start_epoch=start_epoch)
    print(args.loss)
    print(args.model_config)
    print(args.data_config)
    print(model_folder)
    
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['model_state_dict'], strict=False)
    test_model(model, data_loaders['test'], args)
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
def get_model(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config
    length_scale,xy_scaler=initial_scaler(args)
    scaler = StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])
    # args.net.scaler = data_scaler
    args.net.xy_scaler = xy_scaler
    args.net.length_scaler = length_scale
    args.scaler=scaler
    model = create_model(args)
    loss_func = create_loss(args)
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['model_state_dict'], strict=False)
    return model
# code/ACO-MulT-TTE/data/save_models/MulT_TTE__chengdu_MulT_TTE/best_model.pkl code/ACO-MulT-TTE/data/save_models/MulT_TTE__chengdu_MulT_TTE/final_model.pkl
# f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'