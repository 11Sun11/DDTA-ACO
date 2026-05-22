import json
import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss, MSELoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.util import StandardScaler2
from models.MulT_TTE import MulT_TTE
from sklearn.cluster import KMeans
import copy

highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}
def get_length(args,phase):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all
    lens=[]
    new_data=[]
    split_time=[]
    infos=[]
    data = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
    for ind, l in enumerate(data):
        t=l[-1]
        # 路段的链接ID
        length=0
        date=l[2:5]
        
        for x in l[1]:
            info = edgeinfo[x]
            length += info[1]
        for x in l[1]:
            info = edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(info[1])
            infot.append(length)
            infot += list(date)
            split_time.append(info[1]/length*t)
            try:
                infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
            except:
                print(info)
            infos.append(infot)
            # highway length sumoflength date3 gps4
    new_data=torch.FloatTensor(np.asarray(infos))
    split_time=torch.Tensor(split_time)
    return new_data,split_time
def coll_func_linear(data,info_all):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all
    new_data=[]
    split_time=[]
    infos=[]
    # data = np.load(os.path.join(args.absPath,args.data_config['data_dir'], "train" + '.npy'), allow_pickle=True)
    for ind, l in enumerate(data):
        t=l[-1]
        # 路段的链接ID
        length=0
        date=l[2:5]
        for x in l[1]:
            info = edgeinfo[x]
            length += info[1]
        for x in l[1]:
            info = edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(info[1])
            infot.append(length)
            infot += list(date)
            split_time.append([info[1]/length*t])
            try:
                infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
            except:
                print(info)
            infos.append(infot)
            # highway length sumoflength date3 gps4
    new_data=np.asarray(infos)
    new_data[:, 1:3] = scaler.transform(new_data[:, 1:3])
    new_data[:, 6:10] = scaler2.transform(new_data[:, 6:10])
    new_data=torch.FloatTensor(new_data)
    split_time=torch.Tensor(split_time)
    return new_data,split_time
def gen_status_data(args):
    data=np.empty((0, 6))
    time = []
    lens = []
    phases = ['train', 'val', 'test']
    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        data=np.vstack((data, tdata))
        # data+= tdata
        # lens,length=get_length(args,phase)
        time+=[d[-1] for d in tdata]
    time=torch.Tensor(time)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(np.asarray(time).reshape(-1,1))
    # 获取聚类标签
    labels = kmeans.labels_
    df = pd.DataFrame(time)
    df['Cluster'] = labels
    # 根据聚类标签分组并计算平均值
    mean_values = df.groupby('Cluster').mean()
    class_to_state = sorted(mean_values.index, key=lambda i: mean_values.loc[i,0])
    traff_label=[]
    for label in labels:
        traff_label.append(class_to_state[label])
    traff_label=np.asarray(traff_label)
    new_data=np.concatenate((data[:,:-1], traff_label.reshape(-1,1)), axis=1)
    return new_data
def gen_segment_data(args):
    data=np.empty((0, 6))
    time = []
    lens = []
    phases = ['train', 'val', 'test']
    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        data=np.vstack((data, tdata))
        # data+= tdata
        lens=get_length(tdata)
        time+=[d[-1] for d in tdata]
    time=torch.Tensor(time)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(np.asarray(time).reshape(-1,1))
    # 获取聚类标签
    labels = kmeans.labels_
    df = pd.DataFrame(time)
    df['Cluster'] = labels
    # 根据聚类标签分组并计算平均值
    mean_values = df.groupby('Cluster').mean()
    class_to_state = sorted(mean_values.index, key=lambda i: mean_values.loc[i,0])
    traff_label=[]
    for label in labels:
        traff_label.append(class_to_state[label])
    traff_label=np.asarray(traff_label)
    new_data=np.concatenate((data[:,:-1], traff_label.reshape(-1,1)), axis=1)
    return new_data
def split_dataset(data,proportion):
    np.random.shuffle(data)
    sum_number=len(data)
    split_data=dict()
    data_name=["train","test","valid"]
    pre=0
    aft=0
    for i,name in enumerate(data_name):
        aft+=int(proportion[i]*sum_number)
        split_data[name]=data[int(pre):int(aft)]
        np.save(f'data_{name}.npy', split_data[name])
        pre=aft
    return 

# mlm任务的输入link index中需要预测的值不能是本身，否则产生信息泄露，TTE_edge_new_data_end2end_pre更正为TTE_edge_new_data_end2end
def MulT_TTE_collate_func(data, args, info_all):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all
    # 提取时间信息
    time = torch.Tensor([d[-1] for d in data])
    linkids = []
    dateinfo = []
    inds = []

    for ind, l in enumerate(data):
        # 路段的链接ID
        linkids.append(np.asarray(l[1]))
        dateinfo.append(l[2:5])
        inds.append(l[0])
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16)
    def info(xs, date):
        infos = []
        length = 0
        for x in xs:
            info = edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            infot += list(date)
            try:
                infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
            except:
                print(info)
            infos.append(np.asarray(infot))
            # highway length sumoflength date3 gps4

        return infos

    con_links = np.concatenate([info(b, dateinfo[ind]) for ind, b in enumerate(linkids)], dtype='object')
    # lens[:,None]表示一个广播操作，增加一个维度
    # np.arrange(lens.max())表示生成一个从0到lens中的最大值的一个长度为n的序列
    mask = np.arange(lens.max()) < lens[:, None]
    # 创建一个指定形状的np数组（mask.shape,10）
    padded = np.zeros((*mask.shape, 1+2+3+4), dtype=np.float32) #最后一个数是segment embedding维度，需要依据不同的维度更换
    # 按照预定义的缩放器进行缩放
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 6:10] = scaler2.transform(con_links[:, 6:10])
    # 在掩码为1的地方填充相应特征
    padded[mask] = con_links
    rawlinks = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    rawlinks[mask] = np.concatenate(linkids)
    # 接受token和掩码比例生成随机掩码
    def random_mask(tokens: np.array, rate: float):
        # 生成随机数组表示需要被掩码的链接ID
        replaces = np.where(np.random.random(len(tokens)) <= rate)[0]
        # 初始化标签数组，默认填充为-100
        labels = np.full(len(tokens),dtype=np.int16, fill_value=-100)
        # 复制以避免直接修改原始数组
        tokens = tokens.copy()
        labels[replaces] = tokens[replaces]
        # 将被掩码值的边替换为边的数量+1
        tokens[replaces] = np.asarray([args.data_config['edges'] + 1] * len(replaces))   # 此处直接赋值会改变dataset原始值，应该考虑采用深拷贝复制一份新数组再更改
        return labels, tokens
    mask_label_tmp = []
    sub_input_tmp = []
    for k in linkids:
        # 生成随机掩码，掩码矩阵和标签
        tmp1, tmp2 = random_mask(k, rate=args.mask_rate)
        mask_label_tmp.append(tmp1)
        sub_input_tmp.append(tmp2)
    mask_label = np.full(mask.shape, dtype=np.int16, fill_value=-100)
    mask_label[mask] = np.concatenate(mask_label_tmp)
    linkindex = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    linkindex[mask] = np.concatenate(sub_input_tmp)
    mask_encoder = np.zeros(mask.shape, dtype=np.int16)
    mask_encoder[mask] = np.concatenate([[1]*k for k in lens])
    return {'links':torch.FloatTensor(padded), 'lens':torch.LongTensor(lens), 'inds': inds, 'mask_label': torch.LongTensor(mask_label),
            "linkindex":torch.LongTensor(linkindex), 'rawlinks': torch.LongTensor(rawlinks),'encoder_attention_mask': torch.LongTensor(mask_encoder)}, time

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        elif isinstance(dataset[0][1], list):
            self.lengths = [len(d[1]) for d in dataset]
        else:
            self.lengths = [d[0]['lens'] for d in dataset]
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size
# 定义一个函数，接受一个参数，用于传递配置和
def load_datadoct_pre(args):
    global info_all
    edgeinfo, nodeinfo, scaler, scaler2 = None, None, None, None
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config
    with open(os.path.join(args.absPath,args.data_config['edges_dir']), 'rb') as f:
        edgeinfo = pickle.load(f)
    with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
        nodeinfo = pickle.load(f)
    
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
    else:
        ValueError("Wrong Dataset Name")

    info_all = [edgeinfo, nodeinfo, scaler, scaler2]
    return info_all


class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

def load_datadict(args):
    data = {}
    loader = {}
    if args.mode == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val', 'test']
    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        data[phase] = tdata
        print(data[phase].shape)
        loader[phase] = DataLoader(Datadict(data[phase]), batch_sampler=BatchSampler(data[phase], args.data_config['batch_size']),
                                       collate_fn=lambda x: eval(args.data_config['collate_fn'])(x, args, info_all),
                                       pin_memory=True)

    loader['test'] = DataLoader(Datadict(data['test']), batch_size=args.data_config['batch_size'],
                                    collate_fn=lambda x: eval(args.data_config['collate_fn' ])(x, args, info_all),
                                    shuffle=False, pin_memory=True)
    return loader.copy(), StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    args.model_config = model_config
    model_config['pad_token_id'] = args.data_config['edges'] + 1
    if "MulT_TTE" in args.model:
        return MulT_TTE(**model_config)

def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            # mse = torch.mean(torch.pow(preds - labels, 2))
            mse = MSELoss(reduction='mean').forward(preds.view(-1), labels)
            return mse
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 0.1))
            return mape
    elif args.loss == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels))
            return mape
    elif args.loss == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            preds = torch.squeeze(preds, 1)
            smoothL1 = SmoothL1Loss(reduction='mean', beta = args.loss_val).forward(preds, labels)
            return smoothL1
    else:
        raise ValueError("Unknown loss function.")
    return loss

def TensorTransform(data):
    features, truth_data=data
    features['links']=torch.FloatTensor(features['links'])
    features['lens']=torch.LongTensor(features['lens'])
    features['mask_label']=torch.LongTensor(features['mask_label'])
    features['linkindex']=torch.LongTensor(features['linkindex'])
    features['rawlinks']=torch.LongTensor(features['rawlinks'])
    features['encoder_attention_mask']=torch.LongTensor(features['encoder_attention_mask'])
    truth_data=torch.Tensor(truth_data)
    return features,truth_data


