'''
Author: Xin-Can Wang
Date: 2024-12-24 09:36:47
LastEditors: Xin-Can Wang
LastEditTime: 2025-01-12 15:20:55
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/models/MLP.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
from utils.prepare import *
from utils.util import *
# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(367, 10)
        self.timeembed = nn.Embedding(1441, 20)
        self.gpsrep = nn.Linear(4, 16)
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 第一层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 第二层
        self.fc3 = nn.Linear(hidden_size2, output_size)  # 输出层
        self.relu = nn.ReLU()  # ReLU 激活函数

    def forward(self, inputs):
        feature = inputs['links']
        highwayrep= self.highwayembed(feature[:, :, 0].long())
        weekrep = self.weekembed(feature[:, :, 3].long()) #8
        daterep = self.dateembed(feature[:, :, 4].long())  # 367
        timerep = self.timeembed(feature[:, :, 5].long()) #1444
        # print(feature[:, :, 6:10])
        gpsrep = self.gpsrep(feature[:, :, 6:10])
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        x=torch.cat([feature[..., 1:3], highwayrep, gpsrep,datetimerep], dim=-1)
        # 
        x = self.relu(self.fc1(x))  # 第一层和激活
        x = self.relu(self.fc2(x))  # 第二层和激活
        x = self.fc3(x)  # 输出层
        return x
def run_MLP(args):
    info_all=load_datadoct_pre(args)
    train_data=np.load(f'./code/ACO-MulT-TTE/data/{args.data}/status/data_train.npy', allow_pickle=True)
    num_epochs = 20  # 训练轮数
    learning_rate = 0.001  # 学习率
    # test_data=np.load('./code/ACO-MulT-TTE/data/chengdu/status/data_test.npy', allow_pickle=True)
    features,truth_data=MulT_TTE_collate_func(train_data, args, info_all)
    # 标准化特征
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # 将标签进行独热编码
    truth_data = keras.utils.to_categorical(truth_data, num_classes=4)
    # 创建数据集和数据加载器
    data_loader = DataLoader(Datadict(train_data), batch_sampler=BatchSampler(train_data, args.data_config['batch_size']),
                                       collate_fn=lambda x:MulT_TTE_collate_func(x, args, info_all),
                                       pin_memory=True)
    # 初始化模型、损失函数和优化器
    model = MLP(56, 100, 50, 4)
    criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练模型
    for epoch in range(num_epochs):
        for features, labels in data_loader:
            # 前向传播
            labels=labels.to(torch.int64)
            if labels.dim() != 1:
                raise ValueError("Labels tensor must be one-dimensional.")
            # labels = torch.tensor([0, 1, 2, 3, 2, 1, 0]) 
            labels=torch.nn.functional.one_hot(labels, num_classes=4)
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), f'{args.data}_mlp_model.pth')
def pre_state(data):
    model =  MLP(56, 100, 50, 4)

    # 加载模型权重
    model.load_state_dict(torch.load('mlp_model.pth'))

    # 如果需要，将模型设置为评估模式
    model.eval()

    # 现在你可以使用加载的模型进行推断
    # 例如，使用随机输入进行测试
  
    output = model(data)
    return output