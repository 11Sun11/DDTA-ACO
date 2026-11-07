'''
Author: Xin-Can Wang
Date: 2024-12-24 09:36:47
LastEditors: Xin-Can Wang
LastEditTime: 2024-12-24 20:57:00
Description: 
FilePath: /wangxincan/code/ACO-MulT-TTE/models/MLP.py
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils.prepare import *
from utils.util import *
from tqdm import tqdm
# 假设我们有交通数据，使用 Pandas 读取数据
# 这里我们生成一些示例数据
def MLP(args):
    info_all=load_datadoct_pre(args)
    train_data=np.load('./code/ACO-MulT-TTE/data/chengdu/status/data_train.npy', allow_pickle=True)
    # test_data=np.load('./code/ACO-MulT-TTE/data/chengdu/status/data_test.npy', allow_pickle=True)
    features,truth_data=MulT_TTE_collate_func(train_data, args, info_all)
    # 标准化特征
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # 将标签进行独热编码
    truth_data = keras.utils.to_categorical(truth_data, num_classes=4)
    # 构建多层感知机模型
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(features["links"].shape[1],)),  # 第一个隐藏层
        layers.Dense(32, activation='relu'),  # 第二个隐藏层
        layers.Dense(4, activation='softmax')  # 输出层，4个节点用于多分类
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    features_links_np = features["links"].numpy()  # 转换为 NumPy 数组
    features_links_np=features_links_np.reshape(-1, 10)
    # truth_data_np = truth_data.numpy()  # 转换为 NumPy 数组

    model.fit(features_links_np, truth_data,)
    model.save('my_model.h5')
    # 评估模型

    # # 进行预测
    # predictions = model.predict(X_test)
    # predicted_classes = np.argmax(predictions, axis=1)  # 获取预测的类别

# MLP()