"""
数据加载与预处理：自动下载ETTm1数据集，生成滑动窗口样本
支持长期预测任务：历史输入 X，预测未来 Y
"""

import os
import requests
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ETTm1 数据集的直接下载链接（来自ETDataset）
DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
DATA_PATH = "./ETTm1.csv"

def download_data():
    """若本地不存在，则下载ETTm1.csv"""
    if not os.path.exists(DATA_PATH):
        print("下载 ETTm1 数据集...")
        r = requests.get(DATA_URL)
        with open(DATA_PATH, 'wb') as f:
            f.write(r.content)
        print("下载完成。")
    else:
        print("本地已存在 ETTm1.csv，跳过下载。")

def normalize(data, mean=None, std=None):
    """标准化：减均值除标准差"""
    if mean is None:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
    data_norm = (data - mean) / (std + 1e-8)
    return data_norm, mean, std


class TimeSeriesDataset(Dataset):
    """滑动窗口数据集，用于长期预测"""
    def __init__(self, data, input_len=96, pred_len=96, indices=None):
        """
        data: 标准化后的完整数据 (total_steps, num_vars)
        indices: 每个样本的起始索引列表
        """
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.indices = indices if indices is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start : start + self.input_len]          # (input_len, var)
        y = self.data[start + self.input_len : start + self.input_len + self.pred_len]  # (pred_len, var)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_dataloaders(batch_size=32, input_len=96, pred_len=96, val_ratio=0.1, test_ratio=0.2):
    """返回训练、验证、测试数据加载器"""
    download_data()
    df = pd.read_csv(DATA_PATH)  # 第一列是date，后面是数值列

    # 提取数值部分并标准化
    data_raw = df.iloc[:, 1:].values.astype(np.float32)  # (总步数, 变量数)
    data_norm, mean, std = normalize(data_raw)

    total_len = len(data_norm)
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len

    # 生成每个样本的起始索引（滑动窗口步长为1）
    def get_indices(start, end):
        indices = []
        for i in range(start, end - input_len - pred_len + 1):
            indices.append(i)
        return indices

    train_indices = get_indices(0, train_len)
    val_indices = get_indices(train_len, train_len + val_len)
    test_indices = get_indices(train_len + val_len, total_len)

    train_dataset = TimeSeriesDataset(data_norm, input_len, pred_len, train_indices)
    val_dataset = TimeSeriesDataset(data_norm, input_len, pred_len, val_indices)
    test_dataset = TimeSeriesDataset(data_norm, input_len, pred_len, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean, std


# 测试用
if __name__ == "__main__":
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(batch_size=4)
    for x, y in train_loader:
        print("输入形状:", x.shape, "目标形状:", y.shape)
        break