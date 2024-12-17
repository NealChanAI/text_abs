# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from snippets import *

# 配置信息
input_size = 768
hidden_size = 384
epochs = 20
batch_size = 64
threshold = 0.2
data_extract_json = './datasets/train_extract.json'
data_extract_npy = './datasets/train_extract.npy'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualGatedConv1D(nn.Module):
    """门控卷积"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1):
        super(ResidualGatedConv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size,
            dilation=dilation_rate, padding='same'
        )
        self.norm = nn.LayerNorm(out_channels)
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        conv_out = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, 2*out_channels]
        gate = torch.sigmoid(conv_out[..., self.conv1d.out_channels // 2:])
        gated_output = conv_out[..., :self.conv1d.out_channels // 2] * gate
        norm_out = self.norm(gated_output)

        if self.proj:
            x = self.proj(x)

        return x + self.alpha * norm_out


class ExtractModel(nn.Module):
    """抽取式模型"""

    def __init__(self, input_size, hidden_size):
        super(ExtractModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.mask = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            self.dropout
        )
        self.gated_convs = nn.Sequential(
            ResidualGatedConv1D(hidden_size, hidden_size, kernel_size=3, dilation_rate=1),
            self.dropout,
            ResidualGatedConv1D(hidden_size, hidden_size, kernel_size=3, dilation_rate=2),
            self.dropout,
            ResidualGatedConv1D(hidden_size, hidden_size, kernel_size=3, dilation_rate=4),
            self.dropout,
            ResidualGatedConv1D(hidden_size, hidden_size, kernel_size=3, dilation_rate=8),
            self.dropout,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        x = self.mask(x)
        x = self.gated_convs(x)
        x = self.output_layer(x).squeeze(-1)
        return x


def load_data(filename):
    """
    加载数据
    :param filename: 数据集路径，JSON 文件
    :return: list of (texts, labels, summary)
    """
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            texts = record[0]
            labels = list(map(int, record[1]))  # 转换标签为整数
            summary = record[2]
            data.append((texts, labels, summary))
    return data


def evaluate(model, data_loader, threshold=0.2):
    """验证集评估"""
    model.eval()
    total_metrics = {k: 0.0 for k in metric_keys}
    with torch.no_grad():
        for batch_x, batch_y, texts, summaries in data_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu()
            for i, pred in enumerate(preds):
                pred = pred.numpy()
                pred_indices = np.where(pred > threshold)[0]
                pred_summary = ''.join([texts[i][j] for j in pred_indices])
                metrics = compute_metrics(pred_summary, summaries[i], 'char')
                for k, v in metrics.items():
                    total_metrics[k] += v
    return {k: v / len(data_loader.dataset) for k, v in total_metrics.items()}


if __name__ == '__main__':
    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    data_y = np.zeros_like(data_x[..., :1])

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i, int(j)] = 1

    # 数据分割
    train_x, valid_x = data_split(data_x, fold, num_folds, 'train'), data_split(data_x, fold, num_folds, 'valid')
    train_y, valid_y = data_split(data_y, fold, num_folds, 'train'), data_split(data_y, fold, num_folds, 'valid')

    # 数据加载
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                  torch.tensor(train_y, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(valid_x, dtype=torch.float32),
                                  torch.tensor(valid_y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = ExtractModel(input_size, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # 训练模型
    best_metric = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, valid_loader, threshold + 0.1)
        if metrics['main'] >= best_metric:
            best_metric = metrics['main']
            torch.save(model.state_dict(), f'weights/extract_model.{fold}.pth')
        print(f"Epoch {epoch + 1}/{epochs}, Metrics: {metrics}")
