# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from tqdm import tqdm
from extract_model import model, load_data, data_split
from snippets import *

def fold_convert(data, data_x, fold, model, threshold=0.2):
    """
    每一fold用对应的模型做数据转换
    :param data: 原始数据
    :param data_x: 特征数据
    :param fold: 当前 fold
    :param model: 训练好的模型
    :param threshold: 预测阈值
    :return: 转换后的结果
    """
    valid_data = data_split(data, fold, num_folds, 'valid')
    valid_x = data_split(data_x, fold, num_folds, 'valid')

    # 加载对应 fold 的模型权重
    model.load_state_dict(torch.load(f'weights/extract_model.{fold}.pth'))
    model.eval()

    results = []
    with torch.no_grad():
        y_pred = model(torch.tensor(valid_x, dtype=torch.float32).to(device))
        y_pred = y_pred.cpu().numpy()[:, :, 0]

    for d, yp in tqdm(zip(valid_data, y_pred), desc=u'转换中'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        source_1 = ''.join([d[0][int(i)] for i in yp])
        source_2 = ''.join([d[0][int(i)] for i in d[1]])
        result = {
            'source_1': source_1,
            'source_2': source_2,
            'target': d[2],
        }
        results.append(result)

    return results


def convert(filename, data, data_x, model):
    """
    转换为生成式数据
    :param filename: 输出文件名
    :param data: 原始数据
    :param data_x: 特征数据
    :param model: PyTorch 模型
    """
    total_results = []
    for fold in range(num_folds):
        total_results.append(fold_convert(data, data_x, fold, model))

    # 按照原始顺序写入到文件中
    n = 0
    with open(filename, 'w', encoding='utf-8') as f:
        while True:
            i, j = n % num_folds, n // num_folds
            try:
                d = total_results[i][j]
            except IndexError:
                break
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            n += 1


if __name__ == '__main__':
    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    data_seq2seq_json = data_json[:-5] + '_seq2seq.json'

    # 模型迁移到 GPU 或 CPU
    model.to(device)

    # 转换数据并保存
    convert(data_seq2seq_json, data, data_x, model)
    print(u'输出路径：%s' % data_seq2seq_json)
