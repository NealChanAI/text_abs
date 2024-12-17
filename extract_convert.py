# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from tqdm import tqdm
from snippets import compute_main_metric
from data_process import get_csv_data, get_excel_data, train_test_split
import warnings
warnings.filterwarnings("ignore")

# 初始化
maxlen = 256


def text_segmentate(text, length=1, delimiters=u'\n。；：，'):
    """按照标点符号分割文本"""
    sentences = []
    buf = ''
    for ch in text:
        if ch in delimiters:
            if buf:
                sentences.append(buf)
            buf = ''
        else:
            buf += ch
    if buf:
        sentences.append(buf)
    return sentences


def text_split(text, limited=True):
    """将长句按照标点分割为多个子句"""
    texts = text_segmentate(text, 1, u'\n。；：，')
    if limited:
        texts = texts[:maxlen]
    return texts


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """
    在texts中找若干句子，使得它们连起来与summaries尽可能相似
    最终找出文本和摘要中相似度较高的句子对，并将它们的索引返回
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])  # 寻找摘要中最长的句子
    j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])  # 寻找文本中与该摘要句子最相似的句子
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def extract_flow(inputs):
    """抽取式摘要的流程"""
    res = []
    for line in inputs:
        text, summary = line
        texts = text_split(text, True)
        summaries = text_split(summary, False)
        mapping = extract_matching(texts, summaries)
        labels = sorted(set([i[1] for i in mapping]))  # text的索引(已排序)
        pred_summary = ''.join([texts[i] for i in labels])
        metric = compute_main_metric(pred_summary, summary)
        res.append([texts, labels, summary, metric])
    return res


def load_data(filename):
    """加载数据"""
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text = '\n'.join([d['sentence'] for d in l['text']])
            D.append((text, l['summary']))
    return D


def _load_data(dir_name):
    """加载并处理 Excel 数据集"""
    return get_excel_data(dir_name)


def convert(data):
    """分句，并转换为抽取式摘要"""
    D = extract_flow(data)
    total_metric = sum([d[3] for d in D])
    D = [d[:3] for d in D]  # 排除metric指标, [texts, labels, summary]
    print(u'抽取结果的平均指标: %s' % (total_metric / len(D)))
    return D


if __name__ == '__main__':
    data_json = './datasets/train.json'
    data_random_order_json = data_json[:-5] + '_random_order.json'
    data_extract_json = data_json[:-5] + '_extract.json'

    data = _load_data("./datasets/lastest_data_2210/500强企业资讯_link_补充正文.xlsx")
    train_data, test_data = train_test_split(data)
    data = convert(train_data)

    if os.path.exists(data_random_order_json):
        with open(data_random_order_json, 'r') as f:
            idxs = json.load(f)
    else:
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        with open(data_random_order_json, 'w') as f:
            json.dump(idxs, f)

    data = [data[i] for i in idxs]  # 随机打乱数据

    with open(data_extract_json, 'w', encoding='utf-8') as f:
        for d in data:
            tmp_lst = list(d)
            tmp_lst[1] = [str(i) for i in tmp_lst[1]]  # int转str
            d = tuple(tmp_lst)
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(u'输入数据：%s' % data_json)
    print(u'数据顺序：%s' % data_random_order_json)
    print(u'输出路径：%s' % data_extract_json)
