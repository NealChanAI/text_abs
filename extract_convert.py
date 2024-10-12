#! -*- coding: utf-8 -*-
import os
import json
import numpy as np
from tqdm import tqdm
from bert4keras.snippets import open
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import parallel_apply
from snippets import *
from data_process import get_csv_data
from data_process import get_excel_data
from data_process import train_test_split
import warnings
warnings.filterwarnings("ignore")

# 初始化
maxlen = 256


def text_split(text, limited=True):
    """将长句按照标点分割为多个子句"""
    texts = text_segmentate(text, 1, u'\n。；：，')
    if limited:
        # texts = texts[-maxlen:]
        texts = texts[:maxlen]
    return texts


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """在texts中找若干句子，使得它们连起来与summaries尽可能相似
    算法：texts和summaries都分句，然后找出summaries最长的句子，在texts
          中找与之最相似的句子作为匹配，剩下部分递归执行。
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def extract_flow(inputs):
    res = []
    for line in inputs:
        text, summary = line
        texts = text_split(text, True)  # 取前maxlen句, 默认256
        summaries = text_split(summary, False)
        mapping = extract_matching(texts, summaries)
        labels = sorted(set([i[1] for i in mapping]))
        pred_summary = ''.join([texts[i] for i in labels])
        metric = compute_main_metric(pred_summary, summary)
        res.append([texts, labels, summary, metric])
    return res


def load_data(filename):
    """加载数据
    返回：[(text, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text = '\n'.join([d['sentence'] for d in l['text']])
            D.append((text, l['summary']))
    return D


def _load_data(dir_name):
    """
    返回处理完的excel数据集
    """
    # return get_csv_data(dir_name)
    return get_excel_data(dir_name)


def convert(data):
    """分句，并转换为抽取式摘要
    """
    D = extract_flow(data)
    total_metric = sum([d[3] for d in D])
    D = [d[:3] for d in D]
    print(u'抽取结果的平均指标: %s' % (total_metric / len(D)))
    return D


if __name__ == '__main__':
    data_random_order_json = data_json[:-5] + '_random_order.json'
    data_extract_json = data_json[:-5] + '_extract.json'

    data = _load_data("./datasets/lastest_data_2210/500强企业资讯_link_补充正文.xlsx")
    train_data, test_data = train_test_split(data)
    data = convert(train_data)

    if os.path.exists(data_random_order_json):
        idxs = json.load(open(data_random_order_json))
    else:
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        json.dump(idxs, open(data_random_order_json, 'w'))

    data = [data[i] for i in idxs]

    with open(data_extract_json, 'w', encoding='utf-8') as f:
        cnt = 0
        for d in data:
            # print(cnt)
            cnt += 1
            # print(d)
            tmp_lst = list(d)
            tmp_lst[1] = [str(i) for i in tmp_lst[1]]
            d = tuple(tmp_lst)
            # print(d)
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(u'输入数据：%s' % data_json)
    print(u'数据顺序：%s' % data_random_order_json)
    print(u'输出路径：%s' % data_extract_json)
