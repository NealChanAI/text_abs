# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from snippets import *


class GlobalAveragePooling1D(torch.nn.Module):
    """自定义全局池化"""
    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            return (inputs * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            return inputs.mean(dim=1)


# 设置设备为 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained(dict_path)

# 加载预训练 BERT 模型到 GPU
bert_model = BertModel.from_pretrained(checkpoint_path).to(device)
pooling_layer = GlobalAveragePooling1D().to(device)


def load_data(filename):
    """加载数据, 返回：[texts]"""
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            texts = json.loads(l)[0]
            D.append(texts)
    return D


def predict(texts):
    """句子列表转换为句向量"""
    batch_token_ids, attention_masks = [], []
    for text in texts:  # 同一篇文章的多个句子
        encoded = tokenizer.encode_plus(
            text, max_length=512, truncation=True, padding='max_length', return_tensors='pt'
        )
        batch_token_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))

    batch_token_ids = pad_sequence(batch_token_ids, batch_first=True).to(device)
    attention_masks = pad_sequence(attention_masks, batch_first=True).to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=batch_token_ids, attention_mask=attention_masks)
        sequence_output = outputs.last_hidden_state
        pooled_output = pooling_layer(sequence_output, mask=attention_masks)
    return pooled_output.cpu().numpy()


def convert(data):
    """转换所有样本"""
    embeddings = []
    for texts in tqdm(data, desc=u'向量化'):
        outputs = predict(texts)
        print('texts:', texts)
        print(outputs)
        print('len:', len(outputs))
        embeddings.append(outputs)
    embeddings = np.array(embeddings)
    return embeddings


if __name__ == '__main__':
    data_json = './datasets/train.json'
    data_extract_json = data_json[:-5] + '_extract.json'
    data_extract_npy = data_json[:-5] + '_extract'

    data = load_data(data_extract_json)
    embeddings = convert(data)
    np.save(data_extract_npy, embeddings)
    print(u'输出路径：%s.npy' % data_extract_npy)
