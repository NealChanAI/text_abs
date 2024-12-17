import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from snippets import *

# 基本参数
maxlen = 1024
batch_size = 8
epochs = 50
k_sparse = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据文件路径
data_seq2seq_json = './datasets/seq2seq_data.json'
seq2seq_config_json = './datasets/seq2seq_config.json'

# 加载词表和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 自定义数据集
class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        source = record['source_1']
        target = record['target']

        source_enc = self.tokenizer(source, max_length=self.maxlen, padding="max_length", truncation=True, return_tensors="pt")
        target_enc = self.tokenizer(target, max_length=self.maxlen // 2, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0)
        }

# 加载数据
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

# 数据生成器
def create_data_loader(data, tokenizer, maxlen, batch_size):
    dataset = Seq2SeqDataset(data, tokenizer, maxlen)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义模型
class Seq2SeqModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(Seq2SeqModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=3)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(data_loader)

# 主函数
if __name__ == '__main__':
    # 加载数据
    data = load_data(data_seq2seq_json)
    train_data, valid_data = data_split(data, 0, 10, 'train'), data_split(data, 0, 10, 'valid')

    train_loader = create_data_loader(train_data, tokenizer, maxlen, batch_size)
    valid_loader = create_data_loader(valid_data, tokenizer, maxlen, batch_size)

    # 初始化模型和优化器
    model = Seq2SeqModel('bert-base-chinese').to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证模型
        valid_loss = evaluate(model, valid_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Valid Loss: {valid_loss:.4f}")

        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"weights/seq2seq_model_best.pth")
            print("Saved Best Model!")
