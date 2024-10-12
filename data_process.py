import os
import pandas as pd
import random
import math


def get_csv_data(dir_path):
    """
    获取所有的文件，并剔除掉没有正文的数据
    :return: list [(text, summary), (text, summary), ..., (text, summary)]
    """

    res_lst = []
    for root, dirs, files in os.walk(dir_path):
        cnt = 0
        for file in files:
            cnt += 1
            file_path = os.path.join(root, file)
            print(file_path)

            df = pd.read_excel(file_path)
            df = df[pd.notnull(df["content"])]
            if cnt == 1:
                base_df = df
            else:
                base_df = pd.concat([base_df,df],sort=False)

            # if cnt == 2: break

        for idx, data in base_df.iterrows():
            if idx == 0: continue
            res_lst.append((data[3], data[2]))
            # res_lst.append((data[2], data[1]))
    return res_lst


def get_excel_data(dir_path):
    """
    read excel file 
    """
    res_lst = []
    df = pd.read_excel(dir_path, engine='openpyxl')
    for idx, data in df.iterrows():
        # if idx == 0: continue
        res_lst.append((str(data[3]).strip(), str(data[1]).strip()))
    return res_lst


def train_test_split(lst):
    """
    划分训练/测试数据集
    """
    train_data = []
    test_data = []
  
    random.seed(1024)
    train_idx = random.sample([idx for idx in range(len(lst))], math.ceil(len(lst) * 0.8))
    
    for i in range(len(lst)):
        if i in train_idx:
            train_data.append(lst[i])
        else:
            test_data.append(lst[i])

    return train_data, test_data


if __name__ == '__main__':
    # res_lst = get_csv_data("datasets/result/kg_data")
    # print(len(res_lst))
    res_lst = get_excel_data("datasets/lastest_data_2210/500强企业资讯_link_补充正文.xlsx")
    print(res_lst[0])
    print(len(res_lst))
