import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold

def to_data(path1,path2,path3,output_path):
    buggy_arr, fixed_arr, label_arr = [], [], []

    for tran_dir in os.listdir(path1):
        tran_path = os.path.join(path1, tran_dir)
        with open(tran_path, 'rb') as f:
            data = pd.read_pickle(f)
        tran_buggy_list = data.loc[0, 'tran_buggy']
        tran_fixed_list = data.loc[0, 'tran_fixed']
        is_correct = data.loc[0, 'is_correct']

        tran_buggy = ' '.join(tran_buggy_list)
        tran_fixed = ' '.join(tran_fixed_list)
        if is_correct == -1:
            label = 0
        else:
            label = is_correct
        if tran_buggy in buggy_arr and tran_fixed in fixed_arr:
            continue
        buggy_arr.append(tran_buggy)
        fixed_arr.append(tran_fixed)
        label_arr.append(label)

    for tran_dir in os.listdir(path2):
        tran_path = os.path.join(path2, tran_dir)
        with open(tran_path, 'rb') as f:
            data = pd.read_pickle(f)
        tran_buggy_list = data.loc[0, 'tran_buggy']
        tran_fixed_list = data.loc[0, 'tran_fixed']
        is_correct = data.loc[0, 'is_correct']

        tran_buggy = ' '.join(tran_buggy_list)
        tran_fixed = ' '.join(tran_fixed_list)
        if is_correct == -1:
            label = 0
        else:
            label = is_correct
        if tran_buggy in buggy_arr and tran_fixed in fixed_arr:
            continue
        buggy_arr.append(tran_buggy)
        fixed_arr.append(tran_fixed)
        label_arr.append(label)

    for tran_dir in os.listdir(path3):
        tran_path = os.path.join(path3, tran_dir)
        with open(tran_path, 'rb') as f:
            data = pd.read_pickle(f)
        tran_buggy_list = data.loc[0, 'tran_buggy']
        tran_fixed_list = data.loc[0, 'tran_fixed']
        is_correct = data.loc[0, 'is_correct']

        tran_buggy = ' '.join(tran_buggy_list)
        tran_fixed = ' '.join(tran_fixed_list)
        if is_correct == -1:
            label = 0
        else:
            label = is_correct
        if tran_buggy in buggy_arr and tran_fixed in fixed_arr:
            continue
        buggy_arr.append(tran_buggy)
        fixed_arr.append(tran_fixed)
        label_arr.append(label)

    data = buggy_arr, fixed_arr, label_arr
    with open(output_path, 'wb') as f:
        pd.to_pickle(data, f)


def divide_dataset(data_path, output_path):
    '''划分训练数据与测试数据，采用5倍交叉验证的方式进行划分'''
    # 加载数据
    with open(data_path, 'rb') as f:
        buggy_arr, fixed_arr, label_arr = pd.read_pickle(f)
    texts_1, texts_2, labels = np.array(buggy_arr), np.array(fixed_arr), np.array(label_arr)

    # 划分训练与测试数据集
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    index = [[train, test] for train, test in skf.split(texts_1, labels)]
    for i in range(len(index)):
        train_index, test_index = index[i][0], index[i][1]
        train_texts_1, train_texts_2, train_labels = texts_1[train_index], texts_2[train_index], labels[train_index]
        test_texts_1, test_texts_2, test_labels = texts_1[test_index], texts_2[test_index], labels[test_index]
        # 保存第 i 份训练数据
        data_train_path = os.path.join(output_path, 'data_code_train_' + str(i) + '.pkl')
        data_train = np.array(train_texts_1), np.array(train_texts_2), np.array(train_labels)
        with open(data_train_path, 'wb') as f:
            pd.to_pickle(data_train, f)
        # 保存第 i 份测试数据
        data_test_path = os.path.join(output_path, 'data_code_test_' + str(i) + '.pkl')
        data_test = np.array(test_texts_1), np.array(test_texts_2), np.array(test_labels)
        with open(data_test_path, 'wb') as f:
            pd.to_pickle(data_test, f)
            

if __name__ == '__main__':
    data_path = '../dataset/small/data_code.pkl'
    output_path = '../dataset/small'
    divide_dataset(data_path, output_path)

