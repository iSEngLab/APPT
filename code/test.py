import numpy as np
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
import torch
import os
import shutil
from pathlib import Path

from DataLoader import Dataset
from Model import Model
from configs import Config
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score


def tokenizer_head_tail(text, tokenizer, max_length):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_length:
        half_length = int(max_length / 2)
        encoding['input_ids'] = encoding['input_ids'][:half_length] + encoding['input_ids'][-half_length:]
        #        encoding['token_type_ids'] = encoding['token_type_ids'][:half_length] + encoding['token_type_ids'][-half_length:]
        encoding['attention_mask'] = encoding['attention_mask'][:half_length] + encoding['attention_mask'][
                                                                                -half_length:]
        # encoding.pop('token_type_ids')
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in
                                                                   range(len(encoding['attention_mask']), max_length)]
        # encoding.pop('token_type_ids')
    return encoding


def tokenizer_head(text, tokenizer, max_legnth):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_legnth:
        encoding['input_ids'] = encoding['input_ids'][:max_legnth - 1] + encoding['input_ids'][-1:]
        #        encoding['token_type_ids'] = encoding['token_type_ids'][:max_legnth-1] + encoding['token_type_ids'][-1:]
        encoding['attention_mask'] = encoding['attention_mask'][:max_legnth - 1] + encoding['attention_mask'][-1:]
        # encoding.pop('token_type_ids')
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in
                                                                   range(len(encoding['attention_mask']), max_length)]
        # encoding.pop('token_type_ids')
    return encoding


def tokenizer_tail(text, tokenizer, max_legnth):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_legnth:
        encoding['input_ids'] = encoding['input_ids'][:1] + encoding['input_ids'][-max_legnth + 1:]
        #        encoding['token_type_ids'] = encoding['token_type_ids'][:max_legnth-1] + encoding['token_type_ids'][-1:]
        encoding['attention_mask'] = encoding['attention_mask'][:1] + encoding['attention_mask'][-max_legnth + 1:]
        # encoding.pop('token_type_ids')
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in
                                                                   range(len(encoding['attention_mask']), max_length)]
        # encoding.pop('token_type_ids')
    return encoding


def tokenizer_mid(text, tokenizer, max_legnth):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_legnth:
        encoding['input_ids'] = encoding['input_ids'][(len(encoding['input_ids']) - max_length) // 2: (len(
            encoding['input_ids']) + max_length) // 2]
        #        encoding['token_type_ids'] = encoding['token_type_ids'][:max_legnth-1] + encoding['token_type_ids'][-1:]
        encoding['attention_mask'] = encoding['attention_mask'][(len(encoding['input_ids']) - max_length) // 2: (
                                                                                                                            len(
                                                                                                                                encoding[
                                                                                                                                    'input_ids']) + max_length) // 2]
        # encoding.pop('token_type_ids')
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in
                                                                   range(len(encoding['attention_mask']), max_length)]
        # encoding.pop('token_type_ids')
    return encoding


def load(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def test(model, test_loader):
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            out = torch.sigmoid(model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2))

            y_true.append(labels.item())
            y_score.append(out.item())

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    auc_ = auc(fpr, tpr)
    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


if __name__ == '__main__':
    # 配置类
    config = Config()
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # 模型最长输入
    max_length = config.max_length

    data_test_path = config.data_test
    with open(data_test_path, 'rb') as f:
        test_texts_1, test_texts_2, test_labels = pd.read_pickle(f)
        test_texts_1 = list(test_texts_1)
        test_texts_2 = list(test_texts_2)
        test_labels = list(test_labels)
        test_texts_1 = [text.lower() for text in test_texts_1]
        test_texts_2 = [text.lower() for text in test_texts_2]
        # 过拟合检测/正确补丁检测
        test_labels = [0 if label == 1 else 1 for label in test_labels]

    tokenizer_func = {'headTail': tokenizer_head_tail, 'head': tokenizer_head, 'tail': tokenizer_tail,
                      'mid': tokenizer_mid}
    test_dataset = Dataset(tokenizer_func[config.cutMethod], tokenizer, max_length, test_texts_1, test_texts_2,
                           test_labels)

    # 生成训练和测试Dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True)

    # 模型
    model = Model(config)
    model = load(model, config.model_test_path)
    # 定义GPU/CPU
    device = config.device
    model.to(device)
    # 多GPU并行
    model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # 测试
    test(model, test_loader)








