import pandas as pd
import torch.nn as nn
from model import HAN, FocalLoss
import torch
from utils import *
from metrics_utils import *
import time
from config import *
# import warnings
# warnings.filterwarnings('ignore')

def main():
    # 开始迭代
    # 记录开始时间
    for run_time in range(run_times):
        start_time = time.time()
        print('time', run_time)
        # 定义模型和损失函数等
        model = HAN(
            num_meta_paths=len(edge_type),
            in_size=feature_shape,
            hidden_size=hidden_size,
            out_size=num_classes,
            num_heads=num_heads,
            dropout=dropout,
        ).to(device)
        # Focal loss损失函数
        # loss_fcn = FocalLoss()
        # 交叉熵损失函数
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
        # 获取训练集，验证集，测试集
        train_g, train_feature, train_labels, valid_g, valid_feature, valid_labels, test_g, test_feature, test_labels = \
            new_split_data(split_dicts[run_time]['train'], split_dicts[run_time]['train'], split_dicts[run_time]['valid'], split_dicts[run_time]['test'])
        # 把所有数据放在GPU上
        # 训练集
        for key in train_g.keys():
            train_g[key] = [graph.to(device) for graph in train_g[key]]
        for key in train_feature.keys():
            train_feature[key] = torch.tensor(train_feature[key]).float().to(device)
        for key in train_labels.keys():
            train_labels[key] = torch.tensor(train_labels[key]).long().to(device)
        # print(train_g, train_feature, train_labels)
        # 验证集
        valid_g = [graph.to(device) for graph in valid_g]
        valid_feature = torch.tensor(valid_feature).float().to(device)
        valid_labels = torch.tensor(valid_labels).long().to(device)
        # 测试集
        test_g = [graph.to(device) for graph in test_g]
        test_feature = torch.tensor(test_feature).float().to(device)
        test_labels = torch.tensor(test_labels).long().to(device)


        train_id, val_id, test_id = split_dataset(train_feature[split_dicts[run_time]['train']], 3, 2, 5)
        # 对于每个epoch，我都使用逐年的数据来进行训练
        result_valid = []
        result_test = []
        for epoch in range(epochs):
            # 训练
            model.train()
            year = split_dicts[run_time]['train']
            logits = model(train_g[year], train_feature[year])
            loss = loss_fcn(logits[train_id], train_labels[year][train_id])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证
            model.eval()
            loss = loss_fcn(logits[val_id], valid_labels[val_id])
            prob, value = torch.max(logits[val_id], dim=1)
            result_valid.append(evaluate_new(labels=valid_labels[val_id].detach().cpu().numpy(),
                                             y_probs=logits.softmax(dim=1)[val_id, 1].detach().cpu().numpy(),
                                             epo=epoch,
                                             loss=loss.detach().cpu().numpy(),
                                             ))


            # 测试
            model.eval()
            loss = loss_fcn(logits[test_id], test_labels[test_id])
            prob, value = torch.max(logits[test_id], dim=1)
            result_test.append(evaluate_new(labels=test_labels[test_id].detach().cpu().numpy(),
                                             y_probs=logits.softmax(dim=1)[test_id, 1].detach().cpu().numpy(),
                                             epo=epoch,
                                             loss=loss.detach().cpu().numpy(),
                                             ))


        # 保存验证文件和测试文件
        pd.concat(result_valid).to_csv("result_wxg/" + str(run_time) + "_" + str(split_dicts[run_time]['valid']) + "_valid.csv", index=False, encoding='utf-8')
        pd.concat(result_test).to_csv("result_wxg/" + str(run_time) + "_" + str(split_dicts[run_time]['test']) + "_test.csv", index=False, encoding='utf-8')
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        elapsed_time = end_time - start_time

        print(f"程序运行时间：{elapsed_time}秒")

if __name__ == "__main__":
    main()