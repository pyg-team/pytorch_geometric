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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

        train_id, val_id, test_id = split_dataset(train_feature[split_dicts[run_time]['train']], 3, 2, 5)
        # 对于每个epoch，我都使用逐年的数据来进行训练
        result_valid = []
        result_test = []
        for epoch in range(epochs):
            # 训练
            year = split_dicts[run_time]['train']
            model.train()
            logits = model(train_g[year], train_feature[year])
            loss = loss_fcn(logits[train_id], train_labels[year][train_id])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证
            model.eval()
            loss = loss_fcn(logits[val_id], train_labels[year][val_id])
            prob, value = torch.max(logits[val_id], dim=1)
            result_valid.append(evaluate(y_proba=logits.softmax(dim=1)[val_id, 1].detach().cpu().numpy(),
                                    y_pred=value.detach().cpu().numpy(),
                                    label=train_labels[year][val_id].detach().cpu().numpy(),
                                    loss=loss.item(),
                                    epoch=epoch,
                                    params=model.parameters()))


            # 测试
            model.eval()
            loss = loss_fcn(logits[test_id], train_labels[year][test_id])
            prob, value = torch.max(logits[test_id], dim=1)
            res = evaluate(y_proba=logits.softmax(dim=1)[test_id, 1].detach().cpu().numpy(),
                                        y_pred=value.detach().cpu().numpy(),
                                        label=train_labels[year][test_id].detach().cpu().numpy(),
                                        loss=loss.item(),
                                        epoch=epoch,
                                        params=model.parameters())

            result_test.append(res)


        # 保存验证文件和测试文件
        pd.concat(result_valid).to_csv("result/" + str(run_time) + "_" + str(split_dicts[run_time]['valid']) + "_valid.csv", index=False, encoding='utf-8')
        pd.concat(result_test).to_csv("result/" + str(run_time) + "_" + str(split_dicts[run_time]['test']) + "_tests.csv", index=False, encoding='utf-8')
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        elapsed_time = end_time - start_time

        print(f"程序运行时间：{elapsed_time}秒")

if __name__ == "__main__":
    main()