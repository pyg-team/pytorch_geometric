import time

from config import *
from metrics_utils import *
from model import *
from utils import *


class FocalLoss(nn.Module):
    # gamma=0,alpha=0.86,gcn层数为1，应该最优，可以平衡欺诈和非欺诈之间的识别, 我们的数据集
    # alpha=? ,ACM
    def __init__(self, gamma=0, alpha=0.86, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * pt ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * (1 - pt) ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


for key, value in split_dicts.items():
    print('year', value)

    # 加载数据集
    G, ntypes, etypes, labels = load_dataset(value, graph_type_full,
                                             graph_type_etypes, feature_path,
                                             device)

    model = HGT(G, n_inp=G.nodes['L'].data['inp'].shape[1], n_hid=64,
                n_out=labels.max().item() + 1, n_layers=3, n_heads=8,
                use_norm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fcn = FocalLoss()

    train_id, val_id, test_id = split_dataset(labels, 3, 2, 5)

    # 对于每个epoch，使用当年数据73分
    result_train = []
    result_valid = []
    result_test = []

    # 每一年我都执行times次
    for t in range(times):
        # 开始时间
        start_time = time.time()
        for epoch in range(epochs):
            print('epoch', epoch)

            # 训练
            model.train()
            logits = model(G, 'L')
            # The loss is computed only for labeled nodes.
            loss = loss_fcn(
                logits.softmax(dim=1)[train_id, 1],
                labels[train_id].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result_train.append(
                evaluate_new(
                    labels=labels[train_id].detach().cpu().numpy(),
                    y_probs=logits.softmax(dim=1)[train_id,
                                                  1].detach().cpu().numpy(),
                    epo=epoch,
                    loss=loss.detach().cpu().numpy(),
                ))

            # 验证
            model.eval()
            loss = loss_fcn(
                logits.softmax(dim=1)[val_id, 1], labels[val_id].to(device))
            prob, value_label = torch.max(logits.softmax(dim=1)[val_id], dim=1)
            # result_valid.append(evaluate_new(y_proba=logits.softmax(dim=1)[val_id, 1].detach().cpu().numpy(),
            #                              y_pred=value_label.detach().cpu().numpy(),
            #                              label=labels[val_id].detach().cpu().numpy(),
            #                              loss=loss.item(),
            #                              epoch=epoch,
            #                              params=model.parameters()))
            result_valid.append(
                evaluate_new(
                    labels=labels[val_id].detach().cpu().numpy(),
                    y_probs=logits.softmax(dim=1)[val_id,
                                                  1].detach().cpu().numpy(),
                    epo=epoch,
                    loss=loss.detach().cpu().numpy(),
                ))

            # 测试
            model.eval()
            loss = loss_fcn(
                logits.softmax(dim=1)[test_id, 1], labels[test_id].to(device))
            prob, value_label = torch.max(
                logits.softmax(dim=1)[test_id], dim=1)
            # res = evaluate(y_proba=logits.softmax(dim=1)[test_id, 1].detach().cpu().numpy(),
            #                y_pred=value_label.detach().cpu().numpy(),
            #                label=labels[test_id].detach().cpu().numpy(),
            #                loss=loss.item(),
            #                epoch=epoch,
            #                params=model.parameters())
            res = evaluate_new(
                labels=labels[test_id].detach().cpu().numpy(),
                y_probs=logits.softmax(dim=1)[test_id,
                                              1].detach().cpu().numpy(),
                epo=epoch,
                loss=loss.detach().cpu().numpy(),
            )
            # print(res.loc[0, 'Recall'], res.loc[0, 'Recall_macro'], res.loc[0, 'ROC_AUC'], res.loc[0, "KS"])
            result_test.append(res)

        end_time = time.time()
        print("程序运行时间：", end_time - start_time, "秒")
        # 保存验证文件和测试文件
        pd.concat(result_valid).to_csv(
            "HGT_result_wxg/" + str(key) + "_" + str(t) + "_" + str(value) +
            "_train.csv", index=False, encoding='utf-8')
        pd.concat(result_valid).to_csv(
            "HGT_result_wxg/" + str(key) + "_" + str(t) + "_" + str(value) +
            "_valid.csv", index=False, encoding='utf-8')
        pd.concat(result_test).to_csv(
            "HGT_result_wxg/" + str(key) + "_" + str(t) + "_" + str(value) +
            "_test.csv", index=False, encoding='utf-8')
