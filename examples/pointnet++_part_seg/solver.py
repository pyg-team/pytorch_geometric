import torch
import numpy as np


def train_epoch(model, criterion, optimizer, dataloader, device):
    model.train()

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        data = data.to(device)
        pred, _ = model(data)  # pred: (batch_size * ~n, num_classes)
        target = data.y
        loss = criterion(pred, target)
        loss.backward()

        optimizer.step()

        #
        pred_label = pred.detach().max(1)[1]
        correct = pred_label.eq(target.detach()).cpu().sum()
        total = pred_label.shape[0]
        print('[{}/{}] train loss: {} accuracy: {}'.format(
            batch_idx + 1, len(dataloader), loss.item(),
            float(correct.item()) / total))


def test_epoch(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for data in dataloader:
            data = data.to(device)
            pred, _ = model(data)
            target = data.y

            #
            pred_label = pred.max(1)[1]
            correct += pred_label.eq(target).cpu().sum()
            total += pred_label.shape[0]

        print('test accuracy: {}'.format(float(correct.item()) / total))


def eval_mIOU(model, dataloader, device, num_classes):
    model.eval()

    with torch.no_grad():
        shape_ious = []

        for data in dataloader:
            data = data.to(device)

            # pred: (batch_size * ~n, num_classes)
            pred, pred_batch = model(data)
            pred_label = pred.max(1)[1]  # (batch_size * ~n,)
            pred_label, pred_batch = pred_label.cpu().numpy(), pred_batch.cpu(
            ).numpy()
            target_label = data.y.cpu().numpy()

            batch_size = pred_batch.max() + 1

            for shape_idx in range(batch_size):
                pred_label_curr = pred_label[pred_batch == shape_idx]
                target_label_curr = target_label[pred_batch == shape_idx]

                part_labels = range(num_classes)

                part_ious = []
                for part in part_labels:
                    i = np.sum(
                        np.logical_and(pred_label_curr == part,
                                       target_label_curr == part))
                    u = np.sum(
                        np.logical_or(pred_label_curr == part,
                                      target_label_curr == part))
                    if u == 0:
                        iou = 1
                    else:
                        iou = float(i) / u
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))

        return np.mean(shape_ious)
