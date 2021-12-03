from point_transformer_classification import TransitionDown, TransformerBlock
from point_transformer_classification import MLP
from point_transformer_segmentation import TransitionUp, Net

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU

import torch_geometric.transforms as T
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u, mean_iou
from torch_geometric.nn.unpool import knn_interpolate

from torch_cluster import knn_graph

def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_history.append(total_loss)
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 100 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    preds = []  #to compute IoU on the full dataset instead of a per-batch basis
    labels = [] # we will stack the predictions and labels
    for i,data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)


    preds_tensor = torch.hstack(preds).type(torch.LongTensor).cpu()
    labels_tensor = torch.hstack(labels).type(torch.LongTensor).cpu()
    
    i, u = torch.zeros((13)), torch.zeros((13))
    # intersection_and_union does one-hot encoding, making the full labels
    # matrix too large to fit all at once so we do it in two times
    
    i_sub, u_sub = i_and_u(preds_tensor[:2000000], labels_tensor[:2000000], 13)#, data.batch)
    i += i_sub
    u += u_sub
    
    i_sub, u_sub = i_and_u(preds_tensor[2000000:], labels_tensor[2000000:], 13)#, data.batch)
    i += i_sub
    u += u_sub
    
    iou = i / u
    iou = iou[ torch.isnan(iou) == False].mean()
    # Compute mean IoU.
    return iou


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'S3DIS')
    print(path)
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = S3DIS(path, test_area=5, train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = S3DIS(path, test_area=5, train=False,
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(6, train_dataset.num_classes, dim_model=[
        32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    loss_history = []
    test_iou_history = []
    for epoch in range(1, 100):
        train()
        iou = test(test_loader)
        test_iou_history.append(iou)
        print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
        scheduler.step()
    
    
    
    
    
    
