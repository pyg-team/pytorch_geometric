import os.path as osp

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.utils import scatter_
from torch_scatter import scatter_add
from torch_geometric.nn import knn_graph, EdgeConv

from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import one_hot
# from torch_geometric.utils import mean_iou

def mean_iou(pred, target, num_classes, batch=None):
    r"""Computes the mean Intersection over Union score.
    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.
    :rtype: :class:`Tensor`
    """
    pred = one_hot(pred, num_classes, dtype=torch.long)
    target = one_hot(target, num_classes, dtype=torch.long)

    if batch is not None:
        i = scatter_add(pred & target, batch, dim=0).to(torch.float)
        u = scatter_add(pred | target, batch, dim=0).to(torch.float)
    else:
        i = (pred & target).sum(dim=0).to(torch.float)
        u = (pred | target).sum(dim=0).to(torch.float)

    iou = i / u
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou

class MLP(nn.Module):
    """Wrapping class of a MLP block.
    Args:
        layer_dims: list of dimensions as [input_dim, hidden_dims, output_dim],
                    to specify the mlp neuron numbers.
        batch_norm: append a batch normalization layer
        relu: append a ReLu layer
    """
    
    def __init__(self, layer_dims, batch_norm=True, relu=True):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.ReLU()) 
        self.layer = nn.Sequential(*layers)  
    
    def reset_parameters(self):
        reset(self.layer)
        
    def forward(self, x):
        return self.layer(x)
    
class DGCNNSeg(nn.Module):
    """Implementation of Dynamic Graph CNN(https://arxiv.org/abs/1801.07829)
    for segementation task. This is a baseline version that doesn't 
    contain a sptial transform network"""

    def __init__(self, num_classes, K=20, conv_aggr='max', device=None):
        super().__init__()
        print('Build up DGCNNSeg model, num_classes {}, K={}, device:{}'.format(num_classes, K, device))
        self.K = K
        self.num_classes = num_classes
        self.conv_aggr = conv_aggr
        self.edge_conv_dims = [[3, 64, 64], [64, 64, 64], [64, 64]]        
        self.edge_convs = self.make_edge_conv_layers_()
        self.aggr_dim = sum([dims[-1] for dims in self.edge_conv_dims])
        self.glb_aggr = MLP([self.aggr_dim , 1024])
        self.fc = nn.Sequential(MLP([self.aggr_dim + 1024, 512]),
                                MLP([512, 256]), nn.Dropout(0.5),
                                nn.Linear(256, self.num_classes))
        self.device = device
        self.to(self.device)
        
    def make_edge_conv_layers_(self):
        """Define structure of the EdgeConv Blocks
        edge_conv_dims: [[convi_mlp_dims]], e.g., [[3, 64], [64, 128]]
        """        
        layers = []
        for dims in self.edge_conv_dims:
            mlp_dims = [dims[0] * 2] + dims[1::]
            layers.append(EdgeConv(nn=MLP(mlp_dims), aggr=self.conv_aggr))
        return nn.Sequential(*layers)
                    
    def forward(self, pts, batch_ids):
        """
        Input: 
            - data.pos: (B*N, 3) 
            - data.batch: (B*N,)
        Return:
            - out: (B, C), softmax prob
        """
        
        batch_size = max(batch_ids) + 1  
        out = pts
        edge_conv_outs = []
        for edge_conv in self.edge_convs.children():
            # Dynamically update graph
            edge_index = knn_graph(pts, k=self.K, batch=batch_ids)
            out = edge_conv(out, edge_index)
            edge_conv_outs.append(out) 
        conv_cats = torch.cat(edge_conv_outs, dim=-1)  # Skip connection to previous features
        out = self.glb_aggr(conv_cats)  # Global aggregation
        glb_feats = scatter_('max', out, index=batch_ids, dim_size=batch_size) # B, 1024
        glb_feats = glb_feats[batch_ids]  # Expand to B*N, 1024
        out = torch.cat([glb_feats, conv_cats],  dim=-1)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
    
    def set_optimizer_(self, lr=0.001, weight_decay=0, lrd_step=20, lrd_factor=0.5):
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=lr, 
                                          weight_decay=weight_decay)

        # Setup lr scheduler if defined
        if lrd_factor < 1 and lrd_step > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=lrd_step, 
                                                                gamma=lrd_factor)
        else:
            self.lr_scheduler = None
            
    def loss_(self, pts, batch_ids, lbls):
        logits = self.forward(pts, batch_ids)
        loss = F.nll_loss(logits, lbls)
        preds = logits.max(dim=1).indices
        return loss, preds
    
    def pred_(self, pts, batch_ids):
        return self.forward(pts, batch_ids).max(1).indices
    
def train_epoch(net, data_loader):
    net.train()
    if net.lr_scheduler:
        net.lr_scheduler.step()
    
    epoch_loss = 0
    correct = total = 0
    epoch_iou = []
    num_batch = len(data_loader)
    for data in data_loader:
        data = data.to(net.device)
        lbls = data.y
        net.optimizer.zero_grad()
        loss, preds = net.loss_(pts=data.pos, batch_ids=data.batch, lbls=lbls)
        loss.backward()
        net.optimizer.step()
        epoch_loss += loss
        correct += preds.eq(lbls).sum().item()
        total += lbls.numel()
        epoch_iou.append(mean_iou(lbls, preds, net.num_classes, batch=data.batch))
    epoch_loss = epoch_loss / num_batch
    epoch_acc = correct / total
    epoch_iou = torch.cat(epoch_iou, dim=0).mean().item()
    return epoch_loss, epoch_acc, epoch_iou

def test_epoch(net, data_loader):
    net.eval()
    correct = total = 0
    epoch_iou = []
    for data in data_loader:
        data = data.to(net.device)
        lbls = data.y
        with torch.no_grad():
            preds = net.pred_(pts=data.pos, batch_ids=data.batch)   
        correct += preds.eq(lbls).sum().item()
        total += lbls.numel()
        epoch_iou.append(mean_iou(lbls, preds, net.num_classes, batch=data.batch))
    epoch_acc = correct / total
    epoch_iou = torch.cat(epoch_iou, dim=0).mean().item()
    return epoch_acc, epoch_iou

if __name__ == '__main__':
    # Initialize dataset
    category = 'Cap'
    print('Training DGCNN on Shapenet {} category'.format(category))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(path, category, train=True, 
                             transform=transform, pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, train=False, 
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    # Initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DGCNNSeg(num_classes=train_dataset.num_classes, K=20, device=device)
    net.set_optimizer_(lr=0.001, weight_decay=0, lrd_step=20, lrd_factor=0.8)
    
    # Training
    print('Use device:{}.'.format(device))
    for epoch in range(1, 201):
        loss, acc, iou = train_epoch(net, train_loader)
        print('Epoch {}, train loss:{:.4f} acc: {:.4f} iou: {:.4f}'.format(epoch, loss, acc, iou))
        if epoch % 5 == 0:
            test_acc, test_iou = test_epoch(net, test_loader)
            print('Test accuracy:{:.4f} iou: {:.4f}'.format(test_acc, test_iou))
            