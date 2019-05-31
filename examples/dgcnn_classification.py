import os.path as osp

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.utils import scatter_
from torch_geometric.nn import knn_graph, EdgeConv
from torch_geometric.nn.inits import reset

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
          
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
     
class DGCNNCls(nn.Module):
    """Implementation of Dynamic Graph CNN(https://arxiv.org/abs/1801.07829)
    for classification task. This is a baseline version that doesn't 
    contain a sptial transform network"""
    
    def __init__(self, num_classes, K=20, conv_aggr='max', device=None):
        super().__init__()
        print('Build up DGCNNCls model, num_classes {}, K={}, device:{}'.format(num_classes, K, device))
        self.K = K
        self.num_classes = num_classes
        self.conv_aggr = conv_aggr
        self.edge_conv_dims = [3, 64, 64, 64, 128]
        self.edge_convs = self.make_edge_conv_layers_()
        self.glb_aggr = MLP([sum(self.edge_conv_dims[1::]), 1024])
        self.fc = nn.Sequential(MLP([1024, 512]), nn.Dropout(0.5),
                                MLP([512, 256]) , nn.Dropout(0.5),
                                nn.Linear(256, self.num_classes))
        self.device = device
        self.to(self.device)
        
    def make_edge_conv_layers_(self):
        layers = []
        dims =  self.edge_conv_dims
        for i in range(len(dims) - 1):
            mlp_dims = [dims[i] * 2, dims[i+1]]
            layers.append(EdgeConv(nn=MLP(mlp_dims), aggr=self.conv_aggr))
        return nn.Sequential(*layers)
                        
    def forward(self, pts, batch_ids):
        """
        Args: 
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
        out = torch.cat(edge_conv_outs, dim=-1)  # Skip connection to previous features
        out = self.glb_aggr(out)  # Global aggregation  B*N, 1024
        out = scatter_('max', out, index=batch_ids, dim_size=batch_size)  # B, 1024
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
        return loss
    
    def pred_(self, pts, batch_ids):
        return self.forward(pts, batch_ids).max(1).indices

def train_epoch(net, data_loader):
    net.train()
    if net.lr_scheduler:
        net.lr_scheduler.step()
    epoch_loss = 0
    for data in data_loader:
        data = data.to(net.device)
        lbls=data.y
        net.optimizer.zero_grad()
        loss = net.loss_(pts=data.pos, batch_ids=data.batch, lbls=lbls)
        epoch_loss += loss.item()
        loss.backward()
        net.optimizer.step()
    return epoch_loss / lbls.numel()  

def test_epoch(net, data_loader):
    net.eval()
    correct = 0
    for data in data_loader:
        data = data.to(net.device)
        with torch.no_grad():
            pred = net.pred_(pts=data.pos, batch_ids=data.batch)
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)

if __name__ == '__main__':
    # Initialize dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DGCNNCls(num_classes=train_dataset.num_classes, K=20, device=device)
    net.set_optimizer_(lr=0.001, weight_decay=0, lrd_step=20, lrd_factor=0.5)
        
    # Training
    print('Use device:{}.'.format(device))
    for epoch in range(1, 201):
        epoch_loss = train_epoch(net, train_loader)
        print('Epoch {}, loss:{:.4f}'.format(epoch, epoch_loss))
        if epoch % 10 == 0:
            test_acc = test_epoch(net, test_loader)
            print('Test accuracy:{:.4f}'.format(test_acc))
            