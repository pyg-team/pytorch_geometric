from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGraphConv, GraphConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

EPS = 1e-15

class GNN(torch.nn.Module):
    """Graph Neural Network with graph pooling"""

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        num_classes: int,
        hidden_dim: list = [32, 32],
        mlp_hidden_dim: int = 32,
        mu: float = 0.1,
        new_ortho: bool = False,
        cluster_ratio: float = 0.2,
        dropout: float = 0,
        device=None,
        sn=False,
    ):

        super(GNN, self).__init__()

        self.num_layers = len(hidden_dim)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.rms = []
        self.mu = mu
        self.sn = sn
        self.new_ortho = new_ortho
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pooling_type = dense_hoscpool

        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(GraphConv(num_node_features, hidden_dim[i]))  # sparse
            else:
                self.convs.append(
                    DenseGraphConv(hidden_dim[i - 1], hidden_dim[i])
                )  # dense

        # Pooling layers (to learn cluster matrix)
        for i in range(self.num_layers - 1):
            num_nodes = ceil(cluster_ratio * num_nodes)  # K
            self.pools.append(Linear(hidden_dim[i], num_nodes))

        # Dense layers for prediction
        self.lin1 = Linear(hidden_dim[-1], mlp_hidden_dim)
        self.lin2 = Linear(mlp_hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, mask=None):

        # Normalise adj
        edge_weight = None

        ### Block 1
        # convolution
        x = F.relu(self.convs[0](x, edge_index))
        # convert batch sparse to batch dense
        x, mask = to_dense_batch(x, batch)
        # convert sparse adj to dense adj
        adj = to_dense_adj(edge_index, batch, edge_weight)
        # Cluster ass matrix
        s = self.pools[0](x)
        # Pooling
        x, adj, mc, o = self.pooling_type(
            x,
            adj,
            s,
            self.mu,
            alpha=0.5,
            new_ortho=self.new_ortho,
            mask=mask,
        )  # pooling

        ### Middle blocks
        for i in range(1, self.num_layers - 1):
            x = F.relu(self.convs[i](x, adj))  # Convolution
            s2 = self.pools[i](x)  # cluster ass matrix
            x, adj, mc_aux, o_aux = self.pooling_type(
                x,
                adj,
                s2,
                self.mu,
                alpha=0.5,
                new_ortho=self.new_ortho,
            )  # pooling

            mc += mc_aux
            o += o_aux

        ### Last block
        x = self.convs[self.num_layers - 1](x, adj)  # conv
        x = x.mean(dim=1)  # global pooling

        # Dense classifier
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        if self.num_layers > 2:
            # Complex archi: MP - Pooling - MP - Pooling - MP - Global Pooling - Dense (x2)
            return x, mc, o, [s, s2]
        else:
            # Simple archi: MP - Pooling - MP - Global Pooling - Dense (x2)
            return x, mc, o, s


def dense_hoscpool(
    x, adj, s, mu=0.1, alpha=0.5, new_ortho=False, mask=None,
):
    r"""The highe-order pooling operator (HoscPool) from the paper 
    `"Higher-order clustering and pooling for Graph Neural Networks"
    <http://arxiv.org/abs/2209.03473>`_. Based on motif spectral clustering, 
    it captures and combines different levels of higher-order connectivity 
    patterns when coarsening the graph.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
    
    based on the learned cluster assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times K}`. This function returns the pooled feature matrix, the coarsened 
    symmetrically normalised adjacency matrix, the motif spectral clustering loss :math:`\mathcal{L}_{mc}`
    and the orthogonality loss :math:`\mathcal{L}_{o}`.

    .. math::
        \mathcal{L}_{mc} &= - \frac{\alpha_1}{K} \cdot \text{Tr}\bigg(\frac{\mathbf{S}^\top \mathbf{A} \mathbf{S}}
            {\mathbf{S}^\top\mathbf{D}\mathbf{S}}\bigg) - \frac{\alpha_2}{K} \cdot \text{Tr}\bigg(
                \frac{\mathbf{S}^\top\mathbf{A}_{M}\mathbf{S}}{\mathbf{S}^\top\mathbf{D}_{M}\mathbf{S}}\bigg).

        \mathcal{L}_o &= \frac{1}{\sqrt{K}-1} \bigg( \sqrt{K} - \frac{1}{\sqrt{N}}\sum_{j=1}^K ||S_{*j}||_F\bigg)

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): the learnable cluster assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times K}` with number of clusters :math:`K`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mu (Tensor, optional): scalar that controls the importance given to regularization loss
        alpha (Tensor, optional): scalar in [0,1] controlling the importance granted
            to higher-order information (in loss function).
        new_ortho (BoolTensor, optional): either to use new proposed loss or old one
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # Output adjacency and feature matrices
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Motif adj matrix - not sym. normalised
    motif_adj = torch.mul(torch.matmul(adj, adj), adj)
    motif_out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), motif_adj), s)

    mincut_loss = ho_mincut_loss = 0
    # 1st order MinCUT loss
    if alpha < 1:
        diag_SAS = torch.einsum("ijj->ij", out_adj.clone())
        d_flat = torch.einsum("ijk->ij", adj.clone())
        d = _rank3_diag(d_flat)
        sds = torch.matmul(torch.matmul(s.transpose(1, 2), d), s)
        diag_SDS = torch.einsum("ijk->ij", sds) + EPS
        mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        mincut_loss = 1 / k * torch.mean(mincut_loss)

    # Higher order cut
    if alpha > 0:
        diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        d_flat = torch.einsum("ijk->ij", motif_adj)
        d = _rank3_diag(d_flat)
        diag_SDS = (
            torch.einsum("ijk->ij", torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
            + EPS
        )
        ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        ho_mincut_loss = 1 / k * torch.mean(ho_mincut_loss)

    # Combine ho and fo mincut loss.
    # We do not learn these coefficients yet
    hosc_loss = (1 - alpha) * mincut_loss + alpha * ho_mincut_loss

    # Orthogonality loss
    if mu == 0:
        ortho_loss = torch.tensor(0)
    else:
        if new_ortho:
            if s.shape[0] == 1:
                ortho_loss = (
                    (-torch.sum(torch.norm(s, p="fro", dim=-2)) / (num_nodes ** 0.5))
                    + k ** 0.5
                ) / (k ** 0.5 - 1)
            elif mask != None:
                ortho_loss = sum(
                    [
                        (
                            (
                                -torch.sum(
                                    torch.norm(
                                        s[i][: mask[i].nonzero().shape[0]],
                                        p="fro",
                                        dim=-2,
                                    )
                                )
                                / (mask[i].nonzero().shape[0] ** 0.5)
                                + k ** 0.5
                            )
                            / (k ** 0.5 - 1)
                        )
                        for i in range(batch_size)
                    ]
                ) / float(batch_size)
            else:
                ortho_loss = sum(
                    [
                        (
                            (
                                -torch.sum(torch.norm(s[i], p="fro", dim=-2))
                                / (num_nodes ** 0.5)
                                + k ** 0.5
                            )
                            / (k ** 0.5 - 1)
                        )
                        for i in range(batch_size)
                    ]
                ) / float(batch_size)
        else:
            # Orthogonality regularization.
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(k).type_as(ss)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
                dim=(-1, -2),
            )
            ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d + EPS)[:, None]
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, hosc_loss, mu * ortho_loss

def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


def train(loader, optimizer, criterion, model):
    overall_loss = []
    for batch in loader:
        model.train()
        optimizer.zero_grad()
        out, mc, o, s = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss += mc + o
        loss.backward()
        optimizer.step()
        overall_loss.append(loss)
    return torch.mean(torch.stack(overall_loss))

def evaluate(model, loader):
    model.eval()
    correct = 0
    for batch in loader:
        out, mc, o, s = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
    return correct / len(loader.dataset)

def get_data(name, train_ratio=0.6, test_ratio=0.2):
    dataset = TUDataset("TUDataset", name=name)
    dataset = dataset.shuffle()
    train_idx = int(len(dataset) * train_ratio)
    test_idx = int(len(dataset) * (1.0 - test_ratio))
    train_dataset = dataset[:train_idx]
    val_dataset = dataset[train_idx: test_idx]
    test_dataset = dataset[test_idx:]
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    # Load dataset
    train_dataset, val_dataset, test_dataset = get_data(name="MUTAG")

    train_loader = DataLoader(train_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Find max
    max_num_nodes = []
    for batch in train_dataset:
        max_num_nodes.append(batch.x.size(0))
    max_num_nodes = max(max_num_nodes)

    # Load model
    model = GNN(
        num_nodes=max_num_nodes,
        num_node_features=train_dataset.num_node_features,
        num_classes=train_dataset.num_classes,
        hidden_dim=[64, 64],
        mlp_hidden_dim=32,
        mu=0.1,
        cluster_ratio=0.25
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    for epoch in range(1, 200):
        loss = train(train_loader, optimizer, criterion, model)

    # Evaluate model
    test_acc = evaluate(model, test_loader)
    print("Performance accuracy is {:.4f}".format(test_acc))
