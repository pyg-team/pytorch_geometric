import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU
from torch.nn import Identity

import torch_geometric.transforms as T
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool import knn
#from torch_geometric.nn.conv import PointTransformerConv

from torch_cluster import knn_graph
from torch_cluster import fps
from torch_scatter import scatter_max

from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import reset


class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i =  \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \alpha_{i,j} \left(\mathbf{W}_3
        \mathbf{x}_j + \delta_{ij} \right),

    where the attention coefficients :math:`\alpha_{i,j}` and
    positional embedding :math:`\delta_{ij}` are computed as

    .. math::
        \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
        (\mathbf{W}_1 \mathbf{x}_i - \mathbf{W}_2 \mathbf{x}_j +
        \delta_{i,j}) \right)

    and

    .. math::
        \delta_{i,j}= h_{\mathbf{\Theta}}(\mathbf{p}_i - \mathbf{p}_j),

    with :math:`\gamma_\mathbf{\Theta}` and :math:`h_\mathbf{\Theta}`
    denoting neural networks, *i.e.* MLPs, and
    :math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
    each point.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        pos_nn : (torch.nn.Module, optional): A neural network
            :math:`h_\mathbf{\Theta}` which maps relative spatial coordinates
            :obj:`pos_j - pos_i` of shape :obj:`[-1, 3]` to shape
            :obj:`[-1, out_channels]`.
            Will default to a :class:`torch.nn.Linear` transformation if not
            further specified. (default: :obj:`None`)
        attn_nn : (torch.nn.Module, optional): A neural network
            :math:`\gamma_\mathbf{\Theta}` which maps transformed
            node features of shape :obj:`[-1, out_channels]`
            to shape :obj:`[-1, out_channels]`. (default: :obj:`None`)
        add_self_loops (bool, optional) : If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, share_planes: int = 8, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.share_planes = share_planes

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(3, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels)#, bias=False)
        self.lin_src = Linear(in_channels[0], out_channels)#, bias=False)
        self.lin_dst = Linear(in_channels[1], out_channels)#, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        delta = self.pos_nn(pos_i - pos_j)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return (alpha.unsqueeze(1) * (x_j + delta).view(-1, self.share_planes, x_j.shape[1] // self.share_planes)).view(-1,x_j.shape[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')



class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels, bias=False)
        self.lin_out = Lin(out_channels, out_channels, bias=False)

        self.pos_nn = Seq(MLP([3, 3]), Lin(3, out_channels))

        self.attn_nn = Seq(
            BN(out_channels), ReLU(), MLP([out_channels, out_channels // 8]),
            Lin(out_channels // 8, out_channels // 8)
            # NB : last should be // 8 according to original code, but custom
            # implementation makes it impossible at the moment
        )

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

        self.bn1 = BN(in_channels)
        self.bn2 = BN(in_channels)
        self.bn3 = BN(in_channels)

    def forward(self, x, pos, edge_index):
        x_skip = x.clone()
        x = self.bn1(self.lin_in(x)).relu()
        x = self.bn2(self.transformer(x, pos, edge_index)).relu()
        x = self.bn3(self.lin_out(x))
        x = (x + x_skip).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([3 + in_channels, out_channels], bias=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        relative_pos = pos[id_k_neighbor[1]] - pos[id_k_neighbor[0]]
        grouped_x = torch.cat([relative_pos, x[id_k_neighbor[1]]], axis=1)

        # transformation of features through a simple MLP
        x = self.mlp(grouped_x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x, id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True, bias=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class TransitionSummit(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp_sub = Seq(Lin(in_channels, in_channels), ReLU())
        self.mlp = MLP([2 * in_channels, in_channels])

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.mlp(torch.cat((x, self.mlp_sub(x)), 1))
        return x


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], bias=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        blocks = [1, 2, 3, 5, 2]

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(
                Seq(
                    TransitionDown(in_channels=dim_model[i],
                                   out_channels=dim_model[i + 1], k=self.k),
                    *[
                        TransformerBlock(in_channels=dim_model[i + 1],
                                         out_channels=dim_model[i + 1])
                        for k in range(blocks[1:][i])
                    ]))

        # summit layers
        #self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)
        self.mlp_summit = TransitionSummit(dim_model[-1])

        self.transformer_summit = Seq(*[
            TransformerBlock(
                in_channels=dim_model[-1],
                out_channels=dim_model[-1],
            ) for i in range(1)
        ])

        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(
                Seq(
                    TransitionUp(in_channels=dim_model[n - i],
                                 out_channels=dim_model[n - i - 1]),
                    *[
                        TransformerBlock(in_channels=dim_model[n - i - 1],
                                         out_channels=dim_model[n - i - 1])
                        for k in range(1)
                    ]))

        # class score computation
        self.mlp_output = Seq(MLP([dim_model[0], dim_model[0]]),
                              Lin(dim_model[0], out_channels))

    def forward(self, x, pos, batch=None):
        # add dummy features in case there is none
        x = pos if x is None else torch.cat((pos, x), 1)

        out_x = []
        out_pos = []
        out_batch = []
        edges_index = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)
        edges_index.append(edge_index)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):

            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            for layer in self.encoders[i][1:]:
                x = layer(x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
            edges_index.append(edge_index)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        for layer in self.transformer_summit:
            x = layer(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                    pos=out_pos[-i - 2],
                                    pos_sub=out_pos[-i - 1],
                                    batch_sub=out_batch[-i - 1],
                                    batch=out_batch[-i - 2])

            edge_index = edges_index[-i - 2]

            for layer in self.decoders[i][1:]:
                x = layer(x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


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

    preds = [
    ]  #to compute IoU on the full dataset instead of a per-batch basis
    labels = []  # we will stack the predictions and labels
    for i, data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)

    preds_tensor = torch.hstack(preds).type(torch.LongTensor).cpu()
    labels_tensor = torch.hstack(labels).type(torch.LongTensor).cpu()

    i, u = torch.zeros((13)), torch.zeros((13))
    # intersection_and_union does one-hot encoding, making the full labels
    # matrix too large to fit all at once so we do it in two times

    i_sub, u_sub = i_and_u(preds_tensor[:2000000], labels_tensor[:2000000],
                           13)  #, data.batch)
    i += i_sub
    u += u_sub

    i_sub, u_sub = i_and_u(preds_tensor[2000000:], labels_tensor[2000000:],
                           13)  #, data.batch)
    i += i_sub
    u += u_sub

    iou = i / u
    iou = iou[torch.isnan(iou) == False].mean()
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
    pre_transform = None  #T.Compose([])#T.NormalizeScale()
    train_dataset = S3DIS(path, test_area=5, train=True, transform=transform,
                          pre_transform=pre_transform)
    test_dataset = S3DIS(path, test_area=5, train=False,
                         pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    print('done')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(6 + 3, train_dataset.num_classes,
                dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                                gamma=0.1)
    #ptimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=0.0001)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)

    loss_history = []
    test_iou_history = []
    for epoch in range(1, 100):
        train()
        iou = test(test_loader)
        test_iou_history.append(iou)
        print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
        scheduler.step()
