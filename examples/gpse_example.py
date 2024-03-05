import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

from torch_geometric.datasets import ZINC
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.nn import (
    GPSE,
    MLP,
    GCNConv,
    GINConv,
    GPSENodeEncoder,
    Linear,
    global_mean_pool,
)
from torch_geometric.nn.models.gpse import precompute_GPSE
from torch_geometric.transforms import AddGPSE


def load_ZINC(args):
    """Load the ZINC dataset, and generate GPSE encodings for the graphs if
    args.gpse is not None.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    'ZINC_subset')
    gpse_model = GPSE.from_pretrained(
        name=args.gpse,
        root=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                      'GPSE_pretrained')) if args.gpse else None

    if args.gpse and args.as_transform:
        # WARNING: Using a pre_transform will save the encodings to disk,
        # meaning any future runs will use the saved encodings. This is useful
        # for speeding up computation, but may not be desirable, e.g. when
        # experimenting with different pre-trained GPSE models. Alternatively,
        # AddGPSE can be used as a regular transform, which will compute the
        # encodings on-the-fly, but this will slow down the data loading
        # process.
        train_dataset = ZINC(
            path, subset=True, split='train',
            pre_transform=AddGPSE(gpse_model, use_vn=True,
                                  rand_type='NormalSE'))
        test_dataset = ZINC(
            path, subset=True, split='val',
            pre_transform=AddGPSE(gpse_model, use_vn=True,
                                  rand_type='NormalSE'))
    else:
        train_dataset = ZINC(path, subset=True, split='train')
        test_dataset = ZINC(path, subset=True, split='val')

        if args.gpse:
            precompute_GPSE(gpse_model, train_dataset)
            precompute_GPSE(gpse_model, test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    return train_loader, test_loader


class IdentityNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, batch):
        return batch


class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_types=28):
        super().__init__()

        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


class GNNStackStage(torch.nn.Module):
    """Simple Staging mechanism that stacks an arbitrary number of GNN layers
    with skip connections and L2 normalization.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
        conv_type (str): Type of graph convolution in GNN
        stage_type (str): Type of skip connections. Options: 'skipsum' or
        'skipconcat', any other value means no skip connections.
        l2norm (bool): Whether to apply L2 normalization to outputs
    """
    def __init__(self, dim_in, dim_out, num_layers, conv_type='gcn',
                 stage_type='skipsum', l2norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.l2norm = l2norm
        conv_dict = {'gcn': GCNConv, 'gin': GINConv}

        for i in range(num_layers):
            if stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = conv_dict[conv_type](d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch.x = layer(batch.x, batch.edge_index)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif self.stage_type == 'skipconcat' and \
                    i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if self.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class GPSEPlusGNN(torch.nn.Module):
    """A GPSE encoder paired with a GNN module. Consists of:
    - encoder1: An optional encoder that is used to encode raw node features,
        common practice for biochemistry datasets. ZINC uses
        :class:`TypeDictNodeEncoder`, while ogbg-mol* datasets typically use
        :class:`~torch_geometric.graphgym.models.encoder.AtomEncoder`. If
        'none', an :class`IdentityNodeEncoder` is passed that returns the
        inputs as-is.
    - encoder2: GPSE encoder that adds precomputed GPSE encodings in the
        dataset to node features if :obj:`gpse` is :obj:`True`. Otherwise is
        replaced by a linear layer that maps the :obj:`encoder1` outputs to
        the correct dimension.
    - premp: 2-layer MLP before message-passing.
    - gnn: Stacked <num_layers> message-passing layers of :obj:`conv_type`.
    - postmp: 1-layer MLP after message-passing to map GNN node states to a
        single output (for ZINC regression task). For classification tasks,
        :obj:`num_classes` outputs with softmax activation would be required.

    Args:
        dim_emb (int): Dimension of embedding outputs. Equals dimension of
            :obj:`encoder1` outputs (dim_emb - dim_pe_out) and
            :class:`~torch_geometric.nn.GPSENodeEncoder` outputs (dim_pe_out).
        dim_conv (int): Dimension of GNN message-passing layers.
        conv_type (str): Type of graph convolution in GNN.
        num_layers (int): Number of GNN layers.
        dim_pe_in (int): Original dimension of posenc_GPSE, i.e. the
            precomputed GPSE encodings.
        dim_pe_out (int): Desired dimension of GPSE-derived node features,
            mapped from the original GPSE encodings via GPSENodeEncoder.
        encoder (str): Encoding applied to raw node features.
        gpse (bool): Whether to use GPSE encodings.
    """
    def __init__(self, dim_emb, dim_conv, conv_type, num_layers, dim_pe_in,
                 dim_pe_out, encoder='none', gpse=True):
        super().__init__()
        encoder_dict = {
            'none': IdentityNodeEncoder,
            'Atom': AtomEncoder,
            'TypeDict': TypeDictNodeEncoder
        }

        self.encoder1 = encoder_dict[encoder](dim_emb - dim_pe_out)
        self.encoder2 = GPSENodeEncoder(dim_emb, dim_pe_in, dim_pe_out,
                                        expand_x=False) if gpse else (Linear(
                                            dim_emb -
                                            dim_pe_out, dim_emb, bias=True))
        self.premp = MLP([dim_emb, dim_emb, dim_conv])
        self.gnn = GNNStackStage(dim_conv, dim_conv, num_layers, conv_type)
        self.postmp = MLP([dim_conv, 1])

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        batch.x = self.premp(batch.x)
        batch = self.gnn(batch)
        batch = global_mean_pool(batch.x, batch.batch)
        batch = F.dropout(batch, p=0.5, training=self.training)
        batch = self.postmp(batch)
        return batch


def train(loader):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        pred = out.squeeze(-1) if out.ndim > 1 else out
        true = data.y.squeeze(-1) if data.y.ndim > 1 else data.y

        loss = F.mse_loss(pred, true)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)

        pred = out.squeeze(-1) if out.ndim > 1 else out
        true = data.y.squeeze(-1) if data.y.ndim > 1 else data.y

        loss = F.mse_loss(pred, true)
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPSE Example')

    parser.add_argument(
        '--gpse', type=str, default=None, const='molpcba', nargs='?',
        choices=['molpcba', 'zinc', 'pcqm4mv2', 'geom',
                 'chembl'], help='which model weights to use '
        '(default: %(default)s)')
    parser.add_argument(
        '--as_transform', action='store_true',
        help='Whether to apply GPSE as a pre_transform to the '
        'dataset or not')

    args = parser.parse_args()
    train_loader, test_loader = load_ZINC(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPSEPlusGNN(dim_emb=64, dim_conv=128, conv_type='gcn',
                        num_layers=8, dim_pe_in=512, dim_pe_out=32,
                        encoder='TypeDict', gpse=args.gpse).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=5e-4)

    num_epochs = 100
    times = []
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        loss = train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
        times.append(time.time() - start)
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
