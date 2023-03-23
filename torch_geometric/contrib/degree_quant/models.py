import torch
import torch.nn.functional as F
from .gat_conv import GATConvMultiQuant, GATConvQuant
from .gcn_conv import GCNConvMultiQuant, GCNConvQuant
from .gin_conv import GINConvMultiQuant, GINConvQuant
from .linear import LinearQuantized
# from message_passing import MessagePassingMultiQuant
from .quantizer import make_quantizers
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.nn import global_mean_pool


class GIN(nn.Module):
    """

    Args:
        in_channels(int): Number of input features
        out_channels(int): Number of classes
        num_layers:(int): Number of GIN layers to use in the model
        hidden:(int): Hidden dimension for message passing and aggregation
        dq:(bool): Whether to use Degree Quant
        qypte:(str): The Integer precision for Degree Quant
        ste:(bool): Whether to use Straight-Through Estimation for the
        quantization.
        momentum:(int): Value of the momentum coefficient
        percentile:(int): Clips the values at the low and high percentile of
        the real value distribution
        sample_prop:(torch.Tensor): Probability of bernoulli mask for each node
        in the graph

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers: int,
        hidden: int,
        dq: bool,
        qypte: str,
        ste: bool,
        momentum: bool,
        percentile: int,
        sample_prop: torch.Tensor,
    ):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_dq = dq
        if dq is True:
            gin_layer = GINConvMultiQuant
        else:
            gin_layer = GINConvQuant

        lq, mq = make_quantizers(
            qypte,
            dq,
            False,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )
        lq_signed, _ = make_quantizers(
            qypte,
            dq,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )

        self.conv1 = gin_layer(
            ResettableSequential(
                Linear(self.in_channels, hidden),
                ReLU(),
                LinearQuantized(hidden, hidden, layer_quantizers=lq),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True,
            mp_quantizers=mq,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gin_layer(
                    ResettableSequential(
                        LinearQuantized(hidden, hidden,
                                        layer_quantizers=lq_signed),
                        ReLU(),
                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True,
                    mp_quantizers=mq,
                ))

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, self.out_channels,
                                    layer_quantizers=lq)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # NOTE: It is possible to use the same mask consistently or generate a
        # new mask per layer. For other experiments we used a per-layer mask
        # We did not observe major differences but we expect the impact will
        # be layer and dataset dependent. Extensive experiments assessing the
        # difference were not run, however, due to the high cost.
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index, mask)
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        x = global_mean_pool(x, batch)
        # NOTE: the linear layers from here do not contribute significantly to
        # run-time
        # Therefore you probably don't want to quantize these as it will likely
        # have an impact on performance.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # NOTE: This is a quantized final layer. You probably don't want to be
        # this aggressive in practice.
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCN(nn.Module):
    """

    Args:
        in_channels(int): Number of input features
        out_channels(int): Number of classes
        num_layers:(int): Number of GIN layers to use in the model
        hidden:(int): Hidden dimension for message passing and aggregation
        dq:(bool): Whether to use Degree Quant
        qypte:(str): The Integer precision for Degree Quant
        ste:(bool): Whether to use Straight-Through Estimation for the
        quantization.
        momentum:(int): Value of the momentum coefficient
        percentile:(int): Clips the values at the low and high percentile of
        the real value distribution
        sample_prop:(torch.Tensor): Probability of bernoulli mask for each node
        in the graph

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        hidden,
        dq,
        qypte,
        ste,
        momentum,
        percentile,
        sample_prop,
    ):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_dq = dq
        if dq is True:
            gcn_layer = GCNConvMultiQuant
        else:
            gcn_layer = GCNConvQuant

        lq, mq = make_quantizers(
            qypte,
            dq,
            False,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )
        lq_signed, _ = make_quantizers(
            qypte,
            dq,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )

        self.conv1 = gcn_layer(
            in_channels=self.in_channels, out_channels=hidden,
            nn=ResettableSequential(
                Linear(hidden, hidden), ReLU(),
                LinearQuantized(hidden, hidden, layer_quantizers=lq), ReLU(),
                BN(hidden)), layer_quantizers=lq_signed, mp_quantizers=mq)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gcn_layer(
                    in_channels=hidden, out_channels=hidden,
                    nn=ResettableSequential(
                        Linear(hidden, hidden), ReLU(),
                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                        ReLU(), BN(hidden)), layer_quantizers=lq_signed,
                    mp_quantizers=mq))

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, self.out_channels,
                                    layer_quantizers=lq)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):

        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index, mask)
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GAT(nn.Module):
    """

    Args:
        in_channels(int): Number of input features
        out_channels(int): Number of classes
        num_layers:(int): Number of GIN layers to use in the model
        hidden:(int): Hidden dimension for message passing and aggregation
        dq:(bool): Whether to use Degree Quant
        qypte:(str): The Integer precision for Degree Quant
        ste:(bool): Whether to use Straight-Through Estimation for the
        quantization.
        momentum:(int): Value of the momentum coefficient
        percentile:(int): Clips the values at the low and high percentile of
        the real value distribution
        sample_prop:(torch.Tensor): Probability of bernoulli mask for each node
        in the graph

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        hidden,
        dq,
        qypte,
        ste,
        momentum,
        percentile,
        sample_prop,
    ):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_dq = dq
        if dq is True:
            gat_layer = GATConvMultiQuant
        else:
            gat_layer = GATConvQuant

        lq, mq = make_quantizers(
            qypte,
            dq,
            False,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )
        lq_signed, _ = make_quantizers(
            qypte,
            dq,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )

        self.conv1 = gat_layer(
            in_channels=self.in_channels, out_channels=hidden,
            nn=ResettableSequential(
                Linear(hidden, hidden), ReLU(),
                LinearQuantized(hidden, hidden, layer_quantizers=lq), ReLU(),
                BN(hidden)), layer_quantizers=lq_signed, mp_quantizers=mq)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gat_layer(
                    in_channels=hidden, out_channels=hidden,
                    nn=ResettableSequential(
                        Linear(hidden, hidden), ReLU(),
                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                        ReLU(), BN(hidden)), layer_quantizers=lq_signed,
                    mp_quantizers=mq))

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, self.out_channels,
                                    layer_quantizers=lq)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):

        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index, mask)
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class ResettableSequential(Sequential):
    """
        A wrapper for Torch sequential to reset the layer parameters without
        iterating each time.

    """
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


def evaluate_prob_mask(data):
    """
    This model return the probability mask of the input graph using bernoulli
    random masking of each node.

    Parameters
    ----------
    data: Single graph object from PyG

    """
    return torch.bernoulli(data.prob_mask).to(torch.bool)
