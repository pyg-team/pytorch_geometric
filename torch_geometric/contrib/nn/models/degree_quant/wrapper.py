import torch 
import torch.nn.functional as F
from quantizer import LinearQuantized, make_quantizers
from utils import ResettableSequential,evaluate_prob_mask
from torch.nn import Linear, ReLU, BatchNorm1d as BN
from torch import nn 
from torch_geometric.nn import global_mean_pool
from messagepassing import *
from GAT_conv import *
from GCN_conv import  * 
from GIN_conv import *


class GIN(nn.Module):
    def __init__(
        self,
        dataset,
        num_layers,
        hidden,
        dq,
        qypte,
        ste,
        momentum,
        percentile,
        sample_prop,
    ):
        super(GIN, self).__init__()

        self.is_dq = dq
        if dq == True:
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
                Linear(dataset.num_features, hidden),
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
                        LinearQuantized(hidden, hidden, layer_quantizers=lq_signed),
                        ReLU(),
                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True,
                    mp_quantizers=mq,
                )
            )

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq)

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
        # NOTE: the linear layers from here do not contribute significantly to run-time
        # Therefore you probably don't want to quantize these as it will likely have 
        # an impact on performance.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # NOTE: This is a quantized final layer. You probably don't want to be
        # this aggressive in practice.
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCN(nn.Module):

    def __init__(
        self,
        dataset,
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

        self.is_dq = dq
        if dq == True:
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
                            in_channels = dataset.num_features,
                            out_channels = hidden,
                            nn= ResettableSequential(Linear(hidden, hidden),
                                                        ReLU(),
                                                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                                                        ReLU(),
                                                        BN(hidden)
                                                    ),
                            improved=False,
                            cached=False,
                            bias=True,
                            normalize=True,
                            layer_quantizers=lq_signed,
                            mp_quantizers=mq

        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gcn_layer(
                            in_channels = hidden,
                            out_channels = hidden,
                            nn= ResettableSequential(Linear(hidden, hidden),
                                                        ReLU(),
                                                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                                                        ReLU(),
                                                        BN(hidden)
                                                    ),
                            improved=False,
                            cached=False,
                            bias=True,
                            normalize=True,
                            layer_quantizers=lq_signed,
                            mp_quantizers=mq
                )
            )

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq)

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

    def __init__(
        self,
        dataset,
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

        self.is_dq = dq
        if dq == True:
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
            in_channels = dataset.num_features,
            out_channels = hidden,
            nn= ResettableSequential(
                Linear(hidden, hidden),
                ReLU(),
                LinearQuantized(hidden, hidden, layer_quantizers=lq),
                ReLU(),
                BN(hidden)
            ),
            improved=False,
            cached=False,
            bias=True,
            normalize=True,
            layer_quantizers=lq_signed,
            mp_quantizers=mq
            )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gcn_layer(
                in_channels = hidden,
                out_channels = hidden,
                nn = ResettableSequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    LinearQuantized(hidden, hidden, layer_quantizers=lq),
                    ReLU(),
                    BN(hidden)
                    ),
                improved=False,
                cached=False,
                bias=True,
                normalize=True,
                layer_quantizers=lq_signed,
                mp_quantizers=mq
                )
            )

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq)

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