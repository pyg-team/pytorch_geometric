import torch
import torch.nn as nn
from torch_geometric.nn import GENConv
from torch_geometric.nn import DeepGCNs
from torch_geometric.nn.conv.utils.helpers import norm_layer
import torch.nn.functional as F
import logging


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 50:
            self.checkpoint_grad = True
            self.ckp_k = 3

        if (self.learn_t or self.learn_p) and self.num_layers > 15:
            self.checkpoint_grad = True
            self.ckp_k = 9

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            print('GraphConv->LN/BN->ReLU->Dense')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU->Plain')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.skip_block = DeepGCNs(self.block)
        self.norms = torch.nn.ModuleList()
        self.act_layer = nn.ReLU(inplace=False)
        self.dropout_layer = nn.Dropout(self.dropout, inplace=False)

        for layer in range(self.num_layers):

            if conv == 'gen':
                if self.block == 'dense' and layer > 1:
                    in_channels = hidden_channels * layer
                    out_channels = hidden_channels
                    last_channels = in_channels + out_channels
                else:
                    in_channels = hidden_channels
                    out_channels = hidden_channels
                    last_channels = hidden_channels
                gcn = GENConv(in_channels, out_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm,
                              learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge,
                              edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.node_features = torch.load(node_features_file_path).to(
          args.device)

        self.node_one_hot_encoder = torch.nn.Linear(8, 8)
        self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)

        self.edge_encoder = torch.nn.Linear(8, hidden_channels)

        self.node_pred_linear = torch.nn.Linear(last_channels, num_tasks)

    def forward(self, x, node_index, edge_index, edge_attr):

        node_features_1st = self.node_features[node_index]
        node_features_2nd = self.node_one_hot_encoder(x)
        # concatenate
        node_features = torch.cat((node_features_1st, node_features_2nd),
                                  dim=1)
        h = self.node_features_encoder(node_features)

        edge_emb = self.edge_encoder(edge_attr)

        if self.block == 'res+':
            h = self.gcns[0](h, edge_index, edge_emb)

            if self.checkpoint_grad:
                for layer in range(1, self.num_layers):
                    if layer % self.ckp_k != 0:
                        h = self.skip_block((h, edge_index, edge_emb),
                                            self.gcns[layer],
                                            self.norms[layer-1],
                                            self.act_layer, self.dropout_layer,
                                            ckpt_grad=True)
                    else:
                        h = self.skip_block((h, edge_index, edge_emb),
                                            self.gcns[layer],
                                            self.norms[layer-1],
                                            self.act_layer, self.dropout_layer,
                                            ckpt_grad=False)

            else:
                for layer in range(1, self.num_layers):
                    h = self.skip_block((h, edge_index, edge_emb),
                                        self.gcns[layer],
                                        self.norms[layer-1],
                                        self.act_layer, self.dropout_layer)

            h = self.act_layer(self.norms[self.num_layers-1](h))
            h = self.dropout_layer(h)

            return self.node_pred_linear(h)

        elif self.block in ['res', 'dense', 'plain']:

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.checkpoint_grad:
                for layer in range(1, self.num_layers):
                    if layer % self.ckp_k != 0:
                        h = self.skip_block((h, edge_index, edge_emb),
                                            self.gcns[layer],
                                            self.norms[layer],
                                            self.act_layer, self.dropout_layer,
                                            ckpt_grad=True)
                    else:
                        h = self.skip_block((h, edge_index, edge_emb),
                                            self.gcns[layer],
                                            self.norms[layer],
                                            self.act_layer, self.dropout_layer,
                                            ckpt_grad=False)
            else:
                for layer in range(1, self.num_layers):
                    h = self.skip_block((h, edge_index, edge_emb),
                                        self.gcns[layer],
                                        self.norms[layer],
                                        self.act_layer, self.dropout_layer)

            return self.node_pred_linear(h)

        else:
            raise Exception('Unknown block Type')

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
