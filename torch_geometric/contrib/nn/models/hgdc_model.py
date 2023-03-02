import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T


class HGDCNet(torch.nn.Module):
    r"""
    HGDC is a novel GCN-based method of heterophilic graph diffusion convolutional
    networks to boost cancer driver gene identification. HGDC first introduces graph
    diffusion to generate an auxiliary network for capturing the structurally similar
    nodes in a biomolecular network. Then, HGDC designs an improved message aggregation
    and propagation scheme to adapt to the heterophilic setting of biomolecular networks,
    alleviating the problem of driver gene features being smoothed by its neighboring
    dissimilar genes. Finally, HGDC uses a layer-wise attention classifier to predict
    the probability of one gene being a cancer driver gene.
    """
    def __init__(self, in_channels, hidden_channels, ppr_alpha, ppr_eps, net_avg_deg, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ppr_alpha = ppr_alpha
        self.ppr_eps = ppr_eps
        self.net_avg_deg = net_avg_deg
        self.weights = weights

        self.linear1 = Linear(self.in_channels, self.hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(self.hidden_channels, self.hidden_channels, add_self_loops = False) #
        self.conv_k2_1 = GCNConv(2 * self.hidden_channels, self.hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * self.hidden_channels, self.hidden_channels, add_self_loops=False)
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(self.hidden_channels, self.hidden_channels, add_self_loops = False) #
        self.conv_k2_2 = GCNConv(2 * self.hidden_channels, self.hidden_channels, add_self_loops = False)
        self.conv_k3_2 = GCNConv(2 * self.hidden_channels, self.hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(self.hidden_channels, 1)
        self.linear_r1 = Linear(2*self.hidden_channels, 1)
        self.linear_r2 = Linear(2*self.hidden_channels, 1)
        self.linear_r3 = Linear(2*self.hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([self.weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([self.weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([self.weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([self.weights[3]]), requires_grad=True)

    def _generate_auxiliary_network(self,data):
        """
        Construct the auxiliary graph based on the original graph.
        :param args: Arguments received from command line
        :param dataset (class: dictionary): Dataset loaded from a pickle file
        :return (class: torch.LongTensor): edges indexed by ids
        """
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=self.ppr_alpha, eps=self.ppr_eps),
                    sparsification_kwargs=dict(method='threshold', avg_degree=self.net_avg_deg),
                    exact=True)
        data = gdc(data)
        print('Auxiliary network is obtained...')
        return data.edge_index

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = self._generate_auxiliary_network(data)

        edge_index_1, _ = dropout_adj(edge_index_1, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)
        edge_index_2, _ = dropout_adj(edge_index_2, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 =self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out






