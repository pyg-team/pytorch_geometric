import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Dropout, BatchNorm2d, BatchNorm1d, Dropout2d, Parameter

from torch_geometric.nn.kge import KGEModel


class ConvE(KGEModel):
    r"""The ConvE model from the `"Convolutional 2D Knowledge Graph Embeddings"
    <https://arxiv.org/abs/1707.01476>`_ paper.

    :class:`ConvE` uses 2D convolutions over embeddings to model knowledge graphs.
    The scoring function is defined as:

    .. math::
        d(h,r,t) = f(vec(f([h; r] * w)) W)e_t

    where :math:`f` denotes a non-linear activation function, :math:`*` denotes
    the convolution operator, and :math:`h, r, t` are the head, relation, and
    tail embeddings.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        embedding_height (int, optional): The height of the reshaped embedding.
            (default: :obj:`10`)
        input_drop (float, optional): Dropout rate for input embeddings.
            (default: :obj:`0.2`)
        hidden_drop (float, optional): Dropout rate for hidden layers.
            (default: :obj:`0.3`)
        feat_drop (float, optional): Dropout rate for feature maps.
            (default: :obj:`0.2`)
        use_bias (bool, optional): If set to :obj:`True`, adds a bias term.
            (default: :obj:`True`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        embedding_height: int = 10,
        input_drop: float = 0.2,
        hidden_drop: float = 0.3,
        feat_drop: float = 0.2,
        use_bias: bool = True,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.embedding_height = embedding_height
        self.embedding_width = hidden_channels // embedding_height

        # Input processing layers
        self.inp_drop = Dropout(input_drop)
        self.hidden_drop = Dropout(hidden_drop)
        self.feature_map_drop = Dropout2d(feat_drop)

        # Batch normalization layers
        self.bn0 = BatchNorm2d(1)
        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm1d(hidden_channels)

        # Convolution layer
        self.conv = Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)

        # Calculate the size of the flattened conv output
        conv_output_height = embedding_height * 2 - 3 + 1
        conv_output_width = self.embedding_width - 3 + 1
        self.conv_output_size = 32 * conv_output_height * conv_output_width

        # Fully connected layer and bias
        self.fc = torch.nn.Linear(self.conv_output_size, hidden_channels)
        self.register_parameter('b', Parameter(torch.zeros(num_nodes)))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.node_emb.weight)
        torch.nn.init.xavier_normal_(self.rel_emb.weight)
        self.conv.reset_parameters()
        self.bn0.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.fc.reset_parameters()
        self.b.data.zero_()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        # Get embeddings
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)

        # Reshape embeddings
        head = head.view(-1, 1, self.embedding_height, self.embedding_width)
        rel = rel.view(-1, 1, self.embedding_height, self.embedding_width)

        # Stack and process
        stacked_inputs = torch.cat([head, rel], dim=2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        # Convolution
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Score computation
        x = torch.mm(x, self.node_emb.weight.transpose(1, 0))
        x += self.b.expand_as(x)

        # For training, only use scores for the target tail entities
        if self.training:
            x = torch.gather(x, 1, tail_index.unsqueeze(-1)).squeeze(-1)
        
        return x

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        scores = torch.cat([pos_score, neg_score], dim=0)
        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target) 