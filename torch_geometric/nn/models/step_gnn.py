import random

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import ChebConv, knn_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class TempConvBlock(nn.Module):
    r"""This is a temporal convolution block, to be used in the spatio-temporal convolution block.
    It allows one to extract temporal dynamic behavior by using a convolution across the entire time series axis.
    As opposed to common RNN-based models for time series analysis, it is easily parallelizable.

    This is described in the paper at <https://arxiv.org/pdf/1709.04875>.
      "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting."
    See Section 3.3 and 3.4 of the paper for further description of implementation details.

    For a single batch instance of multivariate time series data of shape (channels, num_vars, time_steps),
    this uses two 2D convolutions with kernel (1, _) on the last two dimensions, to learn temporal and node dependencies.
    These results are combined via the GLU (Gated Linear Unit) commonly used in transformer architectures,
    as (conv1_result) * sigmoid(conv2_results), where * denotes elementwise multiplication.
    Then, a third convolution layer is applied to the data (see equation (8) in the paper) and added to the output, and ReLU is taken.

    This code is also based off an existing implementation in Pytorch Geometric Temporal, which does not address the same_output_len issue below.
      https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/stgcn.html#TemporalConv

    Args:
      in_channels (int): Input channel dim.
      out_channels (int): Output channel dim.
      kernel (int): Conv kernel size. Default is 3.
      same_output_len (bool): Default is True. The paper above uses no padding in conv, which means the time steps decrease by (kernel - 1)
        with each convolution. For compatibility with STGCN later (where time steps are the same), we generalize this to use kernel and
        padding sizes so as to not change the time steps. When this argument is set to True, padding is used, and it is ignored otherwise.
      device (str): Device of block. Default is 'cpu'.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3,
                 same_output_len: bool = True, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Define kernel and padding size based on whether time steps should stay the same or change.
        self.same_output_len = same_output_len
        if self.same_output_len:
            self.kernel = (1, 2 * kernel + 1)
            self.padding = (0, kernel)
        else:
            self.kernel = (1, kernel)
            self.padding = None

        # Temporal block uses 3 convolution layers.
        self._convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, self.kernel,
                      padding=self.padding, device=self.device)
            for i in range(3)
        ])

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Args:
          data (torch.FloatTensor): Input data, shape (batch_sz, time_steps, num_vars, in_channels).

        Returns:
          out (torch.FloatTensor): Output data, shape (batch_sz, time_steps, num_vars, out_channels) if self.same_output_len.
            Otherwise, it will be (batch_sz, time_steps - (conv_kernel - 1), num_vars, out_channels).
        """
        data = data.transpose(1, 3)
        out = self._convs[0](data) * F.sigmoid(
            self._convs[1](data))  # First 2 convolutions and GLU.
        out += self._convs[2](data)
        out = F.relu(out)  # Eqn (8) in the paper.
        out = out.transpose(1, 3)
        return out


class SpatioTempConvBlock(nn.Module):
    r"""This is a spatio-temporal convolution block, using Chebyshev convolution blocks and temporal convolution blocks.
    The goal is to combine temporal representations (temp blocks) with spatial representations (Chebyshev blocks) into a spatio-temporal representation.
    The original version (and PyG-Temporal implementation below) consists of a "sandwich" architecture with 1 Chebyshev block:
      TempConvBlock -> ChebConvBlock -> ReLU -> TempConvBlock -> BatchNorm.
    We allow for more general sandwich architectures of the form with any number of Chebyshev blocks:
      TempConvBlock -> ChebConvBlock -> ReLU -> ChebConvBlock -> ReLU -> ... -> ChebConvBlock -> ReLU -> TempConvBlock -> BatchNorm.
    Stacking the Chebyshev layers improves expressivity, and increases the receptive field via larger multi-hop neighborhoods (see existing works like ChebNet).

    This is described in the paper at <https://arxiv.org/pdf/1709.04875>.
      "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting."
    See Section 3.3 and 3.4 of the paper for further description of implementation details.

    This code is also based off an existing implementation in Pytorch Geometric Temporal, but generalizes it to include more Chebsyhev blocks than 1.
      https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/stgcn.html#TemporalConv

    Args:
      num_vars (int): Number of variables/nodes in multivariate time series data.
      in_channels (int): Input channel dim.
      hidden_channels (int): Hidden channel dim to be used in Chebyshev conv blocks.
      out_channels (int): Output channel dim.
      kernel (int): Conv kernel size for temporal conv blocks. Default is 3.
      filter (int): Chebyshev filter size K for Chebyshev conv block. Default is 5.
        Here, K - 1 is the degree of the polynomial approximation used for the graph conv operator in Chebyshev block.
      num_cheb_blocks (int): Number of sandwiched Chebyshev blocks between temporal conv blocks. Default is 1.
      device (str): Device of block. Default is 'cpu'.
    """
    def __init__(self, num_vars: int, in_channels: int, hidden_channels: int,
                 out_channels: int, kernel: int = 3, filter: int = 5,
                 num_cheb_blocks: int = 1, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Relevant shapes and parameters to define temporal and Chebyshev convolutions.
        self.num_vars = num_vars
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.filter = filter
        self.num_cheb_blocks = num_cheb_blocks

        # Temporal convolution block for input and output projections. Make sure time_steps remains the same with same_output_len=True.
        self._temp_conv1 = TempConvBlock(in_channels, hidden_channels,
                                         kernel=kernel, same_output_len=True)
        self._temp_conv2 = TempConvBlock(hidden_channels, out_channels,
                                         kernel=kernel, same_output_len=True)

        # The middle of the sandwich architecture consists of a number of Chebyshev blocks followed by ReLU's.
        self._cheb_convs = nn.ModuleList([
            ChebConv(hidden_channels, hidden_channels, filter)
            for i in range(num_cheb_blocks)
        ])

        # Batch norm which operates on 2D input.
        self._batch_norm = nn.BatchNorm2d(num_vars)

        # Move all learnable layers above to device.
        self.to(self.device)

    def forward(self, data: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """Args:
          data (torch.FloatTensor): Input time series data of shape (batch_sz, time_steps, num_vars, in_channels).
          edge_index (torch.LongTensor): Graph edges in index format, of shape (num_vars, 2).
          edge_weight (torch.FloatTensor): (Optional) Weights of edges.
            Note: This is required if we want to backprop through the adjacency matrix.

        Returns:
          out (torch.FloatTensor): Output data, shape (batch_sz, time_steps, num_vars, out_channels).
            See TempConvBlock argument same_output_len, which must be set to true in this module. It makes sure the time_steps remains the same.
        """
        batch_sz, time_steps, _, _ = data.shape

        out_temp = self._temp_conv1(data)
        out = torch.zeros(out_temp.shape).to(out_temp.device)

        # Chebyshev conv in PyG does not batch, so we must loop over batches and time steps.
        for i in range(self.num_cheb_blocks):
            for b in range(batch_sz):
                for t in range(time_steps):
                    out[b][t] = F.relu(self._cheb_convs[i](
                        out_temp[b][t], edge_index, edge_weight=edge_weight))

        out = self._temp_conv2(out)

        # Reshape to (batch_sz, num_vars, time_steps, out_channels) temporarily.
        out = out.transpose(1, 2)
        out = self._batch_norm(
            out
        )  # Feature dim (num_vars) for BatchNorm2D should be out.shape[1].
        out = out.transpose(1, 2)
        return out


class EnhancedSTGCN(nn.Module):
    """This is an enhanced STGCN (spatio-temporal graph convolutional network).
    The vanilla STGCN stacks spatio-temporal convolution blocks, with a regression layer (MLP) for final predictions.
    While the vanilla version operates only on the actual input data, the enhanced version also uses ENCODED REPRESENTATIONS of the data.
    It also takes in a LEARNED DEPENDENCY GRAPH between time series variables, which is described in more detail in the GraphStructureLearner module.
    The encoded data must be provided as an argument to forward, but can be generated through a number of methods (like STEP pre-training, in the below paper).

    This is described in the paper at <https://arxiv.org/abs/2206.09113>.
      "Pre-training-Enhanced Spatial-Temporal Graph Neural Network For Multivariate Time Series Forecasting."

    The authors of the original paper use GraphWaveNet as the STGNN backend, instead of the vanilla STGCN framework that we described above.
    GraphWaveNet is not straightforward to implement from scratch in PyG (due to various dilated graph conv layers), and they do not use PyG in their repo code.
    On the other hand, STGCN makes use of Chebyshev conv layers, which are readily available in PyG and easy to backprop through.

    The enhanced version of the STGCN works as follows. It adds the representations from:
      a) The vanilla STGCN, applied on the last "patch" of each contiguous chunk of time series data (the last patch_length time steps),
      b) A separate MLP, applied on the encoded representations of the same last "patch" as above.
    The MLP output dimension is shaped so that these representations are compatible (same shape).
    Finally, their sum is passed through a final linear layer to get the reconstructed first "patch" of the next contiguous chunk of time series data.

    During training, the loss from this component is a simple MAE between the true time series patch and the reconstructed patch.
    There is no batching in this module, because it relies intimately on the GraphStructureLearner module for training, which cannot batch kNN.

    Args:
      num_vars (int): Number of variables/nodes.
      patch_length (int): Number of time steps for a patch.
      data_channels (int): Input channel dim for initial data.
      latent_channels (int): Channel dim for encoded representations.
      hidden_channels (int): Hidden channel dim in STGCN.
      out_channels (int): Output channel dim for both STGCN and MLP.
      kernel (int): Conv kernel size in STGCN. Default is 3.
      filter (int): Chebyshev filter size in STGCN. Default is 5.
      num_cheb_blocks (int): Number of sandwiched Chebyshev blocks between temporal conv blocks. Default is 1.
      num_stconv_blocks (int): Number of stacked spatio temporal conv blocks in the STGCN. Default is 1.
      num_mlp_blocks (int): Number of linear layers in the MLP minus 1. (Always want at least 2 linear layers.) Default is 1.
      mlp_ratio (float): Default is 2.0. Means that intermediate layers of MLP are size mlp_ratio times as big as input dim.
      device (str): Device of block. Default is 'cpu'.
    """
    def __init__(self, num_vars: int, patch_length: int, data_channels: int,
                 latent_channels: int, hidden_channels: int, out_channels: int,
                 kernel: int = 3, filter: int = 5, num_cheb_blocks: int = 1,
                 num_stconv_blocks: int = 1, num_mlp_blocks: int = 1,
                 mlp_ratio: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Relevant shapes to define spatio-temporal convolution blocks and MLP layer.
        self.num_vars = num_vars
        self.patch_length = patch_length
        self.data_channels = data_channels
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.filter = filter
        self.num_cheb_blocks = num_cheb_blocks
        self.num_stconv_blocks = num_stconv_blocks
        self.num_mlp_blocks = num_mlp_blocks
        self.mlp_ratio = mlp_ratio

        # Stack spatio-temporal conv blocks to make the STGCN.
        self._stgcn = [
            SpatioTempConvBlock(num_vars, data_channels, hidden_channels,
                                out_channels, kernel=kernel, filter=filter,
                                num_cheb_blocks=num_cheb_blocks, device=device)
        ]
        for i in range(num_stconv_blocks - 1):
            self._stgcn += [
                SpatioTempConvBlock(num_vars, out_channels, hidden_channels,
                                    out_channels, kernel=kernel, filter=filter,
                                    num_cheb_blocks=num_cheb_blocks,
                                    device=device)
            ]
        self._stgcn = nn.ModuleList(self._stgcn)

        # Stack linear layers and ReLU to make the MLP.
        self._mlp = [
            nn.Linear(latent_channels, int(latent_channels * mlp_ratio))
        ]
        for i in range(num_mlp_blocks - 1):
            self._mlp += [
                nn.ReLU(),
                nn.Linear(int(latent_channels * mlp_ratio),
                          int(latent_channels * mlp_ratio))
            ]
        self._mlp += [
            nn.ReLU(),
            nn.Linear(int(latent_channels * mlp_ratio),
                      patch_length * out_channels)
        ]
        self._mlp = nn.Sequential(*self._mlp)

        # Final regression layer.
        self._regression = nn.Linear(out_channels, data_channels)

        # Move everything onto device.
        self.to(self.device)

    def forward(self, last_input_patch: torch.FloatTensor,
                last_latent_patch: torch.FloatTensor,
                dependency_graph: torch.FloatTensor) -> torch.FloatTensor:
        """We do not batch here since the dependency graph computation (kNN) is not easily parallelizable in PyG.

        Args:
        last_input_data (torch.FloatTensor): Original data for *last* patch in *one* batch, shape (patch_length, num_vars, data_channels).
        last_latent_patch (torch.FloatTensor): Encoded data for *last* patch in *one* batch, shape (num_vars, latent_channels).
        dependency_graph (torch.FloatTensor): Adjacency matrix for learned graph structure, shape (num_vars, num_vars).

        Returns:
        out (torch.FloatTensor): Output data, shape (patch_length, num_vars, data_channels).
        These are the predicted data values for the next patch_length time steps.
        """
        # Differentiable operation (only for edge_weights) to allow backprop through adjacency matrix.
        edge_index, edge_weight = dense_to_sparse(dependency_graph)
        edge_weight = edge_weight.float()

        # Pass input data through STGCN blocks iteratively, with the output being (patch_length, num_vars, out_channels).
        stgcn_rep = last_input_patch.unsqueeze(0)
        for i in range(self.num_stconv_blocks):
            stgcn_rep = self._stgcn[i](stgcn_rep, edge_index,
                                       edge_weight=edge_weight)
        stgcn_rep = stgcn_rep[0]

        # Pass encoded data through MLP, and reshape the output to (patch_length, num_vars, out_channels).
        mlp_rep = self._mlp(last_latent_patch)
        mlp_rep = mlp_rep.reshape(
            (self.num_vars, self.patch_length, self.out_channels))
        mlp_rep = mlp_rep.transpose(0, 1)

        # Aggregate representations and regress to get the reconstructed next patch data.
        full_rep = stgcn_rep + mlp_rep
        out = self._regression(full_rep)
        return out

    def mae_loss(self, out_pred: torch.FloatTensor,
                 out_true: torch.FloatTensor) -> torch.FloatTensor:
        """This computes the reconstruction MAE loss, given the predicted patch and the output patch.

        Args:
        out_pred (torch.FloatTensor): Predicted patch of shape (patch_length, num_vars, data_channels).
        out_true (torch.FloatTensor): True patch of shape (patch_length, num_vars, data_channels).

        Returns:
        loss (torch.FloatTensor): MAE loss across all dimensions, which is of shape (1,).
        """
        assert out_pred.shape == out_true.shape
        loss = torch.mean(
            torch.abs(out_pred.reshape(-1) - out_true.reshape(-1)))
        return loss


class GraphStructureLearning(nn.Module):
    r"""The Graph Structure Learning procedure from
    `"Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"
    <https://arxiv.org/pdf/2206.09113>`. The Graph Structure Learning first generates a
    KNN naive adjacency matrix from the latent representations of nodes generated by the
    Encoder in the pre-training stage. Then through a series of fully connected, activation and
    convolutional layers that operate on both the batched latent representations and unbatched raw training data,
    this module attempts to learn Theta, a adjacency matrix of probabilites of positive edges. These probabilities
    are then fed as input to a Gumbel-Softmax 0/1 Sampler, resulting in the final adjacency matrix. The loss is computed with
    cross-entropy between Theta and the KNN adjacency matrix.

    .. math::
        A^a = \text{KNN algorithm}\left(\text{Latent Embeddings}\right)
        G = \text{FC}(\text{vec}(\text{Conv}(S_{\text{train}})))
        Z = \text{relu}(\text{FC}(H)) + G
        \Theta = \text{FC}(\text{relu}(\text{FC}(Z^i ∥ Z^j))),
        where ∥ represents concatenation
        L_\text{graph} =  \sum_{ij} -A_{ij} \log \Theta_{ij} - (1-A_{ij}) \log(1-\Theta_{ij}) (cross-entropy)


    Args:
        global_feature_size (int): The shape of G, global feature of time series.
        num_patches (int): The number of patches in input time series
        batch_size (int): The number of bathces in input data
        num_nodes (int): The number of nodes/variables in input time series
        embedding_dim (int): The dimension of the latent representations
            (default: :obj:`0.`)
        l_train (int): The length of the entire sequence over all training batches
        num_channels (int): The channel dimension of the input time series
        conv_channels (int): The number of channels in the convolutional layer
    """
    def __init__(self, global_feature_size, num_patches, batch_size, num_nodes,
                 embedding_dim, l_train, num_channels, conv_channels=5):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.global_feature_size = global_feature_size
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.l_train = l_train
        self.num_channels = num_channels

        # Layers for computing Z^i.
        self.fc_h = nn.Linear(num_patches * embedding_dim, global_feature_size)

        # Layers for computing G^i.
        self.conv = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.fc_g = nn.Linear(conv_channels * l_train * num_channels,
                              global_feature_size)

        # Layers for computing Θ_ij.
        self.fc_z = nn.Linear(global_feature_size * 2, global_feature_size)
        self.fc_theta = nn.Linear(global_feature_size, 2)
        self.to(self.device)

    def concatenate_nodes(self, latent_representations):
        # We concatenate the latent representation of each node over all patches.
        N, P, d = latent_representations.shape
        concatenated = latent_representations.reshape(N, P * d)
        return concatenated

    def calculate_adj_graph(self, concat_latent_representation, k):
        # We use KNN on the embeddings of the nodes to create a KNN graph in PyTorch geometric.
        N, Pd = concat_latent_representation.shape
        edge_index = knn_graph(concat_latent_representation, k=k, loop=False)
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=N)
        adj_matrix = adj_matrix.squeeze(0)
        return adj_matrix

    def forward(self, latent_representations, train_data):
        r"""Forward pass.

        Args:
            latent_representation (torch.Tensor): The input latent representaions of nodes
            train_data (torch.Tensor): The input training data
        Returns:
            Theta, the adjacency matrix of probabilites.
        """
        H = self.concatenate_nodes(latent_representations)
        s_train_reshaped = train_data.view(self.num_nodes, 1, -1)
        G = self.conv(s_train_reshaped.float())
        G = G.view(self.num_nodes, -1)
        G = self.fc_g(G)

        # Z should be of shape (N, global_feature_size).
        Z = nn.functional.relu(self.fc_h(H)) + G
        N, gfs = Z.shape

        # Create Z_i and Z_j and concatenate them.
        Z_i = Z.unsqueeze(1).expand(N, N, gfs)
        Z_j = Z.unsqueeze(0).expand(N, N, gfs)
        Z_ij = torch.cat([Z_i, Z_j], dim=-1)

        # Calculate Theta, which has shape (N, N, 2).
        Theta = self.fc_z(Z_ij)
        Theta = nn.functional.relu(Theta)
        Theta = self.fc_theta(Theta)
        return Theta

    def graph_regularization_loss(self, Theta, A_knn):
        # This computes cross entropy loss between our KNN adj matrix and the computed Theta.
        Theta_normalized = nn.functional.softmax(Theta, dim=-1)
        Theta_prime = Theta_normalized[:, :, 0]
        loss_matrix = -A_knn * torch.log(Theta_prime + 1e-10) - (
            1 - A_knn) * torch.log(1 - Theta_prime + 1e-10)
        return loss_matrix.sum()

    def gumbel_softmax(self, learned_theta, tau, hard=True):
        # Generate the final A from the learned Theta, using Gumbel Softmax.
        return nn.functional.gumbel_softmax(learned_theta, tau=tau, hard=hard)


class DownstreamModel(nn.Module):
    r"""The Downstream STGNN module from `"Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"
    <https://arxiv.org/pdf/2206.09113>`. After learning an adjacency matrix to capture graph structure from
    the Graph Structure Learning module, the downstream prediction task focuses on combining the representations from
    the pre-trained Encoder, and our backend model of choice: Spatial Temporal Graph Convolutional Network (STGCN). After these
    representations are combined, with the use of a semantic projector, we use a regression layer to forecast the time series. The final loss
    is computed as a linear combination of Mean Absolute Error (MAE) and the Graph Structure Loss


    .. math::
        (Final Adjacency Matrix): A_{ij} = \text{softmax}((\Theta{ij} + \text{iid Gumbel(0,1)})/2)
        H_{\text{final}} = \text{SP}(H_P) + H_{\text{stgcn}}, where H_P represents the latent embeddings, and H_{\text{stgcn}} represents
        the STGCN embeddings
        L_{\text{regression}}= \frac{1}{T_f NC} \sum_{j=1}^{T_f} \sum_{i=1}^{N} \sum_{k=1}^{C} |\hat{y}_{ijk} - y_{ijk}|
        L_{\text{final}} = L_{\text{regression}} + \lambda * L_{\text{graph}}


    Args:
        time_window (int): The temporal window size :math:`T` to define the
            1-hop temporal neighborhood.
    """
    def __init__(self, num_vars, in_channels, batch_size, num_patches, patch_length, \
                 tsformer_dim=24, hidden_stgcn_dim=32, out_stgcn_dim=16, stgcn_kernel=3, stgcn_filter=3, gsl_global_dim=8, gsl_k=10, \
                 lr=0.001, gsl_loss_decay=0.99):

        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_vars = num_vars
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.patch_length = patch_length
        self.train_length = num_patches * batch_size * patch_length
        self.tsformer_dim = tsformer_dim

        self.input_data_sz = (num_vars, self.train_length, in_channels)
        self.encode_data_sz = (batch_size, num_vars, num_patches, tsformer_dim)

        self._enhanced_stgcn = EnhancedSTGCN(num_vars, patch_length,
                                             in_channels, tsformer_dim,
                                             hidden_stgcn_dim, out_stgcn_dim,
                                             stgcn_kernel, stgcn_filter)
        self._graph_structure_learner = GraphStructureLearning(
            gsl_global_dim, num_patches, batch_size, num_vars, tsformer_dim,
            self.train_length, in_channels)
        self.gsl_k = gsl_k  # kNN hyperparameter

        self._full_model = nn.ModuleList(
            [self._enhanced_stgcn, self._graph_structure_learner])
        self.lr = lr
        self.gsl_loss_decay = gsl_loss_decay
        self._optim = torch.optim.Adam(self._full_model.parameters(), lr=lr)
        self.to(self.device)

    # Note: train_batches is different than batch_size. The former is batching during training, the previous is batching that occurred in pretraining.
    def train(self, input_data, encode_data, num_epochs=100, train_batches=32,
              print_every=10):
        assert input_data.shape == self.input_data_sz
        assert encode_data.shape == self.encode_data_sz
        encode_data = encode_data.transpose(0, 1).reshape(
            self.num_vars, self.batch_size * self.num_patches,
            self.tsformer_dim)

        # Set to training mode. Gamma is the contribution of GSL to the loss.
        gamma = 1.0
        self._enhanced_stgcn.train()
        self._graph_structure_learner.train()

        # Batched gradient descent.
        loss_values = []
        for i in range(num_epochs):
            self._optim.zero_grad()  # Zero out gradients.
            loss = 0.0
            gamma *= self.gsl_loss_decay  # Decay contribution of GSL loss over time.

            # Iterate through batches, which will be randomly chosen contiguous P patches in the training set.
            for j in range(train_batches):
                start_idx = random.randint(
                    0, (self.batch_size - 1) * self.num_patches - 1
                )  # Start index of a patch. We will need P patches after this.

                # We need hiddens for the P patches after the start index (inclusive) as input to models.
                cur_window_all_hidden = encode_data[:,
                                                    start_idx:start_idx + self.
                                                    num_patches]  # (N, P, D)
                cur_window_final_hidden = cur_window_all_hidden[:,
                                                                -1]  # (N, D)

                # Our goal in training is to use the last of these P patches to predict the immediate next one.
                # Note information from all P patches is included implicitly (as is data from all training patches).
                # This is because Graph Structure Learning uses that information to pred an adj matrix, used in the STGCN.
                cur_window_final_patch  = input_data[:, (start_idx + self.num_patches - 1) * self.patch_length: \
                                          (start_idx + self.num_patches) * self.patch_length].transpose(0, 1)       # (L, N, C)
                next_window_start_patch = input_data[:, (start_idx + self.num_patches) * self.patch_length: \
                                          (start_idx + self.num_patches + 1) * self.patch_length].transpose(0, 1)   # (L, N, C)

                # Forward pass through Graph Structure Learning to get theta, KNN adj matrix, loss, and pred adj matrix.
                theta = self._graph_structure_learner(cur_window_all_hidden,
                                                      input_data)
                concat_window_hidden = self._graph_structure_learner.concatenate_nodes(
                    cur_window_all_hidden)
                adj_knn = self._graph_structure_learner.calculate_adj_graph(
                    concat_window_hidden, self.gsl_k)
                adj_pred = self._graph_structure_learner.gumbel_softmax(
                    theta, tau=1.0
                )[...,
                  0]  # Shape [N, N, 2] has pos & neg adj matrices, we take pos one.
                loss_graph = self._graph_structure_learner.graph_regularization_loss(
                    theta, adj_knn)

                # Forward pass through Enhanced STGCN, using computed pred adj matrix above, and get associated loss too.
                next_window_start_patch_pred = self._enhanced_stgcn.forward(
                    cur_window_final_patch, cur_window_final_hidden, adj_pred)
                loss_stgcn = self._enhanced_stgcn.mae_loss(
                    next_window_start_patch_pred, next_window_start_patch)

                # Add losses with a decay term.
                loss += loss_stgcn + gamma * loss_graph

            # Backpropagate after running through batches.
            loss.backward()
            self._optim.step()
            if i > 0:
                if i % print_every == 0:
                    print(f'Epoch {i}, Loss={loss.item()}')
                loss_values.append(loss.item())

        self.adj_pred = adj_pred  #save the static adjacency matrix to use in forecasting
        return loss_values

    def forecast(self, input_patch, encode_patch):
        """Forecasting next patch

        Args:
        input_patch (torch.FloatTensor): Input patch of shape (patch_length, num_vars, data_channels).
        encoded_patch (torch.FloatTensor): Encoded data for input patch, shape (num_vars, latent_channels).

        Returns:
        next_patch (torch.FloatTensor): Output data, shape (patch_length, num_vars, data_channels).
         [These are the predicted data values for the next patch_length time steps].
        """
        print(encode_patch.shape)

        self._enhanced_stgcn.eval()
        self._graph_structure_learner.eval()

        with torch.no_grad():
            next_patch = self._enhanced_stgcn.forward(input_patch,
                                                      encode_patch,
                                                      self.adj_pred)

        return next_patch
