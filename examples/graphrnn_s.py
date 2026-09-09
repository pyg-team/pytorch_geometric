import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnnutils

import torch_geometric
from torch_geometric import data, loader
from torch_geometric import transforms as T


class EncodeGraphRNNFeature(T.BaseTransform):
    r"""Encode the graph adjacency matrix into a sequence of adjacency vectors
    of length :obj:`M`.
    This is the approach described in the "GraphRNN: Generating Realistic
    Graphs with Deep Auto-regressive Models"
    <https://arxiv.org/pdf/1802.08773>`_ paper.

    Args:
        M (int): The length of the adjacency vectors.
    """
    def __init__(self, M):
        self.M = M

    @staticmethod
    def extract_bands(adj, M):
        r"""Uses stride tricks to extract :obj:`M`-long bands above the
        diagonal of the given square matrix.

        Args:
            adj (Tensor): The adjacency matrix of shape :obj:`(N, N)`.
            M (int): The number of bands to extract.

        :rtype: Tensor
        """
        N = adj.shape[1]
        padded_adj = torch.zeros((N + M - 1, N))
        padded_adj[M - 1:, :] = adj
        return padded_adj.as_strided(size=(N - 1, M), stride=(N + 1, N),
                                     storage_offset=1)

    @staticmethod
    def bands_to_matrix(bands):
        r"""Given M bands above the diagonal of a square matrix, return the
        full matrix.

        Args:
            bands (Tensor): The M bands of shape :obj:`(N, M)`.

        :rtype: Tensor
        """
        M = bands.shape[1]
        N = bands.shape[0] + 1
        padded_adj = torch.zeros((N + M - 1, N))
        view = padded_adj.as_strided(size=(N - 1, M), stride=(N + 1, N),
                                     storage_offset=1)
        view[:, :] = bands
        return padded_adj[M - 1:, :]

    @staticmethod
    def inverse(y):
        r"""Inverse of the :obj:`__call__` method: given the encoded sequence
        y, return the adjacency matrix.

        Args:
            y (Tensor): The encoded sequence of shape :obj:`(N, M)`.

        :rtype: Tensor
        """
        bands = torch.flip(y, dims=[1])
        adj = EncodeGraphRNNFeature.bands_to_matrix(bands)
        return adj

    def __call__(self, data):
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        sequences = torch.flip(self.extract_bands(adj, self.M), dims=[1])

        # Add SOS (row of ones) and EOS (row of zeros).
        sequences = torch.cat(
            [torch.ones(1, self.M), sequences,
             torch.zeros(1, self.M)], dim=0)

        data.length = data.num_nodes

        data.x = sequences[:-1]
        data.y = sequences[1:]
        return data


class BFS(T.BaseTransform):
    r"""Start a breath first search from a random node and reorder the edge
    list so that the node indices correspond to the breadth-first search order.
    """
    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        assert (data.is_undirected()
                ), "Transform only works for undirected graphs."
        G = torch_geometric.utils.to_networkx(
            data, to_undirected=data.is_undirected())

        start_node = torch.randint(0, data.num_nodes, (1, )).item()

        # Get the breadth-first search order.
        bfs_order = [start_node] + [n for _, n in nx.bfs_edges(G, start_node)]
        perm = torch.tensor(bfs_order).argsort()
        return torch_geometric.data.Data(x=x, edge_index=perm[edge_index],
                                         num_nodes=data.num_nodes)


class GraphRNNTransform(T.Compose):
    def __init__(self, M):
        super(GraphRNNTransform,
              self).__init__([BFS(), EncodeGraphRNNFeature(M=M)])


class CyclesDataset(data.InMemoryDataset):
    r"""Creates a dataset of :obj:`max_n - min_n` cycle graphs.

    Args:
        min_n (int): The minimum number of nodes in the graphs.
        max_n (int): The maximum number of nodes in the graphs.
    """
    def __init__(self, min_n, max_n, transform):
        super().__init__(".", transform)
        graphs = [
            torch_geometric.utils.from_networkx(nx.cycle_graph(i))
            for i in range(min_n, max_n)
        ]
        self.data, self.slices = self.collate(graphs)


def is_cycle(graph):
    return len(nx.cycle_basis(graph)) == 1


def plot_4_graphs(graphs, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    for i, graph in enumerate(graphs):
        plt.subplot(2, 2, i + 1)
        nx.draw_spectral(graph, node_size=100)
    plt.show()


class GraphRNN_S(nn.Module):
    r"""The GraphRNN-S model from the "GraphRNN: Generating Realistic Graphs
    with Deep Auto-regressive Models" <https://arxiv.org/pdf/1802.08773>`_
    paper, consisting of a node-level RNN and a edge-level output layer.

    Args:
        adjacency_size (int): The size of the adjacency vectors.
        adjacency_embedding_size (int): The size of the embedding of the
            of the adjacency vectors before feeding it to the RNN cell.
        hidden_size (int): The size of the hidden vectors of the RNN cell.
        num_layers (int): The number of stacked RNN layers.
        output_embedding_size (int): The size of the embedding of the
            edge level MLP.
    """
    def __init__(
        self,
        *,
        adjacency_size: int,
        adjacency_embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_embedding_size: int,
    ):
        super().__init__()
        self.adjacency_size = adjacency_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None

        self.embedding = nn.Sequential(
            nn.Linear(adjacency_size, adjacency_embedding_size),
            nn.ReLU(),
        )
        self.rnn = nn.RNN(
            input_size=adjacency_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.adjacency_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_embedding_size),
            nn.ReLU(),
            nn.Linear(output_embedding_size, adjacency_size),
            nn.Sigmoid(),
        )

    def mask_out_bits_after_length(self, sequences, lengths):
        sequences = rnnutils.pack_padded_sequence(sequences, lengths,
                                                  batch_first=True,
                                                  enforce_sorted=False)
        sequences = rnnutils.pad_packed_sequence(sequences,
                                                 batch_first=True)[0]
        return sequences

    def forward(self, input_sequences, input_length, hidden=None):
        r"""Forward pass of the GraphRNN-S model.

        Args:
            input_sequences (Tensor): For each graph in the batch, the sequence
                of adjacency vectors (including the first SOS token). Shape:
                :obj:`(batch_size, max_num_nodes, adjacency_size)`
            input_length (Tensor): For each graph in the batch, the length of
                the sequence of adjacency vectors. Shape: :obj:`(batch_size,)`
            hidden (Tensor, optional): Initial hidden state.
        """
        input_sequences = self.embedding(input_sequences)

        # Pack sequences for RNN efficiency.
        input_sequences = rnnutils.pack_padded_sequence(
            input_sequences,
            input_length,
            batch_first=True,
            enforce_sorted=False,
        )
        if hidden is not None:
            output_sequences, self.hidden = self.rnn(input_sequences, hidden)
        else:
            output_sequences, self.hidden = self.rnn(input_sequences)
        # Unpack RNN output.
        output_sequences, output_length = rnnutils.pad_packed_sequence(
            output_sequences, batch_first=True)

        # MLP to get adjacency vectors.
        output_sequences = self.adjacency_mlp(output_sequences)

        return self.mask_out_bits_after_length(output_sequences, input_length)

    def sample(self, batch_size, device, max_num_nodes):
        r"""Sample a batch of graph sequences. Assumes that learned/generated
        graphs are connected, as in the paper.

        Args:
            batch_size (int): The number of graphs to sample.
            device (torch.device): The device to run the sampling on.
            max_num_nodes (int): The maximum number of nodes that a sampled
                graph may contain.

        :rtype Tensor:
        """
        input_sequence = torch.ones(batch_size, 1, self.adjacency_size,
                                    device=device)  # SOS.
        is_not_eos = torch.ones(batch_size, dtype=torch.long)

        sequences = torch.zeros(batch_size, max_num_nodes, self.adjacency_size)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        # Id of the node whose adjacency vector will be sampled.
        # Node 0 is not included.
        node_id = 0
        with torch.no_grad():
            self.hidden = torch.zeros(self.num_layers, batch_size,
                                      self.hidden_size, device=device)
            while is_not_eos.any():
                node_id += 1
                if node_id == max_num_nodes:
                    break

                # generate the sampling probabilities for the adjacency vectors
                output_sequence_probs = self.forward(input_sequence,
                                                     torch.ones(batch_size),
                                                     hidden=self.hidden)
                # sample the adjacency vectors
                mask = torch.rand_like(output_sequence_probs)
                output_sequence = torch.gt(output_sequence_probs, mask)

                # Identify the EOS sequences
                # (these are the adjacency where no edge was sampled).
                is_not_eos *= output_sequence.any(dim=-1).squeeze().cpu()
                seq_lengths += is_not_eos

                sequences[:, node_id - 1] = output_sequence.squeeze(1)
                input_sequence = output_sequence.float()

        # Clean irrelevant bits and enforce creation of connected graph.
        # Pack to 1 + seq_lengths to include empty sequences
        self.mask_out_bits_after_length(sequences, seq_lengths + 1)
        sequences = sequences.tril()

        return sequences[:, :seq_lengths.max()], seq_lengths


parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true",
                    help="Whether to plot graphs.")
args = parser.parse_args()

# The maximum size of a BFS queue on our dataset.
# Can be estimated empirically by running many BFS.
# Denoted M as in the paper.
M = 25
# The maximum number of nodes that the sampler can generate per graph.
SAMPLER_MAX_NUM_NODES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CyclesDataset(transform=GraphRNNTransform(M=M), min_n=3, max_n=50)
dataloader = loader.DataLoader(dataset, batch_size=32, shuffle=True)

if args.plot:
    plot_4_graphs(
        [
            torch_geometric.utils.to_networkx(graph, to_undirected=True)
            for graph in dataset[random.sample(range(len(dataset)), 4)]
        ],
        "Train graphs",
    )

model = GraphRNN_S(
    adjacency_size=M,
    adjacency_embedding_size=64,
    hidden_size=128,
    num_layers=4,
    output_embedding_size=64,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
model = model.to(device)
for epoch in range(1001):
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # (1) Transform the batched graphs to a standard mini-batch of
        # dimensions (B, L, M), where L is max_num_nodes in the batch.
        lengths = batch.length.cpu()  # torch.split needs a tuple of ints.
        lengths_tuple = tuple(lengths.tolist())
        x_padded = rnnutils.pad_sequence(torch.split(batch.x, lengths_tuple),
                                         batch_first=True)
        y_padded = rnnutils.pad_sequence(torch.split(batch.y, lengths_tuple),
                                         batch_first=True)

        output_sequences = model(x_padded, lengths)

        loss = F.binary_cross_entropy(output_sequences, y_padded)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 and batch_idx == 0:
            # Sample some graphs and evaluate them.
            output_sequences, lengths = model.sample(64, device,
                                                     SAMPLER_MAX_NUM_NODES)
            adjs = [
                EncodeGraphRNNFeature.inverse(sequence[:length])
                for sequence, length in zip(output_sequences, lengths)
            ]
            graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]

            if args.plot:
                plot_4_graphs(graphs[:4],
                              "Sampled graphs at epoch {}".format(epoch))

            # Check if the generated graphs are cycles.
            percentage_are_cycles = sum(map(is_cycle, graphs)) / len(graphs)
            print("Percentage of generated graphs that are cycles at epoch "
                  f"{epoch}: {percentage_are_cycles}")
