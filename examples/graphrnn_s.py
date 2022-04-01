import argparse

import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnnutils

import torch_geometric
from torch_geometric import data, transforms as T, loader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot", action="store_true", help="Whether to plot graphs."
)
args = parser.parse_args()


class EncodeGraphRNNFeature(T.BaseTransform):
    def __init__(self, M):
        self.M = M

    @staticmethod
    def extract_bands(adj, M):
        """
        Uses stride tricks to extract the M bands above the diagonal of the
        given square matrix.

        :param adj: dimension N x N
        :param M: number of bands above the diagonal to return
        :returns: dimension (N - 1) x M; the M bands above the diagonal
        """
        N = adj.shape[1]
        adj = adj.reshape(N, N)
        padded_adj = torch.zeros((N + M - 1, N))
        padded_adj[M - 1:, :] = adj
        return padded_adj.as_strided(
            size=(N - 1, M), stride=(N + 1, N), storage_offset=1
        )

    @staticmethod
    def bands_to_matrix(bands):
        """
        Given M bands above the diagonal of a square matrix, return the full
        matrix.

        :param bands: dimension N x M; the M bands above the diagonal
        :returns: the corresponding matrix of dimension N x N
        """
        M = bands.shape[1]
        N = bands.shape[0] + 1
        padded_adj = torch.zeros((N + M - 1, N))
        view = padded_adj.as_strided(
            size=(N - 1, M), stride=(N + 1, N), storage_offset=1
        )
        view[:, :] = bands
        return padded_adj[M - 1:, :]

    @staticmethod
    def inverse(y):
        """
        Inverse of the __call__ method, given the encoded sequence y

        :param y: encoded sequence, without the SOS and EOS tokens
        :returns: the corresponding adjacency matrix
        """
        bands = torch.flip(y, dims=[1])
        adj = EncodeGraphRNNFeature.bands_to_matrix(bands)
        return adj

    def __call__(self, data):
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        sequences = torch.flip(self.extract_bands(adj, self.M), dims=[1])

        # Add SOS (row of ones) and EOS (row of zeros).
        sequences = torch.cat(
            [torch.ones(1, self.M), sequences, torch.zeros(1, self.M)], dim=0
        )

        data.length = data.num_nodes

        data.x = sequences[:-1]
        data.y = sequences[1:]
        return data


class BFS(T.BaseTransform):
    """
    Start a breath first search from a random node and reorder the edge list so
    that the node indices correspond to the breadth-first search order.
    """

    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        assert (
            data.is_undirected()
        ), "Transform only works for undirected graphs."
        G = torch_geometric.utils.to_networkx(
            data, to_undirected=data.is_undirected()
        )

        start_node = torch.randint(0, data.num_nodes, (1,)).item()

        # Get the breadth-first search order.
        bfs_order = [start_node] + [n for _, n in nx.bfs_edges(G, start_node)]
        perm = torch.tensor(bfs_order).argsort()
        return torch_geometric.data.Data(
            x=x, edge_index=perm[edge_index], num_nodes=data.num_nodes
        )


class GraphRNNTransform(T.Compose):
    def __init__(self, M):
        super(GraphRNNTransform, self).__init__(
            [BFS(), EncodeGraphRNNFeature(M=M)]
        )


class CyclesDataset(data.InMemoryDataset):
    """
    Creates a dataset of cycle graphs.
    """

    def __init__(self, min_n, max_n, transform):
        super().__init__(".", transform)
        graphs = [
            torch_geometric.utils.from_networkx(nx.cycle_graph(i))
            for i in range(min_n, max_n)
        ]
        self.data, self.slices = self.collate(graphs)


# The maximum size of a BFS queue on our dataset.
# Can be estimated emperically by running many BFS.
# Denoted M as in the paper.
M = 15
# The maximum number of nodes that the sampler can generate per graph.
SAMPLER_MAX_NUM_NODES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CyclesDataset(transform=GraphRNNTransform(M=M), min_n=3, max_n=50)
dataloader = loader.DataLoader(dataset, batch_size=32, shuffle=True)


def plot_4_graphs(graphs, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    for i, graph in enumerate(graphs):
        plt.subplot(2, 2, i + 1)
        nx.draw_spectral(graph, node_size=100)
    plt.show()


if args.plot:
    plot_4_graphs(
        [
            torch_geometric.utils.to_networkx(graph, to_undirected=True)
            for graph in dataset
        ][:4],
        "Train graphs",
    )


class GraphRNN_S(nn.Module):
    def __init__(
        self,
        *,
        adjacency_size: int,
        embed_first: bool,
        adjacency_embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_embedding_size: int,
    ):
        """
        @param adjacency_size: Size of an adjacency vector. M in the paper.
        @param embed_first: Whether to transform the adjacency vectors before
        feeding them to the RNN cell.
        @param adjacency_embedding_size: If embed_first, then Size of the
        embedding of the adjacency vectors before feeding it to the RNN cell.
        @param hidden_size: Size of the hidden vectors of the RNN cell.
        @param num_layers: Number of stacked RNN layers
        @param output_embedding_size: Size of the embedding of the edge_level
        MLP.
        """
        super().__init__()
        self.adjacency_size = adjacency_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None

        if embed_first:
            self.embedding = nn.Sequential(
                nn.Linear(adjacency_size, adjacency_embedding_size),
                nn.ReLU(),
            )
            input_to_rnn_size = adjacency_embedding_size
        else:
            self.embedding = nn.Identity()
            input_to_rnn_size = adjacency_size

        self.rnn = nn.RNN(
            input_size=input_to_rnn_size,
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
        sequences = rnnutils.pack_padded_sequence(
            sequences, lengths, batch_first=True, enforce_sorted=False
        )
        sequences = rnnutils.pad_packed_sequence(sequences, batch_first=True)[
            0
        ]
        return sequences

    def forward(self, input_sequences, input_length, sampling=False):
        """
        @param input_sequences: (batch_size, max_num_nodes, adjacency_size=M)
        For each graph in the batch, the sequence of adjacency vectors
        (including the first SOS).
        @param input_length: (batch_size,)
            num_nodes for each graph in the batch. Because graph-sequences
            where padded to max_num_nodes.
        """
        input_sequences = self.embedding(input_sequences)

        # Pack sequences for RNN efficiency.
        input_sequences = rnnutils.pack_padded_sequence(
            input_sequences,
            input_length,
            batch_first=True,
            enforce_sorted=False,
        )
        if sampling:
            output_sequences, self.hidden = self.rnn(
                input_sequences, self.hidden
            )
        else:
            output_sequences, self.hidden = self.rnn(input_sequences)
        # Unpack RNN output.
        output_sequences, output_length = rnnutils.pad_packed_sequence(
            output_sequences, batch_first=True
        )

        # MLP to get adjacency vectors.
        output_sequences = self.adjacency_mlp(output_sequences)

        return self.mask_out_bits_after_length(output_sequences, input_length)

    def sample(self, batch_size, device, max_num_nodes):
        """
        Sample a batch of graph sequences.
        @return: Tensor of size (batch_size, max_num_node, self.adjacency_size)
        in the same device as the model.

        Note: In the original implementation a max_num_node is used as a
        placeholder for the generated graphs.  This makes the assumption that
        the largest generated graph will have max_num_nodes.

        Instead, this implementation makes the assumption that generated graphs
        are connected.  This assumption is implicit in the original codebase.

        In any case one of the above assumptions has to be made to know when
        the sampler is done generating a graph.  The disconnected graph
        assumption can be dropped by adding an SOS flag to the model rather
        than an SOS token which can be confused with a disconnected node.
        """
        input_sequence = torch.ones(
            batch_size, 1, self.adjacency_size, device=device
        )  # SOS.
        is_not_eos = torch.ones(batch_size, dtype=torch.long)

        sequences = torch.zeros(batch_size, max_num_nodes, self.adjacency_size)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        # Id of the node to be added to the sequence. Node 0 is not added.
        node_id = 0
        with torch.no_grad():
            self.hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
            while is_not_eos.any():
                node_id += 1
                if node_id == max_num_nodes:
                    break

                output_sequence_probs = self.forward(
                    input_sequence, torch.ones(batch_size), sampling=True
                )
                mask = torch.rand_like(output_sequence_probs)
                output_sequence = torch.gt(output_sequence_probs, mask)

                # Identify the EOS sequences and persist them even if model
                # says otherwise.
                is_not_eos *= output_sequence.any(dim=-1).squeeze().cpu()
                seq_lengths += is_not_eos

                sequences[:, node_id - 1] = output_sequence[:, 0]
                input_sequence = output_sequence.float()

        # Clean irrelevant bits and enforce creation of connected graph.
        # Pack to seq_lengths to include empty sequences. Pack does not support
        # empty sequences.
        self.mask_out_bits_after_length(sequences, seq_lengths + 1)
        sequences = sequences.tril()

        return sequences[:, : seq_lengths.max()], seq_lengths


model = GraphRNN_S(
    adjacency_size=M,
    embed_first=True,
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
        x_padded = rnnutils.pad_sequence(
            torch.split(batch.x, lengths_tuple), batch_first=True
        )
        y_padded = rnnutils.pad_sequence(
            torch.split(batch.y, lengths_tuple), batch_first=True
        )

        output_sequences = model(x_padded, lengths)

        loss = F.binary_cross_entropy(output_sequences, y_padded)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            # Compute the epoch NLL. Can be refactored.
            if batch_idx == 0:
                epoch_nll = 0
            with torch.no_grad():
                # Only leave relevant bits.
                # (In rows i < M, remove bits before i).
                output_sequences *= output_sequences.tril()
                epoch_nll += (
                    F.binary_cross_entropy(
                        output_sequences, y_padded, reduction="sum"
                    ).item()
                    / batch.num_graphs
                )

            if batch_idx == 0:
                # sample some graphs and evaluate them
                output_sequences, lengths = model.sample(
                    64, device, SAMPLER_MAX_NUM_NODES
                )
                adjs = [
                    EncodeGraphRNNFeature.inverse(sequence[:length])
                    for sequence, length in zip(output_sequences, lengths)
                ]
                graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]

                if args.plot:
                    plot_4_graphs(
                        graphs[:4], "Sampled graphs at epoch {}".format(epoch)
                    )

                # check if the generated graphs are cycles
                def is_cycle(G):
                    return len(list(nx.cycle_basis(G))) == 1
                percentage_are_cycles = sum(map(is_cycle, graphs)) / len(
                    graphs
                )
                print(
                    "Percentage of generated graphs that are cycles at epoch "
                    f"{epoch}: {percentage_are_cycles}"
                )
