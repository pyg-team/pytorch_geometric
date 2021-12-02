import os.path as osp
import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.datasets import MovieLens
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from torch_geometric.nn import SAGEConv, to_hetero

parser = argparse.ArgumentParser()
parser.add_argument(
    '--feature_extraction_model', type=str, default='all-MiniLM-L6-v2',
    help='Model used to transform movie titles to node features.'
    'This should be one of the models available in Huggingface'
    'SentenceTransformer (https://huggingface.co/sentence-transformers)')
parser.add_argument(
    '--disjoint_train_ratio', type=float, default=0.0,
    help=('Proportion of training edges used for supervision. '
          'The remainder is disjoint and used for message passing.'
          'This option can be used if target leakage is a concern.'))
parser.add_argument('--num_val', type=float, default=0.1)
parser.add_argument('--num_test', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=201)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.000)
parser.add_argument('--hidden', type=int, default=32)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
dataset = MovieLens(path, model_name=args.feature_extraction_model)
data = dataset[0]

# Add user node features for message passing
data['user'].x = torch.eye(data['user'].num_nodes, dtype=data['movie'].x.dtype)

# We can now convert `data` into an appropriate format for training a
# graph-based machine learning model:

# 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
data = ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# 2. Perform a link-level split into training, validation, and test edges.
transform = RandomLinkSplit(
    num_val=args.num_val,
    num_test=args.num_test,
    disjoint_train_ratio=args.disjoint_train_ratio,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)
train_data, val_data, test_data = transform(data)

# Transform labels to float32 so the mse_loss works
data_dict = {'train': (train_data), 'val': (val_data), 'test': (test_data)}
for d in data_dict.values():
    d[('user', 'rates',
       'movie')].edge_label = d[('user', 'rates',
                                 'movie')].edge_label.type(torch.float32)


# Construct GNN for message passing
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, edge_labels):
        super().__init__()

        self.edge_labels = edge_labels
        num_relations = torch.numel(self.edge_labels)

        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def decode(self, z, edge_index, edge_type):
        # We only need to decode on the ('user','rates','movie') edges
        z_src = z['user']
        z_dst = z['movie']
        z_src = z_src[edge_index[0]]
        z_dst = z_dst[edge_index[1]]
        rel = self.rel_emb[int(edge_type)]
        return torch.sum(z_src * rel * z_dst, dim=1)

    def forward(self, z, edge_label_index):
        # Initialise scores array, which records the DistMult score for
        # each supervision edge and edge_label (rating)
        scores = torch.zeros(edge_label_index.shape[1],
                             self.edge_labels.shape[0])
        for label in self.edge_labels:
            score = self.decode(z, edge_label_index, label)
            scores[:, int(label)] = score

        # Return "probabilities" for each class
        probs = F.softmax(scores, dim=1)
        possible_ratings = torch.Tensor(self.edge_labels)

        # Get ratings (expectation value)
        prod = probs * possible_ratings
        predicted_ratings = torch.sum(prod, dim=1)

        return (predicted_ratings)


class EdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, edge_labels):
        super().__init__()
        # Set up GNN for message passing
        self.encoder = GNN(hidden_channels=hidden_channels,
                           out_channels=hidden_channels)
        # Make the GNN model heterogeneous
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')

        # DistMult based decoder for edge classification
        self.decoder = EdgeDecoder(hidden_channels, edge_labels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Perform message passing
        z = self.encoder(x_dict, edge_index_dict)

        # The edges that go into the decoder are the supervision edges
        out = self.decoder(z, edge_label_index)
        return (out)


# We have an unbalanced dataset, with many labels for higher ratings
# and few for low ratings. Therefore we use a weighted MSE loss.
edge_labels = torch.unique(train_data['user', 'rates', 'movie'].edge_label)
num_samples_per_label = torch.bincount(
    torch.squeeze(train_data['user', 'rates', 'movie']
                  .edge_label.type(torch.int64)))

weights = 1 / num_samples_per_label
# Normalise the weights
weights = weights / torch.sum(weights)


def weighted_mse_loss(input, target, weight):
    expanded_weights = torch.index_select(weight, 0, target.type(torch.int64))
    return torch.mean(expanded_weights * (input - target) ** 2)


model = EdgeClassifier(hidden_channels=args.hidden, edge_labels=edge_labels)


# Get number of model parameters
# Due to lazy initialisation we need to run one model step so the number
# of parameters can be inferred
with torch.no_grad():
    model(train_data.x_dict, train_data.edge_index_dict,
          train_data[('user', 'rates', 'movie')].edge_label_index)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
train_data, val_data, test_data = train_data.to(device), val_data.to(
    device), test_data.to(device)


def train():
    model.train()
    optimizer.zero_grad()

    # Get predicted ratings
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data[('user', 'rates', 'movie')].edge_label_index)

    # Get true ratings
    target = train_data[('user', 'rates', 'movie')].edge_label

    # Apply weighted mse loss
    loss = weighted_mse_loss(pred, torch.squeeze(target), weights)
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    rmse_dict = {}
    data_dict = {'train': (train_data), 'val': (val_data), 'test': (test_data)}
    for key, d in data_dict.items():
        pred = model(d.x_dict, d.edge_index_dict,
                     d[('user', 'rates', 'movie')].edge_label_index)

        target = d[('user', 'rates', 'movie')].edge_label

        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(pred, torch.squeeze(target),
                              reduction='mean'))

        rmse_dict[key] = rmse

    return rmse_dict


best_val_rmse = test_rmse = 100
for epoch in range(1, args.num_epochs):
    train()

    if (epoch % 5 == 0) or (epoch == 1):
        rmse = test()
        if rmse['val'] < best_val_rmse:
            best_val_rmse = rmse['val']
            test_rmse = rmse['test']
        # Printing current metrics
        log = (f"Epoch: {epoch}, "
               f"rmse train: {rmse['train']:.4f}, "
               f"rmse val: {rmse['val']:.4f}, "
               f"rmse test: {rmse['test']:.4f}.")

        print(log)
print((f"Best validation RMSE: {best_val_rmse:.4f}, "
       f"associated test RMSE: {test_rmse:.4f}"))
