import os.path as osp

import torch
from torch.nn import Parameter, ParameterDict
import torch.nn.functional as F

from torch_geometric.datasets import MovieLense

from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from torch_geometric.nn import SAGEConv, to_hetero

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
dataset = MovieLense(path)
data = dataset[0]

# Add dummy user features for message passing
data['user'].x = torch.eye(data['user'].num_nodes, dtype=data['movie'].x.dtype)
print(data)

# We can now convert `data` into an appropriate format for training a
# graph-based machine learning model:

# 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
data = ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# 2. Perform a link-level split into training, validation, and test edges.
transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)
train_data, val_data, test_data = transform(data)
print(train_data)
print(val_data)
print(test_data)

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
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)
    
    def forward(self, z, edge_label_index):
        # Initialise scores array, which records the DistMult score for
        # each supervision edge and edge_label (rating)
        scores = torch.zeros(edge_label_index.shape[1], self.edge_labels.shape[0])
        for label in self.edge_labels:
            score = self.decode(z,edge_label_index, label)
            scores[:,label] = score
        # Return log probabilities for each class
        return(F.log_softmax(scores, dim=0))

class BilinearDecoder(torch.nn.Module):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, edge_labels):
        super().__init__()

        self.edge_labels = edge_labels
        self.local_interaction = ParameterDict()
        for edge_label in edge_labels:
            edge_label_key = str(edge_label.item())
            self.local_interaction[edge_label_key] = Parameter(torch.Tensor(input_dim, input_dim))
        self.reset_parameters()


    def reset_parameters(self):
        # Reset local interaction vectors
        for key in self.local_interaction.keys():
            torch.nn.init.xavier_uniform_(self.local_interaction[key])

    def decode(self, z, edge_index, edge_type):
        z_src = z['user']
        z_dst = z['movie']
        z_src = z_src[edge_index[0]]
        z_dst = z_dst[edge_index[1]]

        intermediate = torch.matmul(z_src, self.local_interaction[str(edge_type.item())])
        outputs = torch.sum(intermediate*z_dst, dim=-1)

        del intermediate
        return outputs

    def forward(self, z, edge_label_index):
        # Initialise scores array, which records the DistMult score for
        # each supervision edge and edge_label (rating)
        scores = torch.zeros(edge_label_index.shape[1], self.edge_labels.shape[0])
        for label in self.edge_labels:
            score = self.decode(z,edge_label_index, label)
            scores[:,label] = score
        # Return log probabilities for each class
        return(F.log_softmax(scores, dim=0))



class EdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, edge_labels, meta_data):
        super().__init__()
        # Set up GNN for message passing
        self.encoder = GNN(hidden_channels=hidden_channels, out_channels=hidden_channels)
        # Make the GNN model heterogeneous
        self.encoder = to_hetero(self.encoder, meta_data, aggr='sum')
        
        # DistMult based "decoder" for edge classification
        self.decoder = BilinearDecoder(hidden_channels, edge_labels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Perform message passing
        z = self.encoder(x_dict, edge_index_dict)
        
        # The edges that go into the decoder are the supervision edges (edge_label_index)
        out = self.decoder(z, edge_label_index)
        return(out)



edge_labels = torch.unique(data['user', 'rates', 'movie'].edge_label)

# Create message passing edge index for heterogeneous data.
mp_edge_index = {}
for key in ['train', 'val', 'test']:
    mp_edge_index[key] = {}
    for rating in edge_labels:
        mask = torch.squeeze(train_data['user', 'rates', 'movie'].edge_label==rating)
        indices =  torch.nonzero(mask)
        
        mp_edge_index[key][('user', f'rates_as_{rating}', 'movie')] = train_data[('user', 'rates', 'movie')].edge_index[:,mask]
        mp_edge_index[key][('movie', f'rev_rates_as_{rating}', 'user')] = torch.vstack([mp_edge_index[key][('user', f'rates_as_{rating}', 'movie')] [1], 
                                                                                        mp_edge_index[key][('user', f'rates_as_{rating}', 'movie')] [0]])


meta_data = (['movie', 'user'], list(mp_edge_index['train'].keys()))
print(meta_data)
hidden_dim = 64
model = EdgeClassifier(hidden_channels=hidden_dim, edge_labels=edge_labels, meta_data=meta_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)


# Get number of model parameters
# (need to run one model step so the number of parameters can be inferred, due to lazy initialisation)
with torch.no_grad():  
    model(train_data.x_dict, mp_edge_index['train'], train_data[('user', 'rates', 'movie')].edge_label_index)
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x_dict, mp_edge_index['train'], train_data[('user', 'rates', 'movie')].edge_label_index)
    
    # Get true ratings
    y = train_data[('user','rates','movie')].edge_label
    y = torch.squeeze(y)
    
    ## Apply multi class loss
    F.nll_loss(out, torch.squeeze(y)).backward()
    optimizer.step()

def test():
    model.eval()
    accs = []
    data_dict = {'train': train_data, 'val': val_data, 'test': test_data}
    for key, d in data_dict.items():
        logits = model(d.x_dict, mp_edge_index[key], d[('user', 'rates', 'movie')].edge_label_index)
        
        pred = logits.max(1)[1]
        y = d[('user','rates','movie')].edge_label
        y = torch.squeeze(y)
        
        acc = pred.eq(y).sum().item() / y.numel()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 400):
    train()
    
    if (epoch % 10 == 0) or (epoch == 1):
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        #print(log.format(epoch, train_acc, best_val_acc, test_acc))
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc))