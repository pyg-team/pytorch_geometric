import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer, XGNNTrainer
from torch_geometric.nn import GCNConv

import random

import torch
import pandas as pd
import networkx as nx
import random
import torch.nn.functional as F
# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T

from torch_geometric.data import DataLoader, Data, Dataset
from tqdm.notebook import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
import copy
class AcyclicGraphDataset(Dataset):
    def __init__(self, pyg_dataset):
        super(AcyclicGraphDataset, self).__init__()
        self.pyg_dataset = pyg_dataset

    def len(self):
        return len(self.pyg_dataset)

    def get(self, idx):
        return self.pyg_dataset[idx]

class CyclicGraphDataset(Dataset):
    def __init__(self, data_list):
        super(CyclicGraphDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

cyclic_dataset = torch.load("/content/claritynet/notebooks/cyclic_dataset.pt")
acyclic_dataset = torch.load("/content/claritynet/notebooks/acyclic_dataset.pt")
print('The {} dataset has {} graphs'.format("cyclic", len(cyclic_dataset)))
print('The {} dataset has {} graphs'.format("acyclic", len(acyclic_dataset)))
cyclic_data = cyclic_dataset[0]
acyclic_data = acyclic_dataset[0]

class IsAcyclic(Dataset):
    def __init__(self, cyclic_data, acyclic_data):
        super(IsAcyclic, self).__init__()
        self.cyclic_data = cyclic_data
        self.acyclic_data = acyclic_data
        # Combine the two datasets
        
        self.data_list = [(data, 0) for data in cyclic_data] + [(data, 1) for data in acyclic_data]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data, label = self.data_list[idx]
        # Ensure the label is a tensor and attach it to the data object
        data.y = torch.tensor([label], dtype=torch.float)
        return data

    def get_idx_split(self, train_ratio=0.7, val_ratio=0.15):
        def split_indices(data, train_ratio, val_ratio):
            dataset_size = len(data)
            indices = list(range(dataset_size))
            random.shuffle(indices)

            train_split = int(train_ratio * dataset_size)
            val_split = int(val_ratio * dataset_size) + train_split

            return indices[:train_split], indices[train_split:val_split], indices[val_split:]

        # Split cyclic and acyclic datasets separately
        cyclic_train, cyclic_val, cyclic_test = split_indices(self.cyclic_data, train_ratio, val_ratio)
        acyclic_train, acyclic_val, acyclic_test = split_indices(self.acyclic_data, train_ratio, val_ratio)

        # Offset acyclic indices by the size of cyclic dataset
        offset = len(self.cyclic_data)
        acyclic_train = [i + offset for i in acyclic_train]
        acyclic_val = [i + offset for i in acyclic_val]
        acyclic_test = [i + offset for i in acyclic_test]

        # Combine the splits from cyclic and acyclic datasets
        train_indices = cyclic_train + acyclic_train
        val_indices = cyclic_val + acyclic_val
        test_indices = cyclic_test + acyclic_test

        # Shuffle combined splits to mix cyclic and acyclic graphs
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        return {
            'train': train_indices,
            'valid': val_indices,
            'test': test_indices
        }

# Assuming 'cyclic_dataset' and 'acyclic_dataset' are already created as per your provided code
dataset = IsAcyclic(cyclic_dataset, acyclic_dataset)

torch.save(dataset, 'is_acyclic.pt')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

split_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)

args = {
    'device': device,
    'input_dim' : 1,
    'gcn_output_dim' : [8, 16],
    'dropout': 0.4,
    'lr': 0.01,
    'weight_decay' : 0.00001,
    'epochs': 100,
}

class GCN(torch.nn.Module):
    def __init__(self, input_dim, gcn_output_dims, dropout, return_embeds=False):
        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        self.convs = torch.nn.ModuleList([GCNConv(in_channels=input_dim, out_channels=gcn_output_dims[0])])
        self.convs.extend([GCNConv(in_channels=gcn_output_dims[i + 0], out_channels=gcn_output_dims[i + 1]) for i in range(len(gcn_output_dims) - 1)])

        self.bns = torch.nn.ModuleList([BatchNorm1d(num_features=gcn_output_dims[l]) for l in range(len(gcn_output_dims) - 1)])
        
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = None

        for i in range(len(self.convs)-1):
          x = F.relu(self.bns[i](self.convs[i](x, adj_t)))
          if self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
          out = x
        else:
          out = self.softmax(x)

        return out

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, gcn_output_dims, output_dim, dropout):
        super(GCN_Graph, self).__init__()

        # self.node_encoder = AtomEncoder(hidden_dim)
        
        self.gnn_node = GCN(input_dim, gcn_output_dims, dropout, return_embeds=True)

        self.pool = global_mean_pool # global averaging to obtain graph representation

        # Output layer
        self.linear = torch.nn.Linear(gcn_output_dims[-1], output_dim) # One fully connected layer as a classifier


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        
        device = edge_index.device
        degrees = torch.sum(edge_index[0] == torch.arange(edge_index.max() + 1, device=device)[:, None], dim=1, dtype=torch.float)
        x = degrees.unsqueeze(1)  # Add feature dimension
        embed = x.to(device)  # Ensure the embedding tensor is on the correct device

        out = None

        node_embeddings = self.gnn_node(embed, edge_index)
        agg_features = self.pool(node_embeddings, batch)
        out = self.linear(agg_features)

        return out

def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          pass
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        optimizer.zero_grad()
        out = model(batch)
        filtered_output = out[is_labeled]

        # Reshape the labels to match the output shape
        filtered_labels = batch.y[is_labeled].unsqueeze(1).type(torch.float32)

        loss = loss_fn(filtered_output, filtered_labels)

        loss.backward()
        optimizer.step()

    return loss.item()

def compute_accuracy(y_true, y_pred):
    # Assuming y_pred are logits; apply sigmoid and round off to get binary predictions
    preds = torch.sigmoid(y_pred) > 0.5
    correct = preds.eq(y_true.view_as(preds)).sum()
    accuracy = correct.float() / y_true.numel()
    return accuracy.item()

def eval(model, device, loader):
    model.eval()
    total_accuracy = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        # Assuming binary classification and batch.y is your ground truth
        accuracy = compute_accuracy(batch.y, pred)
        total_accuracy += accuracy * batch.y.size(0)
        total_samples += batch.y.size(0)

    return total_accuracy / total_samples


model = GCN_Graph(args['input_dim'], args['gcn_output_dim'],
            output_dim=1, dropout=args['dropout']).to(device)
# evaluator = Evaluator(name='ogbg-molhiv')

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr']) #, weight_decay=args['weight_decay']
loss_fn = torch.nn.BCEWithLogitsLoss()

best_model = None
best_valid_acc = 0

# # Training loop remains the same...
# save_path = '/content/claritynet/best_model.pth'
# # Evaluation in your main loop
# for epoch in range(1, 1 + args["epochs"]):
#     print('Training...')
#     train_loss = train(model, device, train_loader, optimizer, loss_fn)

#     print('Evaluating...')
#     train_acc = eval(model, device, train_loader)
#     val_acc = eval(model, device, valid_loader)
#     test_acc = eval(model, device, test_loader)

#     if val_acc > best_valid_acc:
#         best_valid_acc = val_acc
#         best_model = copy.deepcopy(model)
#         torch.save(best_model.state_dict(), save_path)

#     print(f'Epoch: {epoch:02d}, '
#           f'Loss: {train_loss:.4f}, '
#           f'Train Acc: {100 * train_acc:.2f}%, '
#           f'Valid Acc: {100 * val_acc:.2f}% '
#           f'Test Acc: {100 * test_acc:.2f}%')

# # Evaluate the best model
# best_train_acc = eval(best_model, device, train_loader)
# best_val_acc = eval(best_model, device, valid_loader)
# best_test_acc = eval(best_model, device, test_loader)

# print(f'Best model: '
#       f'Train: {100 * best_train_acc:.2f}%, '
#       f'Valid: {100 * best_val_acc:.2f}% '
#       f'Test: {100 * best_test_acc:.2f}%')



# TODO: Get our acyclic dataset
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphGenerator(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features, num_candidate_node_types):
        super(GraphGenerator, self).__init__()
        # TODO: Check 
        self.gcn_layers = torch.nn.ModuleList([
            GCNConv(num_node_features, 16),
            GCNConv(16, 24),
            GCNConv(24, 32)
        ])

        self.mlp_start_node = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU6(),
            torch.nn.Linear(16, num_candidate_node_types),
            torch.nn.Softmax(dim=-1)
        )
        self.mlp_end_node = torch.nn.Sequential(
            torch.nn.Linear(32, 24),
            torch.nn.ReLU6(),
            torch.nn.Linear(24, num_candidate_node_types),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, graph_state, candidate_set):
        # contatenate graph_state features with candidate_set features
        node_features_graph = graph_state.x
        node_features = torch.cat((node_features, candidate_set), dim=0)

        # run through gcn layers
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features, graph_state.edge_index)
        
        # get start node probabilities and mask out candidates
        start_node_probs = self.mlp_start_node(node_features)
        candidate_set_mask = torch.ones_like(start_node_probs)
        candidate_set_mask[candidate_set] = 0
        start_node_probs = start_node_probs * candidate_set_mask

        # sample start node
        start_node = torch.distributions.Categorical(start_node_probs).sample()

        # get end node probabilities and mask out start node
        combined_features = torch.cat((node_features, node_features[start_node].unsqueeze(0)), dim=0)
        end_node_probs = self.mlp_end_node(combined_features)
        end_node_probs[start_node] = 0

        # sample end node
        end_node = torch.distributions.Categorical(end_node_probs).sample()

        return (start_node, end_node), graph_state

class RLGenTrainer(XGNNTrainer):

    def calculate_reward(graph_state, pre_trained_gnn):
        gnn_output = pre_trained_gnn(graph_state)
        # TODO: Implement
        graph_validity_score = ... 
        reward = ... 
        return reward

    # Training function
    def train(graph_generator, pre_trained_gnn, num_epochs, learning_rate):
        optimizer = torch.optim.Adam(graph_generator.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0
            for step in range(max_steps):

                action, new_graph_state = graph_generator(current_graph_state, candidate_set)
                reward = calculate_reward(new_graph_state, pre_trained_gnn)
                
                start_node_log_prob = torch.log(action[0].probs[action[0].sample().item()])
                end_node_log_prob = torch.log(action[1].probs[action[1].sample().item()])
                log_probs = start_node_log_prob + end_node_log_prob
                
                loss = -reward * log_probs
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if reward < 0:
                    current_graph_state = previous_graph_state
            print(f"Epoch {epoch} completed, Total Loss: {total_loss}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


explainer = Explainer(
    model=model,
    algorithm=XGNNExplainer(generative_model = XGNNTrainer(), epochs = 200),
    explanation_type='model',
    node_mask_type=None,
    edge_mask_type=None,
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

print("EXPLAINER DONE!")

class_index = 1
explanation = explainer(data.x, data.edge_index) # Generates explanations for all classes at once
print(explanation)

