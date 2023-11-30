import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer, XGNNTrainer
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool

import random
# # TODO: Get our acyclic dataset
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, dataset)
# data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

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
    
args = {
    'device': device,
    'input_dim' : 1,
    'gcn_output_dim' : [8, 16],
    'dropout': 0.5,
    'lr': 0.001,
    'weight_decay' : 0.00001,
    'epochs': 30,
}

model = GCN_Graph(args['input_dim'], args['gcn_output_dim'],
                  output_dim=1, dropout=args['dropout'])
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)  # Don't forget to move the model to the correct device

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

    def calculate_reward(graph_state, pre_trained_gnn, target_class, num_classes):
        gnn_output = pre_trained_gnn(graph_state)
        class_score = gnn_output[target_class]
        intermediate_reward = class_score - 1 / num_classes

        # Assuming rollout function is defined to perform graph rollouts and evaluate
        final_graph_reward = self.rollout_reward(graph_state, pre_trained_gnn, target_class, num_classes)

        # Compute graph validity score (R_tr)
        # This needs to be defined based on the specific graph rules of your dataset
        graph_validity_score = self.evaluate_graph_validity(graph_state) 

        # Combine the rewards
        lambda_1, lambda_2 = 1, 1  # Hyperparameters, can be tuned
        reward = intermediate_reward + lambda_1 * final_graph_reward + lambda_2 * graph_validity_score
        return reward

    # Training function
    def train(graph_generator, pre_trained_gnn, num_epochs, learning_rate):
        optimizer = torch.optim.Adam(graph_generator.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0
            for step in range(max_steps):

                action, new_graph_state = graph_generator(current_graph_state, candidate_set)
                reward = calculate_reward(new_graph_state, pre_trained_gnn, target_class, num_classes)
                
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
pre_trained_gnn = GCN().to(device) # TODO
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

explainer = Explainer(
    model=model,
    algorithm=XGNNExplainer(generative_model = XGNNTrainer(target_class), epochs = 200),
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

