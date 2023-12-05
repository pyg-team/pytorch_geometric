import os.path as osp
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool

import random
# # TODO: Get our acyclic dataset
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, dataset)
# data = dataset[0]

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
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
        print("batched_data", batched_data)
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        
        device = edge_index.device
        print("edge_index shape:", edge_index.shape)
        print("edge_index", edge_index)
        
        if edge_index.numel() == 0:
            # if edge_index is empty
            degrees = torch.zeros(x.size(0), dtype=torch.float, device=device)
        else:
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

# depending on os change path
path = "examples/explain/best_model.pth"
if os.name == 'nt':
    path = "examples\\explain\\best_model.pth"

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.to(device)

def check_edge_representation(data):
    # Convert edge indices to a set of tuples for easier comparison
    edge_set = {tuple(edge) for edge in data.edge_index.t().tolist()}

    for edge in edge_set:
        reverse_edge = (edge[1], edge[0])
        if reverse_edge not in edge_set:
            return "Edges are represented as single, undirected edges"
    return "Edges are represented with two directed edges, one in each direction"

class GraphGenerator(torch.nn.Module):
    def __init__(self, num_node_features, num_candidate_node_types):
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
            torch.nn.Softmax(dim=0)
        )
        self.mlp_end_node = torch.nn.Sequential(
            torch.nn.Linear(32, 24),
            torch.nn.ReLU6(),
            torch.nn.Linear(24, num_candidate_node_types),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, graph_state, candidate_set):
        # contatenate graph_state features with candidate_set features
        node_features_graph = graph_state.x
        node_features = torch.cat((node_features_graph, candidate_set), dim=0).float()

        # run through GCN layers
        for gcn_layer in self.gcn_layers:
            print("node_features shape:", node_features.shape)
            print("graph_state.edge_index shape:", graph_state.edge_index.shape)
            node_features = gcn_layer(node_features, graph_state.edge_index)
        
        # get start node probabilities and mask out candidates
        start_node_probs = self.mlp_start_node(node_features)

        candidate_set_mask = torch.ones_like(start_node_probs)
        candidate_set_mask[candidate_set] = 0
        start_node_probs = start_node_probs * candidate_set_mask

        # change 0 probabilities to very small number
        start_node_probs[start_node_probs == 0] = 1e-10

        # sample start node
        start_node = torch.distributions.Categorical(start_node_probs).sample()

        # get end node probabilities and mask out start node
        combined_features = torch.cat((node_features, node_features[start_node]), dim=0)
        end_node_probs = self.mlp_end_node(combined_features)
        end_node_probs[start_node] = 0
        
        # change 0 probabilities to very small number
        end_node_probs[end_node_probs == 0] = 1e-10
        
        # sample end node
        end_node = torch.distributions.Categorical(end_node_probs).sample()
        
        print("graph generator output:", (start_node, end_node), graph_state)
        return (start_node, end_node), graph_state

class RLGenExplainer(XGNNExplainer):
    def __init__(self):
        super(RLGenExplainer, self).__init__()
        self.candidate_set = torch.tensor([[0]])  # 2d tensor of features of candidate nodes (node types)
        self.graph_generator = GraphGenerator(1, self.candidate_set.size(0))
        self.max_steps = 10
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.num_classes = 2
    
    def rollout_reward(self, intermediate_graph_state, pre_trained_gnn, target_class, num_classes, num_rollouts=5):
        final_rewards = []
        for _ in range(num_rollouts):
            # Generate a final graph from the intermediate graph state
            final_graph = self.graph_generator(intermediate_graph_state, self.candidate_set)
            print("final_graph", final_graph)
            
            # Evaluate the final graph
            with torch.no_grad():
                gnn_output = pre_trained_gnn(final_graph)
                class_score = gnn_output[target_class]
                reward = class_score - 1 / num_classes
                final_rewards.append(reward)

        # Average the rewards from all rollouts
        average_final_reward = sum(final_rewards) / len(final_rewards)
        return average_final_reward

    def evaluate_graph_validity(self, graph_state):
        # check if graph has duplicated edges
        
        edge_set = set()
        
        for edge in graph_state.edge_index:
            sorted_edge = tuple(sorted(edge))
            
            if sorted_edge in edge_set:
                return -1
            
            edge_set.add(sorted_edge)
            
        return 0
        
        
    def calculate_reward(self, graph_state, pre_trained_gnn, target_class, num_classes):
        gnn_output = pre_trained_gnn(graph_state)
        class_score = gnn_output[target_class]
        intermediate_reward = class_score - 1 / num_classes

        # Assuming rollout function is defined to perform graph rollouts and evaluate
        final_graph_reward = self.rollout_reward(graph_state, pre_trained_gnn, target_class, num_classes)

        # Compute graph validity score (R_tr)
        # defined based on the specific graph rules of the dataset
        graph_validity_score = self.evaluate_graph_validity(graph_state) 

        reward = intermediate_reward + self.lambda_1 * final_graph_reward + self.lambda_2 * graph_validity_score
        return reward

    # Training function
    def train_generative_model(self, model_to_explain, for_class, num_epochs, learning_rate):
        optimizer = torch.optim.Adam(self.graph_generator.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0
            
            # sample from candidate set and create initial graph state
            n = 1 
            perm = torch.randperm(self.candidate_set.size(0))
            sampled_indices = perm[:n]
            x = self.candidate_set[sampled_indices].view(n, 1)  # reshaping to [n, num_features]
            edge_index = torch.tensor([[], []], dtype=torch.long)
            initial_state = Data(x=x, edge_index=edge_index)

            current_graph_state = initial_state

            print("candidate_set (file:  examples/explain/xgnn_explainer.py)", self.candidate_set.shape)
            
            for step in range(self.max_steps):
                action, new_graph_state = self.graph_generator(current_graph_state, self.candidate_set)
                print(action)
                reward = self.calculate_reward(new_graph_state, model_to_explain, for_class, self.num_classes)
                
                start_node_log_prob = torch.log(action[0].probs[action[0].sample().item()])
                end_node_log_prob = torch.log(action[1].probs[action[1].sample().item()])
                log_probs = start_node_log_prob + end_node_log_prob
                
                loss = -reward * log_probs
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if reward >= 0:
                    current_graph_state = new_graph_state
            print(f"Epoch {epoch} completed, Total Loss: {total_loss}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GCN().to(device)
#data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

explainer = Explainer(
    model = model,
    algorithm = RLGenExplainer(),
    explanation_type = 'generative',
    node_mask_type = None,
    edge_mask_type = None,
    model_config = dict(
        mode = 'multiclass_classification',
        task_level = 'node',
        return_type = 'log_probs',
    ),
)

print("EXPLAINER DONE!")

class_index = 1
target = torch.tensor([0, 1])

explanation = explainer(None, None, target=target) # Generates explanations for all classes at once
print(explanation)

