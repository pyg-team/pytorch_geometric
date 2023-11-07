import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer, XGNNGenerator
from torch_geometric.nn import GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]

### REPLACE WITH BETTER DATASET EXAMPLE ################################

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
        # Define GCN layers to learn node features... TODO: Check 
        self.gcn_layers = nn.ModuleList([
            GCNConv(num_node_features, 16),
            GCNConv(16, 24),
            GCNConv(24, 32)
        ])
        # Define MLPs for predicting action probabilities
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
        # Combine candidate set with current graph state
        # Pass the combined graph through GCN layers
        node_features = graph_state.x
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features, graph_state.edge_index)
        
        # Predict starting node probability distribution
        start_node_probs = self.mlp_start_node(node_features)
        # Sample starting node based on the probability distribution
        start_node = torch.distributions.Categorical(start_node_probs).sample()

        # Concatenate the feature of the selected starting node with all node features
        # Predict ending node probability distribution
        combined_features = torch.cat((node_features, node_features[start_node].unsqueeze(0)), dim=0)
        end_node_probs = self.mlp_end_node(combined_features)
        # Sample ending node based on the probability distribution
        end_node = torch.distributions.Categorical(end_node_probs).sample()

        # Return the action (start_node, end_node) and the updated graph state
        return (start_node, end_node), graph_state

class RLGraphGen(XGNNGenerator):
    # Function to calculate the reward based on the GNN's predictions and graph validity
    def calculate_reward(graph_state, pre_trained_gnn):
        # Compute the GNN's predictions for the generated graph
        gnn_output = pre_trained_gnn(graph_state)
        # Determine the validity of the graph according to predefined rules
        graph_validity_score = ... # Your graph validity score logic
        # Calculate the reward
        reward = ... # Combine the GNN output and graph validity into a final reward
        return reward

    # Training function
    def train(graph_generator, pre_trained_gnn, num_epochs, learning_rate):
        optimizer = torch.optim.Adam(graph_generator.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0
            for step in range(max_steps):
                # Generate action and new graph state
                action, new_graph_state = graph_generator(current_graph_state, candidate_set)
                # Calculate reward based on the pre-trained GNN and graph rules
                reward = calculate_reward(new_graph_state, pre_trained_gnn)
                
                # Compute log probabilities for the actions taken
                start_node_log_prob = torch.log(action[0].probs[action[0].sample().item()])
                end_node_log_prob = torch.log(action[1].probs[action[1].sample().item()])
                log_probs = start_node_log_prob + end_node_log_prob
                
                # Compute loss as negative log probability times reward
                loss = -reward * log_probs
                total_loss += loss.item()

                # Perform backpropagation and update the graph generator parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Roll back if the action is not promising
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

#########################################################################

explainer = Explainer(
    model=model,
    algorithm=XGNNExplainer(generative_model = RLGraphGen(num_node_features, 
                                                          num_output_features,
                                                          num_candidate_node_types), 
                            epochs = 200),
    explanation_type='model',
    # node_mask_type='attributes',
    # edge_mask_type='object',
    model_config=dict(
        # ADD ATRIBUTES
        # mode='multiclass_classification',
        # task_level='node',
        # return_type='log_probs',
    ),
)

class_index = 1
explanation = explainer(data.x, data.edge_index) # explained_class=class_index
print(explanation)

# print(f'Generated explanations in {explanation.available_explanations}') # ??

# path = "explanation_graph.png"
# explanation.visualize_subgraph(path, )

# path = 'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.pdf'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")
