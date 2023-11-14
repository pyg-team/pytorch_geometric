import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer, XGNNGenerator
from torch_geometric.nn import GCNConv


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
        node_features = graph_state.x
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features, graph_state.edge_index)
        

        start_node_probs = self.mlp_start_node(node_features)
        start_node = torch.distributions.Categorical(start_node_probs).sample()

        combined_features = torch.cat((node_features, node_features[start_node].unsqueeze(0)), dim=0)
        end_node_probs = self.mlp_end_node(combined_features)

        end_node = torch.distributions.Categorical(end_node_probs).sample()

        return (start_node, end_node), graph_state

class RLGenTrainer(XGNNGenerator):

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
    algorithm=XGNNExplainer(generative_model = XGNNGenerator(), epochs = 200),
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

