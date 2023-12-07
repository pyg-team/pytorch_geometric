import os.path as osp
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, XGNNExplainer
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset


import random
# # TODO: Get our acyclic dataset
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, dataset)
# data = dataset[0]
from xgnn_model import GCN_Graph

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {'device': device,
        'dropout': 0.1,
        'epochs': 1000,
        'input_dim' : 7,
        'opt': 'adam',
        'opt_scheduler': 'none',
        'opt_restart': 0,
        'weight_decay': 5e-5,
        'lr': 0.001}
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
args = objectview(args)
        
model = GCN_Graph(args.input_dim, output_dim=2, dropout=args.dropout).to(device)

# depending on os change path
path = "examples/explain/xgnn/mutag_model.pth"
if os.name == 'nt':
    path = "examples\\explain\\xgnn\\mutag_model.pth"

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.to(device)

def create_single_batch(dataset):
    data_list = [data for data in dataset]
    batched_data = Batch.from_data_list(data_list)
    return batched_data

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
        print("debug: GraphGenerator init")
        print("num_node_features", num_node_features)
        print("num_candidate_node_types", num_candidate_node_types)
        print("debug: GraphGenerator init done")
        self.gcn_layers = torch.nn.ModuleList([
            GCNConv(num_node_features, 16),
            GCNConv(16, 24),
            GCNConv(24, 32)
        ])

        self.mlp_start_node = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU6(),
            torch.nn.Linear(16, 1),
            torch.nn.Softmax(dim=0)
        )
        self.mlp_end_node = torch.nn.Sequential(
            torch.nn.Linear(32, 24),
            torch.nn.ReLU6(),
            torch.nn.Linear(24, 1),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, graph_state, candidate_set):
        # contatenate graph_state features with candidate_set features
        node_features_graph = graph_state.x
        node_features = torch.cat((node_features_graph, candidate_set), dim=0).float()

        # run through GCN layers
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features, graph_state.edge_index)
        # get start node probabilities and mask out candidates
        start_node_probs = self.mlp_start_node(node_features)
        
        candidate_set_mask = torch.ones_like(start_node_probs)
        candidate_set_indices = torch.arange(node_features_graph.shape[0], node_features.shape[0])
        candidate_set_mask[candidate_set_indices] = 0
        start_node_probs = start_node_probs * candidate_set_mask
        # change 0 probabilities to very small number
        #start_node_probs[start_node_probs == 0] = 1e-10
        start_node_probs = start_node_probs.squeeze()

        # sample start node
        p_start = torch.distributions.Categorical(start_node_probs)
        start_node = p_start.sample()
        
        # get end node probabilities and mask out start node
        # combined_features = torch.cat((node_features, node_features[start_node].unsqueeze(0)), dim=0)
        end_node_probs = self.mlp_end_node(node_features)
        start_node_mask = torch.ones_like(end_node_probs)
        start_node_mask[start_node] = 0
        end_node_probs = end_node_probs * start_node_mask
        # change 0 probabilities to very small number
        #end_node_probs[end_node_probs == 0] = 1e-10
        end_node_probs = end_node_probs.squeeze()
        # sample end node
        end_node = torch.distributions.Categorical(end_node_probs).sample()
        if end_node >= graph_state.x.shape[0]: 
            # add new node features to graph state
            graph_state.x = torch.cat((graph_state.x, candidate_set[end_node - graph_state.x.shape[0]].unsqueeze(0)), dim=0).float()
            new_edge = torch.tensor([[start_node], [graph_state.x.shape[0] - 1]])
        else: 
            new_edge = torch.tensor([[start_node], [end_node]])
        graph_state.edge_index = torch.cat((graph_state.edge_index, new_edge), dim=1)
        
        # one hot encoding of start and end node
        start_node_one_hot = torch.zeros_like(start_node_probs)
        start_node_one_hot[start_node] = 1
        end_node_one_hot = torch.zeros_like(end_node_probs)
        end_node_one_hot[end_node] = 1
        
        #print(graph_state.x.size())
        return ((start_node_probs, start_node_one_hot), (end_node_probs, end_node_one_hot)), graph_state

class RLGenExplainer(XGNNExplainer):
    def __init__(self, candidate_set):
        super(RLGenExplainer, self).__init__()
        self.candidate_set = candidate_set
        self.graph_generator = GraphGenerator(self.candidate_set.size(1), self.candidate_set.size(0))
        self.max_steps = 10
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.num_classes = 2
    
    def reward_tf(self, pre_trained_gnn, graph_state, target_class, num_classes):
        graph_state_batch = create_single_batch([graph_state,])
        
        # Move graph batch to the same device as your model
        graph_state_batch = graph_state_batch.to(device)

        # ...
        pre_trained_gnn.eval()
        with torch.no_grad():
            gnn_output = pre_trained_gnn(graph_state_batch)
            probability_of_target_class = gnn_output[0][target_class]
        return probability_of_target_class - 1 / num_classes
    
    def rollout_reward(self, intermediate_graph_state, pre_trained_gnn, target_class, num_classes, num_rollouts=5):
        final_rewards = []
        for _ in range(num_rollouts):
            # Generate a final graph from the intermediate graph state
            _, final_graph = self.graph_generator(intermediate_graph_state, self.candidate_set)
            # Evaluate the final graph
            with torch.no_grad():
                reward = self.reward_tf(pre_trained_gnn, final_graph, target_class, num_classes)
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
        intermediate_reward = self.reward_tf(pre_trained_gnn, graph_state, target_class, num_classes)
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
            random_index = torch.randint(0, self.candidate_set.size(0), (1,))
            sampled_node = self.candidate_set[random_index] # sample a node from the candidate set
            edge_index = torch.tensor([], dtype=torch.long).view(2, -1)
            initial_graph = Data(x=sampled_node, edge_index=edge_index)
            current_graph_state = initial_graph
            
            for step in range(self.max_steps):
                ((p_start, a_start), (p_end, a_end)), new_graph_state = self.graph_generator(current_graph_state, self.candidate_set)
                
                reward = self.calculate_reward(new_graph_state, model_to_explain, for_class, self.num_classes)
                
                LCE_start = F.cross_entropy(p_start, a_start)
                LCE_end = F.cross_entropy(p_end, a_end)
                
                loss = -reward * (LCE_start + LCE_end)
                total_loss += -reward * (LCE_start.item() + LCE_end.item())
                
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

# extract features for the candidate set
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
all_features = torch.cat([data.x for data in dataset], dim=0)
candidate_set = torch.unique(all_features, dim=0)

explainer = Explainer(
    model = model,
    algorithm = RLGenExplainer(candidate_set=candidate_set),
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

