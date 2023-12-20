import copy
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import trange
from xgnn_model import GCN_Graph

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.explain import (
    Explainer,
    ExplanationSetSampler,
    XGNNExplainer,
)
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx


def masked_softmax(vector, mask):
    """Apply softmax to only selected elements of the vector, as indicated by
    the mask. The output will be a probability distribution where unselected
    elements are 0.

    Args:
        vector (torch.Tensor): Input vector.
        mask (torch.Tensor): Mask indicating which elements to softmax.

    Returns:
        torch.Tensor: Softmaxed vector.
    """
    mask = mask.bool()
    masked_vector = vector.masked_fill(~mask, float('-inf'))
    softmax_result = F.softmax(masked_vector, dim=0)

    return softmax_result


class GraphGenerator(torch.nn.Module, ExplanationSetSampler):
    """Graph generator that generates a new graph state from a given graph
    state.

    Inherits:
        torch.nn.Module: Base class for all neural network modules.
        ExplanationSetSampler: Base class for sampling from an explanation set.

    Args:
        candidate_set (dict): Set of candidate nodes for graph generation.
        dropout (float): Dropout rate for regularization.
        initial_node_type (str, optional): Initial node type for graph
        initialization.
    """
    def __init__(self, candidate_set, dropout, initial_node_type=None):
        super(GraphGenerator, self).__init__()
        # TODO: Check
        self.candidate_set = candidate_set
        self.initial_node_type = initial_node_type
        self.dropout = dropout
        num_node_features = len(next(iter(self.candidate_set.values())))
        self.gcn_layers = torch.nn.ModuleList([
            GCNConv(num_node_features, 16),
            GCNConv(16, 24),
            GCNConv(24, 32),
        ])

        self.mlp_start_node = torch.nn.Sequential(torch.nn.Linear(32, 16),
                                                  torch.nn.Linear(16, 1),
                                                  torch.nn.ReLU6())
        self.mlp_end_node = torch.nn.Sequential(torch.nn.Linear(32, 24),
                                                torch.nn.Linear(24, 1),
                                                torch.nn.ReLU6())

    def initialize_graph_state(self, graph_state):
        r"""Initializes the graph state with a single node.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to
            initialize.

        Returns:
            torch_geometric.data.Data: The initialized graph state.
        """
        if self.initial_node_type is None:
            keys = list(self.candidate_set.keys())
            self.initial_node_type = keys[torch.randint(len(keys),
                                                        (1, )).item()]

        feature = candidate_set[self.initial_node_type].unsqueeze(0)
        edge_index = torch.tensor([], dtype=torch.long).view(2, -1)
        node_type = [
            self.initial_node_type,
        ]
        # update graph state
        graph_state.x = feature
        graph_state.edge_index = edge_index
        graph_state.node_type = node_type

    def forward(self, graph_state):
        """Generates a new graph state from the given graph state.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to
            generate a new graph state from.

        Returns:
            ((torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)): The
            logits and one hot encodings for the start and end node.
            torch_geometric.data.Data: The new graph state.
        """
        graph_state = copy.deepcopy(graph_state)
        if graph_state.x.shape[0] == 0:
            self.initialize_graph_state(
                graph_state)  # initialize graph state if it is empty

        # contatenate graph_state features with candidate_set features
        node_features_graph = graph_state.x.detach().clone()
        candidate_features = torch.stack(list(self.candidate_set.values()))
        node_features = torch.cat((node_features_graph, candidate_features),
                                  dim=0).float()
        node_edges = graph_state.edge_index.detach().clone()

        # compute node encodings with GCN layers
        node_encodings = node_features
        for gcn_layer in self.gcn_layers:
            node_encodings = F.relu6(gcn_layer(node_encodings, node_edges))
            node_encodings = F.dropout(node_encodings, self.dropout,
                                       training=self.training)

        # get start node probabilities and mask out candidates
        start_node_logits = self.mlp_start_node(node_encodings)

        candidate_set_mask = torch.ones_like(start_node_logits)
        candidate_set_indices = torch.arange(node_features_graph.shape[0],
                                             node_encodings.shape[0])
        # set candidate set probabilities to 0
        candidate_set_mask[candidate_set_indices] = 0
        start_node_probs = masked_softmax(start_node_logits,
                                          candidate_set_mask).squeeze()

        # sample start node
        p_start = torch.distributions.Categorical(start_node_probs)
        start_node = p_start.sample()

        # get end node probabilities and mask out start node
        end_node_logits = self.mlp_end_node(node_encodings)
        start_node_mask = torch.ones_like(end_node_logits)
        start_node_mask[start_node] = 0
        end_node_probs = masked_softmax(end_node_logits,
                                        start_node_mask).squeeze()

        # sample end node
        end_node = torch.distributions.Categorical(end_node_probs).sample()
        num_nodes_graph = graph_state.x.shape[0]
        if end_node >= num_nodes_graph:
            # add new node features to graph state
            graph_state.x = torch.cat(
                [graph_state.x, node_features[end_node].unsqueeze(0).float()],
                dim=0)
            graph_state.node_type.append(
                list(self.candidate_set.keys())[end_node - num_nodes_graph])
            new_edge = torch.tensor([[start_node], [num_nodes_graph]])
        else:
            new_edge = torch.tensor([[start_node], [end_node]])
        graph_state.edge_index = torch.cat((graph_state.edge_index, new_edge),
                                           dim=1)

        # one hot encoding of start and end node
        start_node_one_hot = torch.eye(start_node_probs.shape[0])[start_node]
        end_node_one_hot = torch.eye(end_node_probs.shape[0])[end_node]

        return ((start_node_logits.squeeze(), start_node_one_hot),
                (end_node_logits.squeeze(), end_node_one_hot)), graph_state

    def sample(self, num_samples: int, **kwargs):
        """Samples a number of graphs from the generator.

        Args:
            num_samples (int): The number of graphs to sample.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If neither num_nodes nor max_steps is specified.

        Returns:
            List[torch_geometric.data.Data]: The list of sampled graphs.
        """
        # extract num_nodes and max_steps from kwargs or set them to None
        num_nodes = kwargs.get('num_nodes', None)
        max_steps = kwargs.get('max_steps', None)

        # check that either num_nodes or max_steps is not None
        if num_nodes is None and max_steps is None:
            raise ValueError("Either num_nodes or max_steps must be specified")

        # create empty graph state
        empty_graph = Data(x=torch.tensor([]), edge_index=torch.tensor([]),
                           node_type=[])
        current_graph_state = copy.deepcopy(empty_graph)

        # sample graphs
        sampled_graphs = []

        max_steps_reached = False
        num_nodes_reached = False
        self.eval()
        for _ in range(num_samples):
            step = 0
            while not max_steps_reached and not num_nodes_reached:
                G = copy.deepcopy(current_graph_state)
                ((p_start, a_start),
                 (p_end, a_end)), current_graph_state = self.forward(G)
                step += 1
                # check if max_steps is reached
                max_steps_reached = max_steps is not None and step >= max_steps
                # check if num_nodes is reached
                num_nodes_reached = (num_nodes is not None
                                     and
                                     current_graph_state.x.shape[0] > num_nodes
                                     )
            # add sampled graph to list
            sampled_graphs.append(G)
            # reset current graph state
            current_graph_state = copy.deepcopy(empty_graph)
            # reset max_steps_reached and num_nodes_reached
            max_steps_reached = False
            num_nodes_reached = False
        return sampled_graphs


class RLGenExplainer(XGNNExplainer):
    """RL-based generator for graph explanations using XGNN.

    Inherits:
        XGNNExplainer: Base class for explanation generation using XGNN method.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        candidate_set (dict): Set of candidate nodes for graph generation.
        validity_args (dict): Arguments for graph validity check.
        initial_node_type (str, optional): Initial node type for graph
        initialization.
    """
    def __init__(self, epochs, lr, candidate_set, validity_args,
                 initial_node_type=None):
        super(RLGenExplainer, self).__init__(epochs, lr)
        self.candidate_set = candidate_set
        self.graph_generator = GraphGenerator(candidate_set, 0.1,
                                              initial_node_type)
        self.max_steps = 10
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.num_classes = 2
        self.validity_args = validity_args

    def reward_tf(self, pre_trained_gnn, graph_state, num_classes):
        r"""Computes the reward for the given graph state by evaluating the
        graph with the pre-trained GNN.

        Args:
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward. graph_state (torch_geometric.data.Data): The
            graph state to compute the reward for.
            num_classes (int): The number of classes in the dataset.

        Returns:
            torch.Tensor: The reward for the given graph state.
        """
        with torch.no_grad():
            gnn_output = pre_trained_gnn(graph_state)
            probability_of_target_class = torch.sigmoid(gnn_output).squeeze()

        return probability_of_target_class - 1 / num_classes

    def rollout_reward(self, intermediate_graph_state, pre_trained_gnn,
                       target_class, num_classes, num_rollouts=5):
        r"""Computes the rollout reward for the given graph state.

        Args:
            intermediate_graph_state (torch_geometric.data.Data): The
            intermediate graph state to compute the rollout reward for.
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward.
            target_class (int): The target class to explain.
            num_classes (int): The number of classes in the dataset.
            num_rollouts (int): The number of rollouts to perform.

        Returns:
            float: The average rollout reward for the given graph state.
        """
        final_rewards = []
        for _ in range(num_rollouts):
            # make copy of intermediate graph state
            intermediate_graph_state_copy = copy.deepcopy(
                intermediate_graph_state)
            _, final_graph = self.graph_generator(
                intermediate_graph_state_copy)
            # Evaluate the final graph
            reward = self.reward_tf(pre_trained_gnn, final_graph, num_classes)
            final_rewards.append(reward)

            del intermediate_graph_state_copy

        average_final_reward = sum(final_rewards) / len(final_rewards)
        return average_final_reward

    def evaluate_graph_validity(self, graph_state):
        r"""Evaluates the validity of the given graph state. Dataset specific
        graph rules are implemented here.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to
            evaluate.

        Returns:
            int: The graph validity score. 0 if the graph is valid,
            -1 otherwise.
        """
        # For mutag, node degrees cannot exceed valency
        degrees = torch.bincount(graph_state.edge_index.flatten(),
                                 minlength=graph_state.num_nodes)
        node_type_valencies = torch.tensor(
            [self.validity_args[type_] for type_ in graph_state.node_type])
        if torch.any(degrees > node_type_valencies):
            return -1
        return 0

    def calculate_reward(self, graph_state, pre_trained_gnn, target_class,
                         num_classes):
        r"""Calculates the reward for the given graph state.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to compute
            the reward for.
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward.
            target_class (int): The target class to explain.
            num_classes (int): The number of classes in the dataset.

        Returns:
            torch.Tensor: The final reward for the given graph state.
        """
        intermediate_reward = self.reward_tf(pre_trained_gnn, graph_state,
                                             num_classes)
        final_graph_reward = self.rollout_reward(graph_state, pre_trained_gnn,
                                                 target_class, num_classes)
        # Compute graph validity score (R_tr),
        # based on the specific graph rules of the dataset
        graph_validity_score = self.evaluate_graph_validity(graph_state)
        reward = (intermediate_reward +
                  self.lambda_1 * final_graph_reward +
                  self.lambda_2 * graph_validity_score)
        return reward

    def train_generative_model(self, model_to_explain, for_class):
        """Trains the generative model for the given number of epochs. We us
        RL approach to train the generative model.

        Args:
            model_to_explain (_type_): The model to explain.
            for_class (_type_): The class to explain.

        Returns:
            torch_geometric.data.Data: The trained generative model.
        """
        optimizer = torch.optim.Adam(self.graph_generator.parameters(),
                                     lr=self.lr, betas=(0.9, 0.99))
        losses = []
        for epoch in trange(self.epochs):

            # create empty graph state
            empty_graph = Data(x=torch.tensor([]), edge_index=torch.tensor([]),
                               node_type=[])
            current_graph_state = empty_graph

            for step in range(self.max_steps):
                model.train()
                optimizer.zero_grad()

                new_graph_state = copy.deepcopy(current_graph_state)
                ((p_start, a_start), (p_end, a_end)
                 ), new_graph_state = self.graph_generator(new_graph_state)
                reward = self.calculate_reward(new_graph_state,
                                               model_to_explain, for_class,
                                               self.num_classes)

                LCE_start = F.cross_entropy(p_start, a_start)
                LCE_end = F.cross_entropy(p_end, a_end)
                loss = -reward * (LCE_start + LCE_end)

                loss.backward()
                optimizer.step()

                if reward > 0:
                    current_graph_state = new_graph_state

            losses.append(loss.item())

        return self.graph_generator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model

args = {
    'device': device,
    'dropout': 0.1,
    'epochs': 1000,
    'input_dim': 7,
    'opt': 'adam',
    'opt_scheduler': 'none',
    'opt_restart': 0,
    'weight_decay': 5e-5,
    'lr': 0.001
}


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


args = objectview(args)

model = GCN_Graph(args.input_dim, output_dim=1,
                  dropout=args.dropout).to(device)

# Assume 'model_to_freeze' is the model you want to freeze
for param in model.parameters():
    param.requires_grad = False

# depending on os change path
path = "examples/explain/xgnn/mutag_model.pth"
if os.name == 'nt':
    path = "examples\\explain\\xgnn\\mutag_model.pth"

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.to(device)

# Train generative model

# extract features for the candidate set
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
all_features = torch.cat([data.x for data in dataset], dim=0)

# graph validity check for number of edges depending on atom
max_valency = {'C': 4, 'N': 5, 'O': 2, 'F': 1, 'I': 7, 'Cl': 7, 'Br': 5}

# node type map that maps node type to a one hot vector encoding torch tensor
candidate_set = {
    'C': torch.tensor([1, 0, 0, 0, 0, 0, 0]),
    'N': torch.tensor([0, 1, 0, 0, 0, 0, 0]),
    'O': torch.tensor([0, 0, 1, 0, 0, 0, 0]),
    'F': torch.tensor([0, 0, 0, 1, 0, 0, 0]),
    'I': torch.tensor([0, 0, 0, 0, 1, 0, 0]),
    'Cl': torch.tensor([0, 0, 0, 0, 0, 1, 0]),
    'Br': torch.tensor([0, 0, 0, 0, 0, 0, 1])
}

explainer = Explainer(
    model=model,
    algorithm=RLGenExplainer(epochs=100, lr=0.01, candidate_set=candidate_set,
                             validity_args=max_valency, initial_node_type='C'),
    explanation_type='generative',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='probs',
    ),
)

# choose target class
class_index = 1

# empty x and edge_index tensors, since we are not explaining one
# specific graph, but existing explainer requires these
x = torch.tensor([])
edge_index = torch.tensor([[], []])

explanation_mutagenic = explainer(x, edge_index, for_class=class_index)
explanation_set_mutagenic = explanation_mutagenic.explanation_set

#######################
# SAMPLE SINGLE GRAPH #
#######################
sampled_graph = explanation_set_mutagenic.sample(num_samples=1,
                                                 num_nodes=10)[0]
# visualize sampled graph with DEFAULT inbuild method
explanation_mutagenic.visualize_explanation_graph(
    sampled_graph, path='examples/explain/xgnn/sample_graph',
    backend='networkx')

##########################
# SAMPLE MULTIPLE GRAPHS #
##########################
node_color_dict = {
    'C': '#0173B2',
    'N': '#DE8F05',
    'O': '#029E73',
    'F': '#D55E00',
    'I': '#CC78BC',
    'Cl': '#CA9161',
    'Br': '#FBAFE4'
}  # colorblind palette

# generate graphs of multiple sizes for mutagenic class

sampled_graphs = []
scores = []
for i in range(3, 11):
    sampled_graphs_i = explanation_set_mutagenic.sample(
        num_samples=5, num_nodes=i)
    scores_i = []
    for sampled_graph in sampled_graphs_i:
        score = model(sampled_graph)
        probability_score = torch.sigmoid(score)
        scores_i.append(probability_score.item())

    # choose graph with best score
    best_graph_index = scores_i.index(max(scores_i))
    scores.append(scores_i[best_graph_index])
    sampled_graphs.append(sampled_graphs_i[best_graph_index])

# visualize sampled graphs with CUSTOM method
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (sampled_graph, score) in enumerate(zip(sampled_graphs, scores)):
    G = to_networkx(sampled_graph, to_undirected=True)
    node_type_dict = dict(enumerate(sampled_graph.node_type))
    nx.set_node_attributes(G, node_type_dict, 'node_type')
    labels = nx.get_node_attributes(G, 'node_type')
    node_color = [
        node_color_dict[key]
        for key in nx.get_node_attributes(G, 'node_type').values()
    ]
    pos = nx.spring_layout(G)
    axes[i].set_title(
        "Max_num_nodes = {}\nprobability = {:.5f}".format(i + 3, score),
        loc="center", fontsize=10)
    # set subtitle to score
    axes
    nx.draw(G, pos=pos, ax=axes[i], cmap=plt.get_cmap('coolwarm'),
            node_color=node_color, labels=labels, font_color='white')

# plt.savefig('examples/explain/xgnn/sample_graphs_custom.png')
fig.suptitle("Sampled explanation graphs for mutagenic class", fontsize=16)
plt.show()

plt.close()

# generate graphs of multiple sizes for non-mutagenic class
class_index = 0

explanation_non_mutagenic = explainer(x, edge_index, for_class=class_index)
explanation_set_non_mutagenic = explanation_non_mutagenic.explanation_set

sampled_graphs = []
scores = []
for i in range(3, 11):
    sampled_graphs_i = explanation_set_non_mutagenic.sample(
        num_samples=5, num_nodes=i)
    scores_i = []
    for sampled_graph in sampled_graphs_i:
        score = model(sampled_graph)
        probability_score = 1 - torch.sigmoid(score)
        scores_i.append(probability_score.item())

    # choose graph with best score
    best_graph_index = scores_i.index(max(scores_i))
    scores.append(scores_i[best_graph_index])
    sampled_graphs.append(sampled_graphs_i[best_graph_index])

# visualize sampled graphs with CUSTOM method
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (sampled_graph, score) in enumerate(zip(sampled_graphs, scores)):
    G = to_networkx(sampled_graph, to_undirected=True)
    node_type_dict = dict(enumerate(sampled_graph.node_type))
    nx.set_node_attributes(G, node_type_dict, 'node_type')
    labels = nx.get_node_attributes(G, 'node_type')
    node_color = [
        node_color_dict[key]
        for key in nx.get_node_attributes(G, 'node_type').values()
    ]
    pos = nx.spring_layout(G)
    axes[i].set_title(
        "Max_num_nodes = {}\nprobability = {:.5f}".format(i + 3, score),
        loc="center", fontsize=10)
    # plot graph with scroe as title
    nx.draw(G, pos=pos, ax=axes[i], cmap=plt.get_cmap('coolwarm'),
            node_color=node_color, labels=labels, font_color='white')

# plt.savefig('examples/explain/xgnn/sample_graphs_custom.png')
fig.suptitle("Sampled explanation graphs for non-mutagenic class", fontsize=16)
plt.show()
