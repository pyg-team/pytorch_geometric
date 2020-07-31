"""
License
Copyright 2018 The GraphNets Authors. All Rights Reserved. Licensed under the
Apache License, Version 2.0 (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
 specific language governing permissions and limitations under the License.


Based on the arXiv paper: Relational inductive biases, deep learning, and graph networks.

This code proposes to deal with the problem of learning a physical model using
graphs and the torch geometric library. This work is based on the work done by
the GraphNets authors. The idea is to see how to use a pytorch based library
to achieve the same result as with the GraphNets library. Some simplifications
have been made compared to the approach proposed in the code using GraphNets.

The goal of the network is to learn to predict the motion of a set of masses
connected by springs. The network is trained to predict the behaviour of a chain
of 5 to 8 masses connected by springs. The first and last masses are fixed,
otherwise the other masses are subject to gravity and Hook's Law.

To control the ability of the network to learn the physical dynamics of the
structure, we compare its output to the true behaviour of the structure. It is
 possible to test the generalizability of the network by using structures with
 more or less masses (example: 4 and 9 masses).
"""

import torch

from torch_geometric.data import Data
from torch_scatter import scatter
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import os

"""
Data management:

Creation of a network with the following attributes (in the form of a tensor):

- nodes:  [𝑥,𝑦,𝑣𝑥,𝑣𝑦,𝑖𝑠_𝑓𝑖𝑥𝑒𝑑] , position and speed of each mass.
- edges:  [𝑘,𝑥𝑟𝑒𝑠𝑡] , string stiffness constant and rest position (for each segment).
- u (global):  [0,−10] , gravity coordinates.

In edge_index, the number of sender and receiver nodes is indicated.
"""
def create_graph(n,d, verbose = False):
    """Define a basic mass-spring system graph structure.

    n masses (1kg) connected by spring. The first and last are fixed. Masses are
    aligned at the begining, and d meters appart (d = rest length for the spring).
    Spring constant: 50N/m. Gravity: 10N in the negative y direction.

    Parameters:
    -----------
        n: int
            Number of masses.
        d: float
            Distance between masses.
        verbose: bool
            Show information about the data.

    Returns:
    --------
        data, the corresponding graph.
    """
    nodes = np.zeros((n,5), dtype=np.float32)
    half_width = d*n/2.0
    nodes[:,0]= np.linspace(-half_width, half_width, num=n, endpoint=False, dtype = np.float32)
    nodes[(0,-1), -1]= 1.

    edge_index = []
    edge_attr = []

    for i in range(n-1):
        left_node = i
        right_node = i+1
        if right_node < n-1:
            edge_attr.append([50., d])
            edge_index.append([left_node, right_node])
        if left_node > 0:
            edge_attr.append([50., d])
            edge_index.append([right_node, left_node])

    nodes = torch.tensor(nodes)
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(edge_index).t().contiguous()
    u = torch.tensor([0., -10.])

    data = Data(x= nodes, edge_index=edge_index, edge_attr= edge_attr, u = u)

    if verbose:
        print("\n---- Verbose ----")
        print("\n>    Statistic")
        print(">>   num nodes: ", data.num_nodes)
        print(">>   num nodes features: ", data.num_node_features)
        print(">>   num edges: ", data.num_edges)
        print("\n>    Data")
        print(">>   nodes: ", data['x'])
        print(">>   edges: ", data['edge_index'])
        print(">>   edges features: ", data['edge_attr'])

    return data

"""
Real simulator:

Torch equivalent of the simulator proposed by the authors of GraphNets.

Several elements:

- hookes_law: Apply Hooke's law to springs connecting some nodes.
- euler_integration: Apply one step of Euler integration. Based on:
𝑑𝑣/𝑑𝑡=𝑎=𝐹/𝑚

and
𝑑𝑥/𝑑𝑡=𝑣

- SpringMassSimulator (class): physic simulator.
- roll_out_physics: Apply some number of steps of physical laws to an interaction network.
- apply_noise: Apply uniformly-distributed noise to a graph of a physical system.
- set_rest_lengths: Computes and sets rest lengths for the springs in a physical system.
- generate_trajectory: Apply noise and then simulates a physical system for a number of steps.
- create_loss: Create supervised loss from target and output.
"""

def hookes_law(receiver_nodes, sender_nodes, k, x_rest):
    """Applies Hooke's law to springs connecting some nodes.

    Parameters:
    -----------
        receiver_nodes: num nodes x num nodes features, tensor
            [x, y, v_x, v_y, is_fixed] features for the receiver node of each
            edge.
        sender_nodes: num nodes x num nodes features, tensor
            [x, y, v_x, v_y, is_fixed] features for the sender node of each
            edge.
        k: tensor
            Spring constant for each edge.
        x_rest: tensor
            Rest length of each edge

    Note:
    -----
        receiver = data['edge_index'][1]
        receiver_nodes = data['x'][receiver]
    """

    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    x = torch.norm(diff, dim=-1, keepdim=True)
    force_magnitude = -1 * k*((x-x_rest)/x)
    force = force_magnitude * diff

    return force

def euler_integration(nodes, force_per_node, step_size):
    """Applies one step of Euler integration.

    Parameters:
    -----------
        nodes: num nodes x num nodes features, tensor
            [x, y, v_x, v_y, is_fixed] features for each node.
        force_per_node: num nodes x 2, tensor
            [f_x, f_y] acting on each edge.
        step_size: float

    Returns:
    --------
        Nodes with positions and velocities updated.
    """
    is_fixed = nodes[..., 4:5].to(torch.float64)
    #force = 0 for fixed nodes
    force_per_node*= 1- is_fixed
    new_vel = (nodes[..., 2:4] + force_per_node*step_size).to(torch.float64)
    new_pos = (nodes[..., :2] +  new_vel*step_size).to(torch.float64)

    update = torch.cat([new_pos, new_vel, is_fixed], axis = -1)
    return update

class SpringMassSimulator(nn.Module):
    """Implements a basic Physics Simulator."""

    def __init__(self, step_size, name="SpringMassSimulator"):
        super(SpringMassSimulator, self).__init__()
        self._step_size = step_size

    def _agregator(self, edges, indices, num_nodes):
        return scatter(edges.t(), indices, dim=-1, dim_size = num_nodes, reduce = "sum").t()

    def forward(self, graph):
        """Build a SpringMassSimulator.

        Parameters:
        -----------
            graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

        Returns:
        --------
            A graph of the same shape as graph, but with:
                - nodes: holds positions and velocities after applying one step of Euler integration.

        """
        receiver = graph['edge_index'][1]
        sender = graph['edge_index'][0]
        receiver_nodes = graph['x'][receiver]
        sender_nodes = graph['x'][sender]

        spring_force_per_edge = hookes_law(receiver_nodes, sender_nodes,
                                                graph['edge_attr'][...,0:1],
                                                graph['edge_attr'][...,1:2])

        spring_force_per_node = self._agregator(spring_force_per_edge, receiver, graph.num_nodes)
        gravity = graph['u'].repeat((graph.num_nodes,1))
        updated_velocities = euler_integration(graph['x'], spring_force_per_node + gravity, self._step_size)
        graph['x'] = updated_velocities
        return graph

def roll_out_physics(simulator, graph, steps, step_size):
    """Apply some number of steps of physical laws to an interaction network.

    Parameters:
    -----------
        simulator: SpringMassSimulator
        graph: a graph with for some N (number of nodes), E (number of edges):
            - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
            - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
            - u: tensor containing the gravitational constant.
        steps: int
        step_size: float

    Returns:
    --------
        A pair of:
            - the graph updated after steps steps of simulation.
            - list of N x 5 tensor of the node features at each step.
    """

    features_node = [graph['x']]
    for _ in range(steps):
        graph = simulator.forward(graph)
        features_node.append(graph['x'])

    return graph, features_node

def apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level):
    """Applies uniformly-distributed noise to a graph of a physical system.

    Noise is applied to:
        - the x and y coordinates (independently) of the nodes;
        - the spring constants of the edges;
        - the y coordinates of the global gravitational constant.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
            - edges: N x 2 tensor of [spring_constant, _] for each edge.
            - nodes: E x 5 tensor of [x, y, _, _, _] features for each nodes.
            - u: tensor containing the gravitational constant.
        node_noise_level:
            Maximum distance to perturb nodes' x and y coordinates.
        edge_noise_level:
            Maximum amount to perturb edge spring constants.
        global_noise_level:
            Maximum amount to perturb the Y component of gravity.

    Returns:
    --------
        The input graph, but with noise applied.
    """
    r_1_node, r_2_node = -node_noise_level, node_noise_level
    r_1_edge, r_2_edge = -edge_noise_level, edge_noise_level
    r_1_global, r_2_global = -global_noise_level, global_noise_level

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    node_position_noise = (r_1_node - r_2_node) * torch.rand(num_nodes, 2) + r_2_node
    edge_spring_constant_noise = (r_1_edge - r_2_edge) * torch.rand(num_edges, 1) + r_2_edge
    global_gravity_y_noise = (r_1_global - r_2_global) * torch.rand(1) + r_2_global

    graph['x']= torch.cat((graph['x'][...,:2]+node_position_noise, graph['x'][...,2:]), axis=-1)
    graph['edge_attr'] = torch.cat((graph['edge_attr'][...,:1]+edge_spring_constant_noise, graph['edge_attr'][...,1:]), axis=-1)
    graph['u'] = torch.cat((graph['u'][...,:1], graph['u'][...,1:] + global_gravity_y_noise), axis=-1)

    return graph

def set_rest_lengths(graph):
    """Computes and sets rest lengths for the springs in a physical system.

    The rest length is taken to be the distance between each edge's nodes.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
            - edges: N x 2 tensor of [spring_constant, _] for each edge.
            - nodes: E x 5 tensor of [x, y, _, _, _] features for each nodes.

    Returns:
    --------
        The input graph, but with [spring_constant, rest_length] for each edge.
    """
    receiver = graph['edge_index'][1]
    sender = graph['edge_index'][0]
    receiver_nodes = graph['x'][receiver]
    sender_nodes = graph['x'][sender]

    diff = receiver_nodes[...,:2] - sender_nodes[...,:2]
    rest_length = torch.norm(diff, dim=-1, keepdim=True).to(torch.float64)

    graph['edge_attr'] = torch.cat((graph['edge_attr'][...,:1].to(torch.float64), rest_length), axis=-1)

    return graph

def generate_trajectory(simulator, graph, steps, step_size, node_noise_level,
                        edge_noise_level, global_noise_level):
        """Applies noise and then simulates a physical system for a number of steps.

        Parameters:
        -----------
            simulator: SpringMassSimulator
            graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.
            steps: int
            step_size: float
            node_noise_level:
                Maximum distance to perturb nodes' x and y coordinates.
            edge_noise_level:
                Maximum amount to perturb edge spring constants.
            global_noise_level:
                Maximum amount to perturb the Y component of gravity.

        Returns:
        --------
            A pair of:
                - the input graph with rest lengths computed and noise applied.
                - list of N x 5 tensor of the node features at each step.
        """
        graph = apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level)
        graph = set_rest_lengths(graph)
        _, features_node = roll_out_physics(simulator, graph, steps, step_size)
        return graph, features_node

def create_loss(target, output):
    """Create supervised loss from target and output.

    Parameters:
    -----------
        target: tensor
            The target nodes.
        output: graph
            Output graph from the model.

    Returns:
    --------
        Loss value (tensor).
    """
    return torch.sum((target-output["x"])**2)

"""
Blocks for Graph Networks:

From GraphNets library. Several elements:

- broadcast_1_to_2: propagates the features from 1 to 2.
- 1To2Aggregator: propagates and reduces the features from 1 to 2.
- EdgeBlock, NodeBlock and GlobalBlock: elementary graph networks that only
update the edges (resp. the nodes, the globals) of their input graph.
"""
def broadcast_globals_to_edges(graph):
    """Propagates the features from globals to edges.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        The globals propagate to edges.
    """
    return graph['u'].repeat((graph.num_edges,1))

def broadcast_globals_to_nodes(graph):
    """Propagates the features from globals to node.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        The globals propagate to nodes.
    """
    return graph['u'].repeat((graph.num_nodes,1))

def broadcast_sender_nodes_to_edges(graph):
    """Propagates the features from sender nodes to edges.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        The sender nodes propagate to edges.
    """
    sender = graph['edge_index'][0]
    sender_nodes = graph['x'][sender]
    return sender_nodes

def broadcast_receiver_nodes_to_edges(graph):
    """Propagates the features from receiver nodes to edges.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        The receiver nodes propagate to edges.
    """
    receiver = graph['edge_index'][1]
    receiver_nodes = graph['x'][receiver]
    return receiver_nodes

def EdgesToGlobalsAggregator(graph):
    """Aggregates the features from edges nodes to globals.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        Sum all the edge attributes.
    """
    return torch.sum(graph["edge_attr"], dim=0)

def NodesToGlobalsAggregator(graph):
    """Aggregates the features from nodes nodes to globals.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        Sum all the nodes.
    """
    return torch.sum(graph["x"], dim=0)

def SentEdgesToNodesAggregator(graph):
    """Aggregates the features from sender edges nodes to nodes.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        Sum all the edges attributes related to sender nodes.
    """
    sender_idx = graph['edge_index'][0]
    return scatter(graph["edge_attr"].t(), sender_idx, dim=-1, dim_size = graph.num_nodes).t()

def ReceivedEdgesToNodesAggregator(graph):
    """Aggregates the features from receiver edges nodes to nodes.

    Parameters:
    -----------
        graph: a graph with for some N (number of nodes), E (number of edges):
                - edges: N x 2 tensor of [spring_constant, rest_length] for each edge.
                - nodes: E x 5 tensor of [x, y, v_x, v_y, is_fixed] features for each nodes.
                - u: tensor containing the gravitational constant.

    Returns:
    --------
        Sum all the edges attributes related to receiver nodes.
    """
    receiver_idx = graph['edge_index'][1]
    return scatter(graph["edge_attr"].t(), receiver_idx, dim=-1, dim_size = graph.num_nodes).t()

class EdgeBlock(nn.Module):
    """Edge block.

    A block that updates the features of each edge in a graph based on
    (a subset of) the previous edge features, the features of the adjacent nodes,
    and the global features of the corresponding graph.

    """
    def __init__(self, edge_model_fn, use_edges=True, use_receiver_nodes = True,
                use_sender_nodes = True, use_globals = True):
        """Initialization.

        Parameters:
        -----------
            edge_model_fn: callable (Pytorch module)
                Calablde to be used as the edge model. This module should take a
                tensor (of concatenated input features for each edge) and return
                a tensor (of output features for each edge).
            use_edges: bool, default = True
                Wheteher to condition on edge attributes.
            use_receiver_nodes: bool, default = True
                Wheter to condition on receiver node attributes.
            use_sender_nodes: bool, default = True
                Whether to condition on sender node attributes.
            use_globals: bool, default = True
                Whether to condition on global attributes.

        Raises:
        -------
            ValueError: When fields that are required are missing.
        """
        super(EdgeBlock, self).__init__()

        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                            "use_receiver_nodes or use_globals must be True")

        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

        self._edge_model = edge_model_fn

    def forward(self, graph):
        """Forward method.

        Parameters:
        -----------
            graph: a graph containing
                - individual edges features if use_edges = True
                - individual nodes features if use_receiver_nodes | use_sender_nodes = True
                - globals if use_globals = True
                Warning: these elements should be concatenable on the last axis.

        Returns:
        --------
            edges: tensor
                The updated edges.
        """
        edges_to_collect = []

        if self._use_edges:
            edges_to_collect.append(graph["edge_attr"])
        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))
        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))
        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(graph))

        collected_edges = torch.cat(edges_to_collect, axis = -1)
        updated_edges = self._edge_model(collected_edges)

        graph["edge_attr"] = updated_edges
        return graph

class NodeBlock(nn.Module):
    """Node block.

    A block that updates the features of each node in gaph based on (a subset of)
    the previous node features, the aggregated features of the adjacent edges, and
    the global features of the corresponding graph.
    """
    def __init__(self, node_model_fn, use_received_edges = True, use_sent_edges = False,
                use_nodes = True, use_globals = True):
        """Initialization

        Parameters:
        -----------
            node_model_fn: callable (Pytorch module)
                Callable to be used as the node model. This module should take a
                tensor (of concatenated input features for each node) and return
                a tensor (of output features for each node).
            use_received_edges: bool, default = True
                Whether to condition on aggregated edges received by each node.
            use_sent_edges: bool, default = False
                Whether to condition on aggregated edges sent by each node.
            use_nodes: bool, default = True
                Whether to condition on node attributes.
            use_globals: bool, default = True
                Whether to condition on global attributes.

        Raises:
        -------
            ValueError: When fields that are required are missing.
        """
        super(NodeBlock, self).__init__()

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
                raise ValueError("At least one of use_received_edges, use_sent_edges, "
                       "use_nodes or use_globals must be True.")

        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        self._node_model = node_model_fn
        if self._use_received_edges:
            self._received_edges_aggregator = ReceivedEdgesToNodesAggregator
        if self._use_sent_edges:
            self._sent_edges_aggregator = SentEdgesToNodesAggregator

    def forward(self, graph):
        """Forward method.

        Parameters:
        -----------
            graph: a graph containing
                - individual edges features if use_edges = True
                - individual nodes features if use_receiver_nodes | use_sender_nodes = True
                - globals if use_globals = True
                Warning: these elements should be concatenable on the last axis.

        Returns:
        --------
            edges: tensor
                The updated edges.
        """
        nodes_to_collect = []

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))
        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))
        if self._use_nodes:
            nodes_to_collect.append(graph['x'])
        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(graph))

        collected_nodes = torch.cat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)

        graph["x"] = updated_nodes
        return graph

class GlobalBlock(nn.Module):
    """Global block.

    A block that updates the global features of a graph based on (a subset of)
    the previous global features, the aggregated features of the edges of the graph
    and the aggregated features of the nodes of the graph.
    """

    def __init__(self, global_model_fn, use_edges=True, use_nodes = True, use_globals = True):
        """Initialization.

        Parameters:
        -----------
            global_model_fn: callable (Pytorch module)
                Callable to be used as the global model. This module should take a
                tensor (of concatenated input features for each node) and return
                a tensor (of output features for each node).
            use_edges: bool, default = True
                Whether to condition on aggregated edges.
            use_nodes: bool, default = True
                Whether to condition on node attributes.
            use_globals: bool, default = True
                Whether to condition on global attributes.

        Raises:
        -------
            ValueError: When fiels that are required are missing.
        """

        super(GlobalBlock, self).__init__()
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                       "use_nodes or use_globals must be True.")

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        self._global_model = global_model_fn
        if self._use_edges:
            self._edges_aggregator = EdgesToGlobalsAggregator
        if self._use_nodes:
            self._nodes_aggregator = NodesToGlobalsAggregator

    def forward(self, graph):
        """Forward method.

        Parameters:
        -----------
            graph: a graph containing
                - individual edges features if use_edges = True
                - individual nodes features if use_nodes = True
                - globals if use_globals = True
                Warning: these elements should be concatenable on the last axis.

        Returns:
        --------
            edges: tensor
                The updated edges.
        """
        globals_to_collect = []
        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph))
        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph))
        if self._use_globals:
            globals_to_collect.append(graph['u'])

        collected_globals = torch.cat(globals_to_collect, axis=-1)
        updated_globals = self._global_model(collected_globals)

        graph['u'] = updated_globals
        return graph

"""Create the Networks
"""

#Global variables

LATENT_SIZE = 64
NODE_SIZE = 5
EDGE_SIZE = 2
GLOBAL_SIZE = 2

class MLP(nn.Module):
    """Classical MLP.
    """
    def __init__(self, output_size, latent_size):
        """Initialization.

        Parameters:
        -----------
            output_size: int
                The size of the output.
            latent_size: int
                The size of all the hidden layers.
        """
        super(MLP, self).__init__()
        self.latent_size = latent_size
        self.output_size = output_size
        self.initialized = False

    def _initialize_network(self, x):
        """Initialize the layers of the network with the good input shape.

        Parameters:
        -----------
            x: tensor
                The input of the netwok.
        """
        if len(x.size())>1:
            input_size = x.size()[1]
        else:
            input_size = x.size()[0]
        # Hidden layers
        self.lin1 = nn.Linear(input_size, self.latent_size)
        self.lin2 = nn.Linear(self.latent_size,self.latent_size)
        # Output layer
        self.out = nn.Linear(self.latent_size, self.output_size)

        self.initialized = True


    def forward(self, x):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            x: tensor
                The input of the netwok.

        Returns:
        --------
            The output of the network.
        """
        if not self.initialized:
            self._initialize_network(x)

        x = F.relu(self.lin1(x.float()))
        x = F.relu(self.lin2(x))
        x = self.out(x)

        return x

class GraphIndependent(nn.Module):
    """A graph bloc that applies models to the graph elements independently.
    """
    def __init__(self, encoder = True):
        """Initialization.

        Parameters:
        -----------
            encoder: bool
                Choose the kind of network we build.
        """
        super(GraphIndependent,self).__init__()
        if encoder:
            self._edge_mlp = MLP(LATENT_SIZE, LATENT_SIZE)
            self._node_mlp = MLP(LATENT_SIZE, LATENT_SIZE)
            self._global_mlp = MLP(LATENT_SIZE, LATENT_SIZE)
        else:
            self._edge_mlp = MLP(EDGE_SIZE, LATENT_SIZE)
            self._node_mlp = MLP(NODE_SIZE, LATENT_SIZE)
            self._global_mlp = MLP(GLOBAL_SIZE, LATENT_SIZE)

    def forward(self, graph):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            graph:
                The input of the netwok.

        Returns:
        --------
            The output of the network.
        """
        graph['x'] = self._node_mlp(graph['x'])
        graph['edge_attr'] = self._edge_mlp(graph['edge_attr'])
        graph['u'] = self._global_mlp(graph['u'])

        return graph

class GraphNetwork(nn.Module):
    """Implementation of a Graph Network.
    """

    def __init__(self):
        """Initialization.
        """
        super(GraphNetwork, self).__init__()
        self._edge_mlp = MLP(LATENT_SIZE, LATENT_SIZE)
        self._node_mlp = MLP(LATENT_SIZE, LATENT_SIZE)
        self._global_mlp = MLP(LATENT_SIZE, LATENT_SIZE)

        self._edge_block = EdgeBlock(self._edge_mlp, use_edges=True, use_receiver_nodes = False,
                        use_sender_nodes = False, use_globals = False)
        self._node_block = NodeBlock(self._node_mlp, use_received_edges = False, use_sent_edges = False,
                    use_nodes = True, use_globals = False)
        self._global_block = GlobalBlock(self._global_mlp, use_edges=True, use_nodes = True, use_globals = True)

    def forward(self, graph):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            graph:
                The input of the netwok.

        Returns:
        --------
            The output of the network.
        """
        return self._global_block(self._node_block(self._edge_block(graph)))

class EncodeProcessDecode(nn.Module):
    """Full encode-process-decode model.

    The model contains three components:
    - An "Encoder" graph net, which independently encodes the edges, node, and
    global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, wher "t" is
    the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node and global
    attributes (does not compute relations etc.), on each message-passing step.

                          Hidden(t)   Hidden(t+1)
                             |            ^
                *---------*  |  *------*  |  *---------*
                |         |  |  |      |  |  |         |
      Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
                |         |---->|      |     |         |
                *---------*     *------*     *---------*
    """
    def __init__(self):
        """Initialization.
        """
        super(EncodeProcessDecode, self).__init__()
        self._encoder = GraphIndependent()
        self._core = GraphNetwork()
        self._decoder = GraphIndependent(False)

    def forward(self, graph, num_processing_steps):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            graph:
                The input of the netwok.
            num_processing_steps: int
                Number of iterations in the loop.

        Returns:
        --------
            The output of the network.
        """
        input_op = graph.clone()
        latent = self._encoder(input_op)
        latent0 = latent
        #outputs = []

        for _ in range(num_processing_steps):
            idx = latent['edge_index']
            cat_x = torch.cat([latent0['x'], latent['x']], axis=-1)
            cat_idx = torch.cat([latent0['edge_index'], idx+latent0['edge_index'].size()[0]], axis=-1)
            cat_edge = torch.cat([latent0['edge_attr'], latent['edge_attr']], axis=-1)
            cat_global = torch.cat([latent0['u'], latent['u']], axis=-1)
            core_input = Data(x= cat_x, edge_index=cat_idx, edge_attr= cat_edge, u = cat_global)
            latent = self._core(core_input)
            decoded = self._decoder(latent.clone())
            decoded['edge_attr'] = idx

            #outputs.append(decoded)

        return decoded

"""Plot function
"""

def plot_springs(target_nodes, output_node, path, name):
    """Plots the springs (target and output).

    Parameters:
    -----------
        target_nodes: tensor
            The real spring.
        output_node: tensor
            The prediction of the network.
        path: string
            Path to save the figure.
        name: string
            Name of the figure.
    """
    os.makedirs(path, exist_ok=True)
    try:
        x_target = target_nodes[...,0].t().numpy()
        y_target = target_nodes[...,1].t().numpy()
        x_output = output_node[...,0].t().numpy()
        y_output = output_node[...,1].t().numpy()
    except:
        x_target = target_nodes[...,0].t().detach().numpy()
        y_target = target_nodes[...,1].t().detach().numpy()
        x_output = output_node[...,0].t().detach().numpy()
        y_output = output_node[...,1].t().detach().numpy()
    plt.figure()
    plt.plot(x_target, y_target, marker='o', label='Target')
    plt.plot(x_output, y_output, marker='+', label='Output')
    plt.legend()
    plt.title("Position")
    plt.savefig(path + '/' + name + '.png')
    plt.close()

"""Set up and generate data
"""

# Model parameters.
num_processing_steps_tr = 1
num_processing_steps_ge = 1

# Data / training parameters.
batch_size_tr = 256
batch_size_ge = 100
num_time_steps = 50
step_size = 0.1
num_masses_min_max_tr = (5, 9)
dist_between_masses_min_max_tr = (0.2, 1.0)

#create the model and the tru physics simulator for data generation
model = EncodeProcessDecode()
simulator = SpringMassSimulator(step_size=step_size)

#Base graphs for training
num_masses_tr = np.random.randint(*num_masses_min_max_tr, size=batch_size_tr)
dist_between_masses_tr = np.random.uniform(*dist_between_masses_min_max_tr, size= batch_size_tr)
static_graph_tr = [create_graph(n,d) for n,d in zip(num_masses_tr, dist_between_masses_tr)]

#Base graphs for testing: 4 and 9 nodes
create_graph_4_ge = [create_graph(4,0.5)]*batch_size_ge
create_graph_9_ge = [create_graph(9,0.5)]*batch_size_ge

#Generate a training trajectory by adding noise to initial position, spring constants and gravity
print(">    Generate true trajectories - train ...")
initial_conditions, true_trajectory_tr_s = [], []
for elt in static_graph_tr:
    initial_condition, true_trajectory_tr = generate_trajectory(
        simulator,
        elt,
        num_time_steps,
        step_size,
        node_noise_level=0.04,
        edge_noise_level=5.0,
        global_noise_level=1.0)
    initial_conditions.append(initial_condition)
    true_trajectory_tr_s.append(true_trajectory_tr)
print(">    ... End.")

print(">    Generate true trajectories - validation ...")
val_initial_conditions, val_true_trajectory_tr_s = [], []
for elt in create_graph_4_ge:
    val_initial_condition, val_true_trajectory_tr = generate_trajectory(
        simulator,
        elt,
        num_time_steps,
        step_size,
        node_noise_level=0,
        edge_noise_level=0,
        global_noise_level=0)
    val_initial_conditions.append(initial_condition)
    val_true_trajectory_tr_s.append(true_trajectory_tr)

for elt in create_graph_9_ge:
    val_initial_condition, val_true_trajectory_tr = generate_trajectory(
        simulator,
        elt,
        num_time_steps,
        step_size,
        node_noise_level=0,
        edge_noise_level=0,
        global_noise_level=0)
    val_initial_conditions.append(initial_condition)
    val_true_trajectory_tr_s.append(true_trajectory_tr)
print(">    ... End.")

#Optimizer
learning_rate = 1e-4
#init layers with good input
print(">    Init optimizer ...")
model(initial_condition, num_processing_steps_tr)
optimizer = optim.Adam(model.parameters(), lr= learning_rate)
print(">    ... End.")

"""Train and validation"""

print(">    Train the network ...")
epochs = 3000
freq_val = 5

train_loss = []
val_loss = []
train_epoch = []
val_epoch = []

for e in range(epochs):
    model.train()
    print(">>   Epoch {}/{}".format(e, epochs))
    tot_loss = 0
    for initial_condition, true_trajectory in zip(initial_conditions, true_trajectory_tr_s):
        t = np.random.randint(0, num_time_steps-num_processing_steps_tr)
        initial_condition['x'] = true_trajectory[t]
        target_nodes = true_trajectory[t+num_processing_steps_tr]

        output = model(initial_condition, num_processing_steps_tr)
        loss = create_loss(target_nodes, output)
        tot_loss+= loss

    plot_springs(target_nodes, output['x'],  "./Results/images", 'train_{}'.format(e))
    loss = tot_loss/batch_size_tr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss)
    train_epoch.append(e)
    print(">>   Train loss: {:.4f}".format(loss))

    if e%freq_val == 0:
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            for initial_condition, true_trajectory in zip(val_initial_conditions, val_true_trajectory_tr_s):
                t = np.random.randint(0, num_time_steps-num_processing_steps_tr)
                initial_condition['x'] = true_trajectory[t]
                target_nodes = true_trajectory[t+num_processing_steps_tr]

                output = model(initial_condition, num_processing_steps_tr)
                loss = create_loss(target_nodes, output)
                tot_loss+= loss
            plot_springs(target_nodes, output['x'],  "./Results/images/", 'val_{}'.format(e))
            loss = tot_loss/(2*batch_size_ge)

        val_loss.append(loss)
        val_epoch.append(e)
        print(">>   Validation loss: {:.4f}".format(loss))



plt.figure()
plt.plot(val_epoch, val_loss, label='Validation')
plt.plot(train_epoch, train_loss, label='Train')
plt.legend()
plt.title("Loss")
plt.savefig('./Results/loss.png')

print(">    ... End.")
