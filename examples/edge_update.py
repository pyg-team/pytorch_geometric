from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


class EdgeMessagePassingExample(MessagePassing):

    def forward(self, data: Data):
        x = self.propagate(data.edge_index, data.x)
        edge_attr = self.update_edges(data.edge_index, data.x, data.edge_attr)


    def edge_update(self, x_i, x_j, edge_features):
        return edge_features


    def message(self, x_i):
        return x_i