from rich.console import Console
from rich.table import Table
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import GDELTLite, Planetoid
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn.models.graph_mixer import LinkEncoder, NodeEncoder
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

HIDE_TENSOR = False


def _describe_data(
    data,
    attribute_dict,
    title="PyG Data Description",
    color="green",
):
    table = Table(title=title)
    table.add_column("name", justify="left", style=color, no_wrap=True)
    table.add_column("size", justify="left", style=color)
    table.add_column("known size", justify="left", style=color)
    table.add_column("type", justify="left", style=color)
    table.add_column("content", justify="left", style=color)

    for attr_name, attr_known_shape in attribute_dict.items():
        attr = getattr(data, attr_name, None)
        if attr is None:
            continue

        if isinstance(attr, torch.Tensor):
            size_str = str(list(attr.shape))
        elif isinstance(attr, list):
            size_str = str([len(attr)])
        elif isinstance(attr, int):
            size_str = "1"
        else:
            size_str = ""

        attr_str = str(attr)
        if HIDE_TENSOR and isinstance(attr, torch.Tensor):
            attr_str = "-"

        table.add_row(
            attr_name,
            size_str,
            "\\" + attr_known_shape if attr_known_shape[0] == "[" else attr_known_shape,
            type(attr).__name__,
            attr_str,
        )
    console = Console()
    console.print(table)


def describe_data(data):
    print(data, data.is_undirected())

    ### Node-related attributes
    attribute_known_shapes = {
        "num_nodes": "1",
        "x": "[num_nodes, num_node_features]",
        "y": "[num_nodes]",
        "node_time": "[num_nodes] TOCHECK",
        "num_sampled_nodes": "[batch_size, num_neighbors]",
        "n_id": "[num_nodes]",
    }
    _describe_data(
        data, attribute_known_shapes, "node-related attributes", color="blue"
    )

    ### Edge-related attributes
    attribute_known_shapes = {
        "num_edges": "1",
        "edge_attr": "[num_edges, num_edge_features]",
        "edge_index": "[2, num_edges]",
        "edge_label": "[batch_size]",
        "edge_time": "[num_edges]",
        "edge_label_time": "[batch_size]",
        "edge_label_index": "[2, batch_size]",
        "num_sampled_edges": "[1]",
        "e_id": "[num_edges]",
    }
    _describe_data(data, attribute_known_shapes, "edge-related attributes", color="red")
    attribute_known_shapes = {
        "input_id": "[batch_size]",
        "time": "TODO",
        "batch": "TODO",
    }
    _describe_data(data, attribute_known_shapes, "misc attributes", color="yellow")


class GraphMixer(torch.nn.Module):
    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        link_encoder_k: int = 30,
        link_encoder_hidden_channels: int = 12,
        link_encoder_out_channels: int = 34,
        link_encoder_time_channels=56,
        node_encoder_time_window: int = 78,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.link_encoder = LinkEncoder(
            k=link_encoder_k,
            in_channels=num_edge_feats,
            hidden_channels=link_encoder_hidden_channels,
            out_channels=link_encoder_out_channels,
            time_channels=link_encoder_time_channels,
            is_sorted=False,
            dropout=dropout,
        )
        self.node_encoder = NodeEncoder(time_window=node_encoder_time_window)
        self.link_classifier = torch.nn.Linear(
            (link_encoder_out_channels + num_node_feats) * 2, 1
        )

    def forward(self, x, edge_index, edge_attr, edge_time, seed_time, data):
        # [num_nodes, link_encoder_out_channels]
        link_feat = self.link_encoder(
            edge_index,
            edge_attr,
            edge_time,
            seed_time,
        )

        # [num_nodes, num_node_feats]
        node_feat = self.node_encoder(
            x,
            edge_index,
            edge_time,
            seed_time,
        )

        # [num_nodes, link_encoder_out_channels + num_node_feats]
        feats = torch.cat([link_feat, node_feat], dim=-1)

        # TODO: Filter out non-root nodes earlier than here if possible
        feats_src = feats[data.edge_label_index[0]]
        feats_dst = feats[data.edge_label_index[1]]
        feat_pairs = torch.cat([feats_src, feats_dst], dim=-1)

        # [batch_size, 1]
        out = self.link_classifier(feat_pairs).squeeze(-1)
        return out


def main():
    # TODO: Split train/val/test
    data = GDELTLite("data")[0]
    # describe_data(data)

    # TODO: Enable negative sampling
    K = 2
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[7],
        # num_neighbors=[-1]  # to only use K most recent ones in the model
        # neg_sampling_ratio=0.0,
        edge_label=torch.ones(data.num_edges),
        time_attr="edge_time",
        edge_label_time=data.edge_time,
        # edge_label_index=data.edge_index,
        batch_size=13,
        shuffle=False,
    )
    model = GraphMixer(
        num_node_feats=data.x.size(1),
        num_edge_feats=data.edge_attr.size(1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        model.train()
        for sampled_data in loader:
            # sampled_data: num_edges == batch_size * K
            describe_data(sampled_data)
            optimizer.zero_grad()
            pred = model(
                sampled_data.x,
                sampled_data.edge_index,
                sampled_data.edge_attr.to(torch.float),
                sampled_data.edge_time.to(torch.float),
                sampled_data.edge_label_time,
                sampled_data,  # FIXME: REMOVE
            )
            print(sampled_data)
            print(pred.size(), sampled_data.edge_label.size())
            loss = F.binary_cross_entropy_with_logits(
                pred,
                sampled_data.edge_label,
            )
            loss.backward()
            optimizer.step()
            print(loss.item())
            break


if __name__ == "__main__":
    main()
