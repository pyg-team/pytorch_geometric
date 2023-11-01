# TODO: move model to torch_geometric.nn.models.graph_mixer
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import GDELTLite, Planetoid
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn.models.graph_mixer import LinkEncoder, NodeEncoder
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from tqdm import tqdm




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

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_time,
        seed_time,
        edge_label_index,
    ):
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
        # [batch_size, dim]
        feats_src = feats[edge_label_index[0]]
        # [batch_size, dim]
        feats_dst = feats[edge_label_index[1]]
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
        batch_size=13,
        shuffle=True,
    )
    model = GraphMixer(
        num_node_feats=data.x.size(1),
        num_edge_feats=data.edge_attr.size(1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        train_loss = 0.0
        model.train()
        for sampled_data in tqdm(loader):
            # sampled_data: num_edges == batch_size * K
            optimizer.zero_grad()
            pred = model(
                sampled_data.x,
                sampled_data.edge_index,
                sampled_data.edge_attr.to(torch.float),
                sampled_data.edge_time.to(torch.float),
                sampled_data.edge_label_time,
                sampled_data.edge_label_index,
            )
            loss = F.binary_cross_entropy_with_logits(
                pred,
                sampled_data.edge_label,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(loss.item())
            break
        break


if __name__ == "__main__":
    main()
