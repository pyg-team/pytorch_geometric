import torch

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)


@register_node_encoder('Integer')
class IntegerFeatureEncoder(torch.nn.Module):
    """
    Provides an encoder for integer node features.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes (int): the number of classes for the
        embedding mapping to learn from

    Example:
        >>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
        >>> batch = torch.randint(0, 10, (10, 2))
        >>> encoded_batch = encoder(batch)
        >>> print(encoded_batch.x)
        # tensor([[0.2470, 0.7026, 0.9272, 0.3419, 0.5079, 0.0506, 0.8246,
        #          0.0228, 0.9619, 0.3349, 0.8910],
        #          [0.4278, 0.0592, 0.1844, 0.5261, 0.3682, 0.6348, 0.4851,
        #           0.7007, 0.4754, 0.3943, 0.7505]])
    """
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_classes, emb_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_node_encoder('Atom')
class AtomEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None

    Example:
        >>> encoder = AtomEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoded_batch = encoder(batch)
        >>> print(encoded_batch.x)
        # tensor([[0.2470, 0.7026, 0.9272],
        #          [0.4278, 0.0592, 0.1844],
        #          [0.8246, 0.0228, 0.9619],
        #          [0.3349, 0.3943, 0.8910],
        #          [0.5079, 0.6348, 0.8246],
        #          [0.0506, 0.7007, 0.4851],
        #          [0.8246, 0.4754, 0.3943],
        #          [0.0228, 0.7505, 0.7007],
        #          [0.9619, 0.3943, 0.3349],
        #          [0.3349, 0.8910, 0.5079]])
    """
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features
        return batch


@register_edge_encoder('Bond')
class BondEncoder(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension


    Example:
        >>> encoder = BondEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoded_batch = encoder(batch)
        >>> print(encoded_batch.edge_attr)
        # tensor([[0.2470, 0.7026, 0.9272],
        #          [0.4278, 0.0592, 0.1844],
        #          [0.8246, 0.0228, 0.9619],
        #          [0.3349, 0.3943, 0.8910],
        #          [0.5079, 0.6348, 0.8246],
        #          [0.0506, 0.7007, 0.4851],
        #          [0.8246, 0.4754, 0.3943],
        #          [0.0228, 0.7505, 0.7007],
        #          [0.9619, 0.3943, 0.3349],
        #          [0.3349, 0.8910, 0.5079]])
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding
        return batch
