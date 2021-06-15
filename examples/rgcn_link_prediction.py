from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets.link_prediction import LinkPrediction
from torch_geometric.nn import GAE, RGCNConv
from torch_geometric.utils.evaluate_relational_link_prediction import evaluate_relational_link_prediction

""""
Implements link prediction on the FB15k237 datasets according to
the `"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.
Caution: very high memory consumption. You should have at least 64GB RAM in your machine. 
"""


class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCNEncoder, self).__init__()
        self.linear_features = nn.Linear(in_channels, 500)
        self.conv1 = RGCNConv(500, out_channels, num_relations=num_relations, num_blocks=5)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations, num_blocks=5)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_features(x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, relation_embedding_dim):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

    def forward(self, z, edge_index, edge_type):
        s = z[edge_index[0, :]]
        r = self.relation_embedding[edge_type]
        o = z[edge_index[1, :]]
        score = torch.sum(s * r * o, dim=1)
        return score


def negative_sampling(edge_index, edge_type, num_nodes, num_edges, omega=1.0, corrupt_subject=True,
                      corrupt_predicate=False, corrupt_object=True):
    negative_samples_count = int(omega * edge_type.size(0))
    permutation = np.random.choice(np.arange(edge_type.size(0)), negative_samples_count)
    s = edge_index[0][permutation]
    r = edge_type[permutation]
    o = edge_index[1][permutation]

    permutation_variants = np.random.choice(np.array([0, 1, 2])[[corrupt_subject, corrupt_predicate, corrupt_object]],
                                            negative_samples_count)

    random_nodes = np.random.choice(np.arange(num_nodes), negative_samples_count)
    random_labels = np.random.choice(np.arange(num_edges), negative_samples_count)

    # replace subjects with random node
    s_permutation_mask = permutation_variants == 0
    s_true_nodes = s.clone()
    s_true_nodes[s_permutation_mask] = 0
    s_random_nodes = torch.tensor(random_nodes)
    s_random_nodes[~s_permutation_mask] = 0
    s_permuted = s_true_nodes + s_random_nodes

    # replace edge labels with random label
    r_permutation_mask = permutation_variants == 1
    r_true_predicates = r.clone()
    r_true_predicates[r_permutation_mask] = 0
    r_random_labels = torch.tensor(random_labels)
    r_random_labels[~r_permutation_mask] = 0
    negative_sample_edge_type = r_true_predicates + r_random_labels

    # replace objects with random node
    o_permutation_mask = permutation_variants == 2
    o_true_nodes = o.clone()
    o_true_nodes[o_permutation_mask] = 0
    o_random_nodes = torch.tensor(random_nodes)
    o_random_nodes[~o_permutation_mask] = 0
    o_permuted = o_true_nodes + o_random_nodes

    # remove sampled triples which are contained in the real graph and therefore no negative samples
    overlap_to_true_triples = (s == s_permuted) * (r == negative_sample_edge_type) * (o == o_permuted)

    s_permuted = s_permuted[~overlap_to_true_triples]
    negative_sample_edge_type = negative_sample_edge_type[~overlap_to_true_triples]
    o_permuted = o_permuted[~overlap_to_true_triples]

    neg_samples_mask = torch.cat((torch.ones(edge_index.size(1), dtype=torch.bool),
                                  torch.zeros(negative_sample_edge_type.size(0), dtype=torch.bool)), 0)
    edge_index = torch.cat((edge_index, torch.cat((s_permuted, o_permuted), 0).reshape(2, -1)), 1)
    edge_type = torch.cat((edge_type, negative_sample_edge_type), 0)
    return edge_index, edge_type, neg_samples_mask


if __name__ == '__main__':
    epochs = 10000
    out_channels = 500
    reg_ratio = 1e-2
    batch_size = None
    evaluate_every = 100
    n_epochs_stop = 5  # early stopping

    writer = SummaryWriter()

    # load data and add node features
    dataset = LinkPrediction('../data', 'FB15k-237')
    data = dataset[0]
    print(data)
    print('num nodes:', data.num_nodes)
    print('num_relations:', data.num_relations)

    model = GAE(RGCNEncoder(data.num_features, out_channels, num_relations=data.num_relations * 2),
                DistMultDecoder(data.num_relations * 2, out_channels))

    device = torch.device('cpu')  # cuda not possible yet, due to high memory consumption
    print('device:', device)
    model = model.to(device)
    x = data.x.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # for early stopping
    best_mrr_val_score = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        # sample random batch of triples if specified
        batch_perm = np.arange(data.train_edge_index.size(1))
        np.random.shuffle(batch_perm)
        if batch_size:
            batch_perm = torch.tensor(batch_perm[:batch_size])
        else:
            batch_perm = torch.tensor(batch_perm)
        train_edge_index_epoch = torch.index_select(data.train_edge_index, 1, batch_perm)
        train_edge_types_epoch = data.train_edge_type[batch_perm]

        # add negative samples
        train_edge_index_epoch_neg_sampled, train_edge_types_epoch_neg_sampled, train_neg_sample_mask_graph_epoch_neg_sampled = negative_sampling(
            train_edge_index_epoch, train_edge_types_epoch, data.num_nodes, data.num_relations, omega=1.0)

        # create bi-directional graph
        train_edge_index_epoch = torch.cat((train_edge_index_epoch, train_edge_index_epoch[[1, 0]]), 1)
        train_edge_types_epoch = torch.cat((train_edge_types_epoch, train_edge_types_epoch + data.num_relations), 0)

        x = x.to(device)
        train_edge_index_epoch = train_edge_index_epoch.to(device)
        train_edge_types_epoch = train_edge_types_epoch.to(device)

        # encode only with true edges
        z = model.encode(x=x, edge_index=train_edge_index_epoch, edge_type=train_edge_types_epoch)

        # decode for all edges, including negative samples
        train_edge_index_epoch_neg_sampled = train_edge_index_epoch_neg_sampled.to(device)
        train_edge_types_epoch_neg_sampled = train_edge_types_epoch_neg_sampled.to(device)
        probability = model.decode(z, train_edge_index_epoch_neg_sampled, train_edge_types_epoch_neg_sampled)

        # compute loss for positive as well as for negative edge examples
        loss_train = F.binary_cross_entropy_with_logits(probability, train_neg_sample_mask_graph_epoch_neg_sampled.type(
            torch.float).to(device)) + reg_ratio * (
                             torch.mean(z.pow(2)) + torch.mean(model.decoder.relation_embedding.pow(2)))

        print(epoch, ':', loss_train)
        writer.add_scalar('Loss/train', loss_train, epoch)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if (epoch + 1) % evaluate_every == 0:
            with torch.no_grad():
                # perform evaluation on cpu due to high memory consumption caused by large number of edges
                model.eval()

                train_edge_index = torch.cat((data.train_edge_index, data.train_edge_index[[1, 0]]), 1)
                train_edge_type = torch.cat((data.train_edge_type, data.train_edge_type + data.num_relations), 0)

                train_edge_index = train_edge_index.to(device)
                train_edge_type = train_edge_type.to(device)

                z = model.encode(x=x, edge_index=train_edge_index, edge_type=train_edge_type)

                edge_index_all = torch.cat(
                    (data.test_edge_index,
                     data.train_edge_index,
                     data.valid_edge_index), dim=1)
                edge_type_all = torch.cat((data.test_edge_type,
                                           data.train_edge_type,
                                           data.valid_edge_type))

                all_triplets = torch.stack((edge_index_all[0], edge_type_all, edge_index_all[1]), 1)
                valid_triplets = torch.stack((data.valid_edge_index[0], data.valid_edge_type, data.valid_edge_index[1]),
                                             1)

                scores_valid = evaluate_relational_link_prediction(latent_node_features=z,
                                                                   latent_edge_features=model.decoder.relation_embedding,
                                                                   test_triplets=valid_triplets,
                                                                   all_triplets=all_triplets)

                print('valid scores', scores_valid)
                writer.add_scalar('MRR/valid', scores_valid['MRR'], epoch)
                writer.add_scalar('H@3/valid', scores_valid['H@3'], epoch)

                if scores_valid['MRR'] > best_mrr_val_score:
                    epochs_no_improve = 0
                    best_mrr_val_score = scores_valid['MRR']

                    # store model parameters
                    torch.save({'z': z,
                                'w': model.decoder.relation_embedding}, '../data/trained_models/embeddings.pt')
                    torch.save(model.state_dict(), '../data/trained_models/model.pt')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= n_epochs_stop:
                        print('Early stopping: no improvement in validation MRR score')

                        test_triplets = torch.stack(
                            (data.test_edge_index[0], data.test_edge_type, data.test_edge_index[1]), 1)

                        scores_test = evaluate_relational_link_prediction(latent_node_features=z,
                                                                          latent_edge_features=model.decoder.relation_embedding,
                                                                          test_triplets=test_triplets,
                                                                          all_triplets=all_triplets)

                        print('test scores', scores_test)
                        break
