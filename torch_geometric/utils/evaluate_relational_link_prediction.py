from tqdm import tqdm
import numpy as np
import torch


def evaluate_relational_link_prediction(latent_node_features, latent_edge_features, test_triplets,
                                        all_triplets) -> dict:
    r""" Evaluates relational link prediction according to Borders et al. "Translating Embeddings for
    Modeling Multi-relational Data" (2013) by computing the scores: MRR, Hits@1, Hits@3, Hits@10

        Args:
            latent_node_features (Tensor): Computed node embeddings.
            latent_edge_features (Tensor): Computed edge (type) embeddings.
            test_triplets (Tensor): Contains test triples (head, relation, tail) .
            all_triplets (Tensor): Contains all triples (head, relation, tail) of the graph.

        :rtype: Dict
        """

    with torch.no_grad():
        num_nodes = latent_node_features.size(0)
        ranks_r = {r: [] for r in np.unique(all_triplets[:, 1])}  # stores ranks computed for each relation

        for triplet in tqdm(test_triplets):
            head, relation, tail = triplet

            # permute object
            # try all nodes as tails, but delete all true triplets with the same head and relation as the test triplet
            delete_triplet_index = torch.nonzero(
                torch.logical_and(all_triplets[:, 0] == head, all_triplets[:, 1] == relation))
            delete_entity_index = torch.flatten(all_triplets[delete_triplet_index, 2]).numpy()

            tails = torch.cat(
                (torch.from_numpy(np.array(list(set(np.arange(num_nodes)) - set(delete_entity_index)))), tail.view(-1)))

            # head nods and relations are all the same
            heads = torch.zeros(tails.size(0)).fill_(head).type(torch.long)
            edge_types = torch.zeros(tails.size(0)).fill_(relation).type(torch.long)

            s = latent_node_features[heads]
            r = latent_edge_features[edge_types]
            o = latent_node_features[tails]
            scores = torch.sum(s * r * o, dim=1)
            scores = torch.sigmoid(scores)
            target = torch.tensor(len(tails) - 1)
            rank = sort_and_rank(scores, target)
            ranks_r[relation.item()].append(rank)

            # permute subject
            delete_triplet_index = torch.nonzero(
                torch.logical_and(all_triplets[:, 1] == relation, all_triplets[:, 2] == tail))
            delete_entity_index = torch.flatten(all_triplets[delete_triplet_index, 0]).numpy()

            heads = torch.cat(
                (torch.from_numpy(np.array(list(set(np.arange(num_nodes)) - set(delete_entity_index)))), head.view(-1)))

            # tail nods and relations are all the same
            edge_types = torch.zeros(heads.size(0)).fill_(relation).type(torch.long)
            tails = torch.zeros(heads.size(0)).fill_(tail).type(torch.long)

            s = latent_node_features[heads]
            r = latent_edge_features[edge_types]
            o = latent_node_features[tails]
            scores = torch.sum(s * r * o, dim=1)
            scores = torch.sigmoid(scores)

            target = torch.tensor(len(tails) - 1)
            rank = sort_and_rank(scores, target)
            ranks_r[relation.item()].append(rank)

        # compute scores
        k = []
        for x in ranks_r.values(): k.extend(x)
        ranks = torch.tensor(k)
        scores = {'MRR': torch.mean(1.0 / ranks.float()).item(),
                  'H@1': torch.mean((ranks <= 1).float()).item(),
                  'H@3': torch.mean((ranks <= 3).float()).item(),
                  'H@10': torch.mean((ranks <= 10).float()).item()}

    return scores


def sort_and_rank(score, target):
    _, indices = torch.sort(score, descending=True)
    rank = torch.reshape(torch.nonzero(indices == target), (-1,)).item() + 1
    return rank


def get_ranks(scores, true_edges) -> np.ndarray:
    """
    Computes the ranks for the edges marked as true edges (ones in true_edges) according to the given scores.
    """
    y_pred = np.array(scores)
    y_true = np.array(true_edges)

    if y_pred.size != y_true.size or y_pred.ndim != 1 or y_true.ndim != 1:
        raise ArithmeticError('input not valid (check size and shape)')

    idx = np.argsort(y_pred)[::-1]
    y_ord = y_true[idx]
    ranks = np.where(y_ord == 1)[0] + 1

    # true edges do not affect the rank: decrease each rank of true edges by the number of true edges ranked before
    ranks_cleared = ranks - np.arange(len(ranks))

    return ranks_cleared
