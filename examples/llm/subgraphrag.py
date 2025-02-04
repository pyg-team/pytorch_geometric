import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from dataset import WebQSPDataset
from torch.optim import Adam
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SubgraphRAGRetriever


def prepare_sample(device, sample):
    return sample.to(device)


def train(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    for sample in tqdm(train_loader):
        if len(sample.x) == 0:
            continue
        num_non_text_entities = 0
        sample = prepare_sample(device, sample)
        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr, num_non_text_entities,
                                   sample.topic_entity_one_hot)

        target_triple_probs = sample.triple_scores
        target_triple_probs = target_triple_probs.unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred_triple_logits,
                                                  target_triple_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        epoch_loss += loss

    epoch_loss /= len(train_loader)

    log_dict = {'loss': epoch_loss}
    return log_dict


@torch.no_grad()
def eval(device, data_loader, model):
    model.eval()

    metric_dict = defaultdict(list)

    for sample in tqdm(data_loader):
        num_non_text_entities = 0
        sample = prepare_sample(device, sample)

        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr, num_non_text_entities,
                                   sample.topic_entity_one_hot).reshape(-1)

        target_triple_probs = sample.triple_scores
        a_entity_ids = sample.a_entity_ids[0]

        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(pred_triple_logits,
                                               descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))

        target_triple_ids = target_triple_probs.nonzero().squeeze(-1).cpu()
        num_target_triples = len(target_triple_ids)

        if num_target_triples == 0:
            continue

        num_total_entities = len(sample.x) + num_non_text_entities

        k_list = [100]

        for k in k_list:
            recall_k_sample = (triple_ranks_pred[target_triple_ids]
                               < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(recall_k_sample /
                                                     num_target_triples)

            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[sample.edge_index[0][triple_mask_k]] = 1.
            entity_mask_k[sample.edge_index[1][triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_ids].sum().item()
            metric_dict[f'ans_recall@{k}'].append(recall_k_sample_ans /
                                                  len(a_entity_ids))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)

    return metric_dict


@torch.no_grad()
def test(device, data_loader, model, checkpoint_dir):
    model.eval()

    pred_dict = dict()
    for sample in tqdm(data_loader):
        if len(sample.x) == 0:
            continue
        num_non_text_entities = 0
        sample = prepare_sample(device, sample)

        entity_list = sample.entity_list[
            0]  # + raw_sample['non_text_entity_list']
        relation_list = sample.relation_list[0]
        rel_ids = sample.rel_ids[0]
        top_K_triples = []
        target_relevant_triples = []

        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr, num_non_text_entities,
                                   sample.topic_entity_one_hot)

        pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
        top_K_results = torch.topk(pred_triple_scores,
                                   min(args.max_K, len(pred_triple_scores)))
        top_K_scores = top_K_results.values.cpu().tolist()
        top_K_triple_IDs = top_K_results.indices.cpu().tolist()

        for j, triple_id in enumerate(top_K_triple_IDs):
            top_K_triples.append(
                (entity_list[sample.edge_index[0][triple_id].item()],
                 relation_list[rel_ids[triple_id]],
                 entity_list[sample.edge_index[1][triple_id].item()],
                 top_K_scores[j]))

        target_relevant_triple_ids = sample.triple_scores.nonzero().reshape(
            -1).tolist()
        for triple_id in target_relevant_triple_ids:
            target_relevant_triples.append((
                entity_list[sample.edge_index[0][triple_id].item()],
                relation_list[rel_ids[triple_id]],
                entity_list[sample.edge_index[1][triple_id].item()],
            ))

        sample_dict = {
            'question':
            sample.question,
            'scored_triples':
            top_K_triples,
            'q_entity':
            sample.q_entity,
            'q_entity_in_graph':
            [entity_list[e_id] for e_id in sample.q_entity_ids[0]],
            'a_entity':
            sample.a_entity,
            'a_entity_in_graph':
            [entity_list[e_id] for e_id in sample.a_entity_ids[0]],
            'max_path_length':
            sample.max_path_length,
            'target_relevant_triples':
            target_relevant_triples
        }

        pred_dict[sample.id[0]] = sample_dict

    torch.save(pred_dict, os.path.join(checkpoint_dir, 'retrieval_result.pth'))
    return pred_dict


def reason():
    pass


def main(args):
    num_epochs = 10000

    device = torch.device('cuda:0')
    torch.set_num_threads(16)
    seed_everything(42)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        exp_prefix = 'webqsp'
        checkpoint_dir = f'{exp_prefix}_{ts}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, 'data', 'WebQSPDataset')
    train_set = WebQSPDataset(path, split='train')
    val_set = WebQSPDataset(path, split='val')

    train_loader = DataLoader(train_set, batch_size=1, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, pin_memory=True)

    test_set = WebQSPDataset(path, split='test')
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)

    emb_size = train_set[0]['q_emb'].shape[-1]
    DDE_kwargs = {'num_rounds': 2, 'num_reverse_rounds': 2}
    model = SubgraphRAGRetriever(emb_size, topic_pe=True,
                                 DDE_kwargs=DDE_kwargs).to(device)

    if os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        print("Loading model from checkpoint...")
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"),
                       weights_only=True))
    else:
        print("Training model...")
        optimizer = Adam(model.parameters(), lr=1e-3)

        num_patient_epochs = 0
        best_val_metric = 0
        for epoch in range(num_epochs):
            num_patient_epochs += 1

            val_eval_dict = eval(device, val_loader, model)
            target_val_metric = val_eval_dict['triple_recall@100']

            if target_val_metric > best_val_metric:
                num_patient_epochs = 0
                best_val_metric = target_val_metric
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, "model.pt"))

                val_log = {'val/epoch': epoch}
                for key, val in val_eval_dict.items():
                    val_log[f'val/{key}'] = val
                print(f"{epoch} val: {val_log}")

            train_log_dict = train(device, train_loader, model, optimizer)

            train_log_dict.update({
                'num_patient_epochs': num_patient_epochs,
                'epoch': epoch
            })
            print(f"train log dict: {train_log_dict}")

            if num_patient_epochs == args.patience:
                break
    test(device, test_loader, model, checkpoint_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='webqsp',
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument(
        '--patience', type=int, default=10, help=
        "Number of epochs with no improvement in validation accuracy before stopping"
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default='',
        help='Directory where the checkpoint and scores will be saved')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()

    main(args)
