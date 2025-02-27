import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import re
import string
from functools import lru_cache

from collections import defaultdict
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import WebQSPDataset
from retriever_model import SubgraphRAGRetriever
from torch_geometric import seed_everything

from torch_geometric.nn.nlp import LLM

SYS_PROMPT = ('Based on the triplets from a knowledge graph, '
    'please answer the given question. '
    'Please keep the answers as simple as possible and return all the '
    'possible answers as a list, each with a prefix "ans:".')

def prepare_sample(device, sample):
    return sample.to(device)

def train(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    for sample in tqdm(train_loader):
        if len(sample.x) == 0:
            continue
        sample = prepare_sample(device, sample)
        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr, sample.topic_entity_one_hot)

        target_triple_probs = sample.target_triple_scores
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
        sample = prepare_sample(device, sample)

        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr,
                                   sample.topic_entity_one_hot).reshape(-1)

        target_triple_probs = sample.target_triple_scores
        a_entity_ids = sample.a_entity_ids[0]

        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(pred_triple_logits,
                                               descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))

        target_triple_ids = target_triple_probs.nonzero().squeeze(-1).cpu()
        num_target_triplets = len(target_triple_ids)

        if num_target_triplets == 0:
            continue

        num_total_entities = len(sample.x)

        k_list = [100]

        for k in k_list:
            recall_k_sample = (triple_ranks_pred[target_triple_ids]
                               < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(recall_k_sample /
                                                     num_target_triplets)

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
            0]
        relation_list = sample.relation_list[0]
        rel_ids = sample.rel_ids[0]
        top_K_triplets = []
        target_relevant_triplets = []

        pred_triple_logits = model(sample.edge_index, sample.q_emb, sample.x,
                                   sample.edge_attr, num_non_text_entities,
                                   sample.topic_entity_one_hot)

        pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
        top_K_results = torch.topk(pred_triple_scores,
                                   min(args.max_K, len(pred_triple_scores)))
        top_K_scores = top_K_results.values.cpu().tolist()
        top_K_triple_IDs = top_K_results.indices.cpu().tolist()

        for j, triple_id in enumerate(top_K_triple_IDs):
            top_K_triplets.append(
                (entity_list[sample.edge_index[0][triple_id].item()],
                 relation_list[rel_ids[triple_id]],
                 entity_list[sample.edge_index[1][triple_id].item()],
                 top_K_scores[j]))

        target_relevant_triple_ids = sample.target_triple_scores.nonzero().reshape(
            -1).tolist()
        for triple_id in target_relevant_triple_ids:
            target_relevant_triplets.append((
                entity_list[sample.edge_index[0][triple_id].item()],
                relation_list[rel_ids[triple_id]],
                entity_list[sample.edge_index[1][triple_id].item()],
            ))

        sample_dict = {
            'question':
            sample.question,
            'scored_triplets':
            top_K_triplets,
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
            'target_relevant_triplets':
            target_relevant_triplets
        }

        pred_dict[sample.id[0]] = sample_dict

    torch.save(pred_dict, os.path.join(checkpoint_dir, 'retrieval_result.pth'))
    return pred_dict

def llm_infer(llm, user_query, sys_prompt, max_tokens=4096):
    conversation = [{"role": "system", "content": sys_prompt}]
    conversation.append({"role": "user", "content": user_query})
    output = llm.inference([conversation], max_tokens=max_tokens)[0]
    return output

def prepare_prompt(sample, K_triplets, threshold=0.0):
    question_prompt = "Question:\n" + sample['question'][0]
    if question_prompt[-1] != '?':
        question_prompt += '?'
    if K_triplets > 0:
        input_triplets = sample['scored_triplets']
        if threshold > 0.0:
            input_triplets = [(triplet[0], triplet[1], triplet[2])
                            for triplet in input_triplets
                            if triplet[3] >= threshold]
        # Ensure that triplets are unique
        input_triplets = list(dict.fromkeys(input_triplets))
        input_triplets = input_triplets[:K_triplets]
        input_triplets = [f"({triplet[0]},{triplet[1]},{triplet[2]})"
                        for triplet in input_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    else:
        triplet_prompt = ''

    if triplet_prompt == '':
        user_query = question_prompt
    else:
        user_query = "\n\n".join([triplet_prompt, question_prompt])

    return user_query



def reason(pred_dict, model_name, K_triplets, max_tokens=4096):
    llm = LLM(model_name=model_name, backend='openai')
    llm.llm.generation_config.pad_token_id = llm.tokenizer.eos_token_id

    llm_preds = []
    for _, sample in tqdm(pred_dict.items()):
        user_query = prepare_prompt(sample,
                                    K_triplets)
        llm_preds.append(llm_infer(llm, user_query, SYS_PROMPT, max_tokens))
    return llm_preds

################################
########## Metrics #############
################################

@lru_cache(maxsize=1000)
def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def compute_scores(pred_dict, llm_preds, double_check=False):
    hits_any = 0
    hits_at_1 = 0
    all_precision = []
    all_recall = []
    all_f1 = []

    ans_pattern = re.compile(r'ans:', re.IGNORECASE)
    total_samples = len(pred_dict)

    for i, (_, sample) in enumerate(tqdm(pred_dict.items())):
        gt_answers = sample['a_entity'][0]
        prediction_text = llm_preds[i].lower()

        if not gt_answers or not ans_pattern.search(prediction_text):
            all_precision.append(0)
            all_recall.append(0)
            all_f1.append(0)
            continue

        all_predictions = [p.strip() for p in prediction_text.split('ans:')[1:]]

        gt_answers = [normalize(answer) for answer in gt_answers]
        all_predictions = [normalize(pred) for pred in all_predictions]

        first_pred = all_predictions[0]
        for answer in gt_answers:
            if (answer in first_pred) or (double_check and first_pred in answer):
                hits_at_1 += 1
                break

        matched_indices = set()
        for pred in all_predictions:
            for j, answer in enumerate(gt_answers):
                if j in matched_indices:
                    continue
                if (answer in pred) or (double_check and pred in answer):
                    matched_indices.add(j)
                    break

        num_matches = len(matched_indices)

        hits_any += (num_matches > 0)
        precision = num_matches / len(all_predictions)
        recall = num_matches / len(gt_answers)

        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    hit_rate = hits_any / total_samples
    hit_at_1 = hits_at_1 / total_samples
    avg_precision = sum(all_precision) / total_samples
    avg_recall = sum(all_recall) / total_samples
    avg_f1 = sum(all_f1) / total_samples

    print("\n" + "="*50)
    print(f"Evaluation Metrics ({total_samples} samples):")
    print("-"*50)
    print(f"Hit (Any): {hit_rate:.4f} ({hits_any}/{total_samples})")
    print(f"Hit@1:     {hit_at_1:.4f} ({hits_at_1}/{total_samples})")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print("="*50)

    return {
        'hit_rate': hit_rate,
        'hit_at_1': hit_at_1,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'total_samples': total_samples
    }

def compute_metrics(pred_dict, llm_preds):
    print("\n" + "="*50)
    print(f"Evaluation Metrics Vanilla):")
    compute_scores(pred_dict, llm_preds, False)
    print("\n" + "="*50)
    print(f"Evaluation Metrics with Bidirectional Check):")
    compute_scores(pred_dict, llm_preds, True)


def main(args):
    pred_file_path = os.path.join(args.checkpoint_dir,
                                  'retrieval_result.pth')
    llm_pred_file = os.path.join(args.checkpoint_dir,
                                    'llm_pred.pt')
    k_triplets = 0 if args.no_triplets else args.k_triplets

    if os.path.exists(pred_file_path):
        pred_dict = torch.load(pred_file_path, weights_only=False)
        print("Retriever prediction file found, proceed to reasoning")
        if not os.path.exists(llm_pred_file):
            llm_preds = reason(pred_dict, args.model_name, k_triplets)
            torch.save(llm_preds, llm_pred_file)
        else:
            print("LLM preds found, loading")
            llm_preds = torch.load(llm_pred_file, weights_only=False)
        compute_metrics(pred_dict, llm_preds)
        return
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

    train_set = WebQSPDataset(path, split='train', subgraphrag=True)
    val_set = WebQSPDataset(path, split='val', subgraphrag=True)
    test_set = WebQSPDataset(path, split='test', subgraphrag=True)

    train_loader = DataLoader(train_set, batch_size=1, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)

    emb_size = train_set[0]['q_emb'].shape[-1]
    model = SubgraphRAGRetriever(emb_size, topic_pe=True,
                                 dde_rounds=2,
                                 rev_dde_rounds=2).to(device)

    if os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        print("Loading model from checkpoint...")
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"),
                       weights_only=True))
    else:
        print("Training Retriever model...")
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
            print(f"{train_log_dict}")

            if num_patient_epochs == args.patience:
                break
    pred_dict = test(device, test_loader, model, checkpoint_dir)

    print("Reasoning with LLM")
    llm_preds = reason(pred_dict, args.model_name, k_triplets)
    torch.save(llm_preds, llm_pred_file)
    compute_metrics(pred_dict, llm_preds)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='webqsp',
                        choices=['webqsp', 'cwq'], help='Dataset name')
    # Retriever args
    parser.add_argument(
        '--patience', type=int, default=10, help=
        "Number of epochs with no improvement in validation accuracy before stopping"
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default='',
        help='Directory where the checkpoint and scores will be saved')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')

    # Reasoning args
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--k_triplets", type=int, default=100,
                        help="Top K triplets used for reasoning")
    parser.add_argument("--no_triplets", action="store_true", help="Overrides --k_triplets")
    args = parser.parse_args()

    main(args)
