"""This example implements the G-Retriever model
(https://arxiv.org/abs/2402.07630) using PyG.

G-Retriever significantly reduces hallucinations by 54% compared to the
stand-alone LLM baseline.

Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`
"""
import argparse
import math
import os.path as osp
import re
import time

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM


def compute_metrics(eval_output):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    all_hit = []
    all_precision = []
    all_recall = []
    all_f1 = []

    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        try:
            pred = pred.split('[/s]')[0].strip().split('|')
            hit = re.findall(pred[0], label)
            all_hit.append(len(hit) > 0)

            label = label.split('|')
            matches = set(pred).intersection(set(label))
            precision = len(matches) / len(set(label))
            recall = len(matches) / len(set(pred))
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        except Exception as e:
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print(f'Exception: {e}')
            print('------------------')

    hit = sum(all_hit) / len(all_hit)
    precision = sum(all_precision) / len(all_precision)
    recall = sum(all_recall) / len(all_recall)
    f1 = sum(all_f1) / len(all_f1)

    print(f'Hit: {hit:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')


def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)


def load_params_dict(model, save_path):
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model


def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name == 'llm':
        return model(batch.question, batch.label, batch.desc)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)


def inference_step(model, batch, model_save_name):
    if model_save_name == 'llm':
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)


def train(
    num_epochs,
    hidden_channels,
    num_gnn_layers,
    batch_size,
    eval_batch_size,
    lr,
    checkpointing=False,
    tiny_llama=False,
):
    def adjust_learning_rate(param_group, LR, epoch):
        # Decay the learning rate with half-cycle cosine after warmup
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lr = LR
        else:
            lr = min_lr + (LR - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                               (num_epochs - warmup_epochs)))
        param_group['lr'] = lr
        return lr

    start_time = time.time()
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', '..', 'data', 'WebQSPDataset')
    train_dataset = WebQSPDataset(path, split='train')
    val_dataset = WebQSPDataset(path, split='val')
    test_dataset = WebQSPDataset(path, split='test')

    seed_everything(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    gnn = GAT(
        in_channels=1024,
        hidden_channels=hidden_channels,
        out_channels=1024,
        num_layers=num_gnn_layers,
        heads=4,
    )
    if tiny_llama:
        llm = LLM(
            model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
            num_params=1,
        )
        model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
    else:
        llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf', num_params=7)
        model = GRetriever(llm=llm, gnn=gnn)

    model_save_name = 'gnn_llm' if num_gnn_layers is not None else 'llm'
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))
    grad_steps = 2

    best_epoch = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        if epoch == 0:
            print(f"Total Preparation Time: {time.time() - start_time:2f}s")
            start_time = time.time()
            print("Training beginning...")
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = get_loss(model, batch, model_save_name)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)

            optimizer.step()
            epoch_loss = epoch_loss + float(loss)

            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')

        val_loss = 0
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")
        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(model, f'{model_save_name}_best_val_loss_ckpt.pt')
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{model_save_name}_best_val_loss_ckpt.pt',
        )

    model.eval()
    eval_output = []
    print("Final evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            pred = inference_step(model, batch, model_save_name)
            eval_data = {
                'pred': pred,
                'question': batch.question,
                'desc': batch.desc,
                'label': batch.label
            }
            eval_output.append(eval_data)
        progress_bar_test.update(1)

    compute_metrics(eval_output)
    print(f"Total Training Time: {time.time() - start_time:2f}s")
    save_params_dict(model, f'{model_save_name}.pt')
    torch.save(eval_output, f'{model_save_name}_eval_outs.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--tiny_llama', action='store_true')
    args = parser.parse_args()

    start_time = time.time()
    train(
        args.epochs,
        args.gnn_hidden_channels,
        args.num_gnn_layers,
        args.batch_size,
        args.eval_batch_size,
        args.lr,
        checkpointing=args.checkpointing,
        tiny_llama=args.tiny_llama,
    )
    print(f"Total Time: {time.time() - start_time:2f}s")
