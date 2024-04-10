"""This example implements G-retriever using PyG.
Original Paper: https://arxiv.org/abs/2402.07630
“G-Retriever significantly reduces hallucinations
by 54% compared to the [LLM] baseline“.

requirements on top of basic PyG:
pip install peft datasets transformers pcst_fast sentencepiece tqdm pandas
"""
import argparse
import gc
import math
import re
import time
from os import path

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.data import DataLoader
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.nn.models.gnn_llm import GNN_LLM, LLM

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '[/s]'
IGNORE_INDEX = -100
llama2_str_name = "meta-llama/Llama-2-7b-chat-hf"
max_txt_len = 512
max_new_tokens = 32
pad_token_id = 0
padding_side = 'left'


def detect_hallucinate(pred, label):
    try:
        pred = pred.split('[/s]')[0].strip().split('|')
        correct_hit = len(re.findall(pred[0], label)) > 0
        correct_hit = correct_hit or any(
            [label_i in pred.lower() for label_i in label.split('|')])
        hallucination = not correct_hit
        return hallucination
    except:  # noqa
        return "skip"


def compute_accuracy(eval_output):
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

    return hit


def main(since: float, num_epochs: int, hidden_channels: int,
         num_gnn_layers: int, batch_size: int, lr: float):
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
        param_group["lr"] = lr
        return lr

    seed_everything(42)

    dataset = WebQSPDataset()
    idx_split = dataset.split_idxs

    # Step 1: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Step 2: Build Model
    model = GNN_LLM(gnn_hidden_channels=hidden_channels,
                    num_gnn_layers=num_gnn_layers)

    # Step 3 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': .05
        },
    ], betas=(0.9, 0.95))
    grad_steps = 2
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || \
        all params: {all_param} || \
        trainable%: {100 * trainable_params / all_param}")

    # Step 4 Training
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.
        if epoch == 0:
            prep_time = round(time.time() - since, 2)
            print("Total Prep Time (prep_time) =", prep_time)
            print("Training beginning...")
        epoch_str = f"Epoch: {epoch + 1}|{num_epochs}"
        loader = tqdm(train_loader, desc=epoch_str)
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)

            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f",Train Loss (Epoch Mean): {train_loss}")

        val_loss = 0.
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss}")

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5 Evaluating
    print("Final Evaluation...")
    model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)

    # Step 6 Post-processing & compute metrics
    acc = compute_accuracy(eval_output)
    print(f'Test Acc {acc}')
    # save model
    print("Saving Model...")
    torch.save(model, "gnn_llm.pt")
    print("Done!")
    return prep_time, dataset, model


def minimal_demo(model, dataset):
    # Step 1: Define a single batch size test loader
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    # batch size 1 loader for simplicity
    loader = DataLoader(test_dataset, batch_size=1, drop_last=False,
                        pin_memory=True, shuffle=False)
    # define the pure pretrained LLM
    pure_llm = LLM()

    # Step loop through the loader and run both models
    gnn_llm_hallucin_sum = 0
    pure_llm_hallucin_sum = 0
    final_prnt_str = ""
    print("Checking LLM vs GNN+LLM for hallucinations...")
    for batch in tqdm(loader):
        question = batch.question[0]
        correct_answer = batch.label[0]
        gnn_llm_out = model.inference(batch)
        pure_llm_out = pure_llm.inference(batch)
        gnn_llm_pred = gnn_llm_out['pred'][0]
        pure_llm_pred = pure_llm_out['pred'][0]
        gnn_llm_hallucinates = detect_hallucinate(gnn_llm_pred, correct_answer)
        pure_llm_hallucinates = detect_hallucinate(pure_llm_pred,
                                                   correct_answer)
        if gnn_llm_hallucinates == "skip" or pure_llm_hallucinates == "skip":
            # skipping since hard to evaluate if the answer is a hallucination
            continue
        gnn_llm_hallucin_sum += bool(gnn_llm_hallucinates)
        pure_llm_hallucin_sum += bool(pure_llm_hallucinates)
        # showcase LLM hallucinations solved by GNN
        if pure_llm_hallucinates and not gnn_llm_hallucinates:
            final_prnt_str += "Prompt: " + question + "\n"
            final_prnt_str += "Correct Answer: " + correct_answer + "\n"
            final_prnt_str += "Pure LLM Prediction: " + pure_llm_pred + "\n"
            final_prnt_str += "GNN+LLM Prediction:" + gnn_llm_pred + "\n"
            final_prnt_str += "#" * 20 + "\n"
    print("Total GNN+LLM Hallucinations:", gnn_llm_hallucin_sum)
    print("Total Pure LLM Hallucinations:", pure_llm_hallucin_sum)
    percent = 100.0 * round(1 -
                            (gnn_llm_hallucin_sum / pure_llm_hallucin_sum), 2)
    print(f"GNN reduces hallucinations by: ~{percent}%")
    print("Note: hallucinations detected by regex hence the ~")
    print("Potential instances where GNN solves the hallucinations of LLM:")
    print(final_prnt_str)


if __name__ == "__main__":
    # check if saved model
    if path.exists("gat_llama.pt"):
        print("Existing trained model found.")
        # ask if want to retrain or skip to demo
        print("Would you like to retrain?")
        user_input = str(input("(y/n):")).lower()
        retrain = user_input == "y"
    else:
        retrain = True
    if retrain:
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_channels', type=int, default=1024)
        parser.add_argument('--num_gnn_layers', type=int, default=4)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=4)
        args = parser.parse_args()
        since = time.time()
        prep_time, dataset, model = main(since, args.epochs,
                                         args.hidden_channels,
                                         args.num_gnn_layers, args.batch_size,
                                         args.lr)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
        print("E2E time minus Prep Time =", e2e_time - prep_time, "seconds")
    else:
        model = torch.load("gnn_llm.pt")
        dataset = WebQSPDataset()
    print("Would you like a minimal demo showcasing how \
     GNN+LLM can solve LLM hallucinations?")
    user_input = str(input("(y/n):")).lower()
    if user_input == "y":
        minimal_demo(model, dataset)
