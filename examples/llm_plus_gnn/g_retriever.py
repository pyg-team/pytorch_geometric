"""This example implements G-retriever using PyG.
Original Paper: https://arxiv.org/abs/2402.07630
“G-Retriever significantly reduces hallucinations
by 54% compared to the [LLM] baseline“.
requirements on top of basic PyG:
pip install datasets transformers pcst_fast sentencepiece tqdm pandas
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
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.text import LLM


def detect_hallucinate(pred, label):
    try:
        split_pred = pred.split('[/s]')[0].strip().split('|')
        correct_hit = len(re.findall(split_pred[0], label)) > 0
        correct_hit = correct_hit or any(
            [label_i in pred.lower() for label_i in label.split('|')])
        hallucination = not correct_hit
        return hallucination
    except:  # noqa
        return "skip"


def compute_accuracy(eval_output) -> float:
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


def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    torch.save(state_dict, save_path)


def load_params_dict(model, save_path):
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model


def get_loss(model, batch, model_save_name) -> torch.Tensor:
    if model_save_name == "llm":
        return model(batch.question, batch.label, batch.desc)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.ptr, batch.label, batch.edge_attr, batch.desc)


def inference_step(model, batch, model_save_name):
    if model_save_name == "llm":
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.ptr, batch.edge_attr,
                               batch.desc)


def train(since, num_epochs, hidden_channels, num_gnn_layers, batch_size,
          eval_batch_size, lr, loss_fn, inference_fn, model=None, dataset=None,
          checkpointing=False):
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
    if dataset is None:
        dataset = WebQSPDataset()
    idx_split = dataset.split_idxs

    # Step 1: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Step 2: Build Model
    if model is None:
        model = GRetriever(gnn_hidden_channels=hidden_channels,
                           num_gnn_layers=num_gnn_layers)
    if num_gnn_layers is not None:
        model_save_name = "gnn_llm"
    else:
        model_save_name = "llm"

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
    if model is None:
        trainable_params, all_param = model.print_trainable_params()
        print(f"trainable params: {trainable_params} || \
            all params: {all_param} || \
            trainable%: {100 * trainable_params / all_param}")

    best_val_loss = float('inf')
    # Step 4 Training
    best_epoch = 0
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
            loss = loss_fn(model, batch, model_save_name)
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
                loss = loss_fn(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss}")
        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best val loss model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(model, model_save_name + "_best_val_loss_ckpt.pt")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5 Evaluating
    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(model,
                                 model_save_name + "_best_val_loss_ckpt.pt")
    model.eval()
    eval_output = []
    print("Final Evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = inference_fn(model, batch, model_save_name)
            output["label"] = batch.label
            eval_output.append(output)
        progress_bar_test.update(1)

    # Step 6 Post-processing & compute metrics
    acc = compute_accuracy(eval_output)
    print(f'Test Acc {acc}')
    # save model
    print("Saving Model...")
    save_params_dict(model, model_save_name + ".pt")
    print("Saving eval output for downstream demo...")
    torch.save(eval_output, model_save_name + "_eval_outs.pt")
    print("Done!")
    return prep_time, dataset, eval_output


def minimal_demo(gnn_llm_eval_outs, dataset, lr, epochs, batch_size,
                 eval_batch_size, loss_fn, inference_fn,
                 skip_pretrained_LLM=False):
    if not skip_pretrained_LLM:
        print("First comparing against a pretrained LLM...")
    # Step 1: Define a single batch size test loader
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    # batch size 1 loader for simplicity
    loader = DataLoader(test_dataset, batch_size=1, drop_last=False,
                        pin_memory=True, shuffle=False)
    pure_llm = LLM()
    if path.exists("demo_save_dict.pt"):
        print("Saved demo outputs for LLM and GNN+LLM found.")
        print("Would you like to redo untuned LLM eval?")
        user_input = str(input("(y/n):")).lower()
        skip_step_one = user_input == "n"
    else:
        skip_step_one = False

    if not skip_step_one:
        gnn_llm_hallucin_sum = 0
        pure_llm_hallucin_sum = 0
        gnn_save_list = []
        untuned_llm_save_list = []
        gnn_llm_preds = []
        for out in gnn_llm_eval_outs:
            gnn_llm_preds += out['pred']
        print(
            "Checking pretrained LLM vs trained GNN+LLM for hallucinations...")
        for i, batch in enumerate(tqdm(loader)):
            question = batch.question[0]
            correct_answer = batch.label[0]
            if skip_pretrained_LLM:
                pure_llm_pred = None
                pure_llm_hallucinates = False
            else:
                # GNN+LLM only using 32 tokens to answer, give untrained LLM more
                pure_llm_out = pure_llm.inference(batch.question, batch.desc,
                                                  max_out_tokens=256)
                pure_llm_pred = pure_llm_out['pred'][0]
                pure_llm_hallucinates = detect_hallucinate(
                    pure_llm_pred, correct_answer)
                untuned_llm_save_list += [(pure_llm_pred,
                                           pure_llm_hallucinates)]

            gnn_llm_pred = gnn_llm_preds[i]
            gnn_llm_hallucinates = detect_hallucinate(gnn_llm_pred,
                                                      correct_answer)
            gnn_save_list += [(gnn_llm_pred, gnn_llm_hallucinates)]

            if gnn_llm_hallucinates == "skip" or pure_llm_hallucinates == "skip":  # noqa
                # skipping when hallucination is hard to eval
                continue
            gnn_llm_hallucin_sum += int(gnn_llm_hallucinates)
            pure_llm_hallucin_sum += int(pure_llm_hallucinates)
        if not skip_pretrained_LLM:
            print("Total Pure LLM Hallucinations:", pure_llm_hallucin_sum)
            print("Total GNN+LLM Hallucinations:", gnn_llm_hallucin_sum)
            percent = 100.0 * round(
                1 - (gnn_llm_hallucin_sum / pure_llm_hallucin_sum), 2)
            print(f"GNN reduces pretrained LLM hallucinations by: ~{percent}%")
            print("Note: hallucinations detected by regex hence the ~")
            print("Now we see how the LLM compares when finetuned...")
            print("Saving outputs of GNN+LLM and pretrained LLM...")
        save_dict = {
            "gnn_save_list": gnn_save_list,
            "untuned_llm_save_list": untuned_llm_save_list,
            "gnn_llm_hallucin_sum": gnn_llm_hallucin_sum,
            "pure_llm_hallucin_sum": pure_llm_hallucin_sum
        }
        torch.save(save_dict, "demo_save_dict.pt")
        print("Done!")
    else:
        save_dict = torch.load("demo_save_dict.pt")
        gnn_save_list = save_dict["gnn_save_list"]
        untuned_llm_save_list = save_dict["untuned_llm_save_list"]
        gnn_llm_hallucin_sum = save_dict["gnn_llm_hallucin_sum"]
        pure_llm_hallucin_sum = save_dict["pure_llm_hallucin_sum"]

    trained_llm_hallucin_sum = 0
    untuned_llm_hallucin_sum = pure_llm_hallucin_sum
    final_prnt_str = ""
    if path.exists("llm.pt") and path.exists("llm_eval_outs.pt"):
        print("Existing finetuned LLM found.")
        print("Would you like to retrain?")
        user_input = str(input("(y/n):")).lower()
        retrain = user_input == "y"
    else:
        retrain = True
    if retrain:
        print("Finetuning LLM...")
        since = time.time()
        _, _, pure_llm_eval_outputs = train(since, 1, None, None, batch_size,
                                            eval_batch_size, lr, loss_fn,
                                            inference_fn, model=pure_llm,
                                            dataset=dataset)
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
    else:
        pure_llm_eval_outputs = torch.load("llm_eval_outs.pt")
    pure_llm_preds = []
    for out in pure_llm_eval_outputs:
        pure_llm_preds += out['pred']
    print("Final comparison between all models...")
    for i, batch in enumerate(tqdm(loader)):
        question = batch.question[0]
        correct_answer = batch.label[0]
        gnn_llm_pred, gnn_llm_hallucinates = gnn_save_list[i]
        untuned_llm_pred, untuned_llm_hallucinates = untuned_llm_save_list[i]
        if gnn_llm_hallucinates == "skip" or untuned_llm_hallucinates == "skip":  # noqa
            continue
        pure_llm_pred = pure_llm_preds[i]
        pure_llm_hallucinates = detect_hallucinate(pure_llm_pred,
                                                   correct_answer)
        if pure_llm_hallucinates == "skip":
            continue
        trained_llm_hallucin_sum += int(pure_llm_hallucinates)
        if skip_pretrained_LLM:
            # we did not check the untrained LLM, so do not decide to demo
            # based on this.
            untuned_llm_hallucinates = True
        if untuned_llm_hallucinates and pure_llm_hallucinates and not gnn_llm_hallucinates:  # noqa
            final_prnt_str += "Prompt: '" + question + "'\n"
            final_prnt_str += "Label: '" + correct_answer + "'\n"
            if not skip_pretrained_LLM:
                final_prnt_str += "Untuned LLM Output: '" + untuned_llm_pred + "'\n"  # noqa
            final_prnt_str += "Tuned LLM Output: '" + pure_llm_pred + "'\n"
            final_prnt_str += "GNN+LLM Output: '" + gnn_llm_pred + "'\n"
            final_prnt_str += "\n" + "#" * 20 + "\n\n"
    if not skip_pretrained_LLM:
        print("Total untuned LLM Hallucinations:", untuned_llm_hallucin_sum)
    print("Total tuned LLM Hallucinations:", trained_llm_hallucin_sum)
    print("Total GNN+LLM Hallucinations:", gnn_llm_hallucin_sum)
    if not skip_pretrained_LLM:
        percent = 100.0 * round(
            1 - (gnn_llm_hallucin_sum / untuned_llm_hallucin_sum), 2)
        print(f"GNN reduces untuned LLM hallucinations by: ~{percent}%")
    tuned_percent = 100.0 * round(
        1 - (gnn_llm_hallucin_sum / trained_llm_hallucin_sum), 2)
    print(f"GNN reduces tuned LLM hallucinations by: ~{tuned_percent}%")
    print("Note: hallucinations detected by regex hence the ~")
    print("Potential instances where GNN solves the hallucinations of LLM:")
    print(final_prnt_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument(
        "--checkpointing", action="store_true",
        help="Use this flag to checkpoint each time a \
        new best val loss is achieved")

    args = parser.parse_args()
    # check if saved model
    retrain = True
    if path.exists("gnn_llm.pt") and path.exists("gnn_llm_eval_outs.pt"):
        print("Existing trained model found.")
        print("Would you like to retrain?")
        user_input = str(input("(y/n):")).lower()
        retrain = user_input == "y"
    else:
        retrain = True
    if retrain:
        since = time.time()
        prep_time, dataset, gnn_llm_eval_outs = train(
            since, args.epochs, args.gnn_hidden_channels, args.num_gnn_layers,
            args.batch_size, args.eval_batch_size, args.lr, get_loss,
            inference_step, checkpointing=args.checkpointing)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
        print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
    else:
        gnn_llm_eval_outs = torch.load("gnn_llm_eval_outs.pt")
        dataset = WebQSPDataset()
    print("Here's a demo showcasing how GNN reduces LLM hallucinations:")
    minimal_demo(gnn_llm_eval_outs, dataset, args.lr, args.epochs, get_loss,
                 inference_step, args.batch_size, args.eval_batch_size)
