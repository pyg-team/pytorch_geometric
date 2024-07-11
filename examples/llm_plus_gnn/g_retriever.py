"""This example implements G-retriever using PyG.
Original Paper: https://arxiv.org/abs/2402.07630
“G-Retriever significantly reduces hallucinations
by 54% compared to the [LLM] baseline“.
Requirements on top of basic PyG:
`pip install datasets transformers pcst_fast sentencepiece tqdm accelerate`.
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
import torch.nn as nn
from typing import Callable, List, Type, Dict, Any

from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from torch_geometric.datasets import UpdatedWebQSPDataset, WebQSPDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.nlp import LLM
import multiprocessing as mp

def _detect_hallucinate(inp):
    pred, label = inp
    try:
        split_pred = pred.split('[/s]')[0].strip().split('|')
        correct_hit = len(re.findall(split_pred[0], label)) > 0
        correct_hit = correct_hit or any(
            [label_i in pred.lower() for label_i in label.split('|')])
        hallucination = not correct_hit
        return hallucination
    except:  # noqa
        return "skip"


def detect_hallucinate(pred_batch, label_batch):
    r"""An approximation for the unsolved task of detecting hallucinations.
    We define a hallucination as an output that contains no instances of
    acceptable label.
    """
    with mp.Pool(len(pred_batch)) as p:
        res = p.map(_detect_hallucinate, zip(pred_batch, label_batch))
    return res

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
          checkpointing=False, tiny_llama=False, model_save_name=None):
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
        gc.collect()
    elif not isinstance(dataset, Dataset) and callable(dataset):
        dataset = dataset()
        gc.collect()
    idx_split = dataset.split_idxs

    # Step 1: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Step 2: Build Model
    if model is None:
        gc.collect()
        if tiny_llama:
            model = GRetriever(
                llm_to_use="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
                num_llm_params=1,  # 1 Billion
                gnn_hidden_channels=hidden_channels,
                num_gnn_layers=num_gnn_layers,
                mlp_out_dim=2048,
            )
        else:
            model = GRetriever(llm_to_use="meta-llama/Llama-2-7b-chat-hf",
                               gnn_hidden_channels=hidden_channels,
                               num_gnn_layers=num_gnn_layers)
    if model_save_name is None:
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
            pred = inference_fn(model, batch, model_save_name)
            eval_data = {
                "pred": pred,
                "question": batch.question,
                "desc": batch.desc,
                "label": batch.label
            }
            eval_output.append(eval_data)
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

def _eval_hallucinations_on_loader(outs, loader, eval_batch_size):
    model_save_list = []
    model_preds = []
    for out in outs:
        model_preds += out['pred']
    for i, batch in enumerate(loader):
        correct_answer = batch.label

        model_pred = model_preds[i*eval_batch_size:(i+1)*eval_batch_size]
        model_hallucinates = detect_hallucinate(model_pred,
                                                correct_answer)
        model_save_list += [tup for tup in zip(model_pred, model_hallucinates)]
    return model_save_list


def benchmark_models(models: List[Type[nn.Module]], model_names: List[str], model_kwargs: List[Dict[str, Any]], dataset: Dataset, lr: float, epochs: int, batch_size: int, eval_batch_size: int, loss_fn: Callable, inference_fn: Callable, skip_LLMs: bool = True, tiny_llama: bool = False, checkpointing: bool = True):

    model_log: Dict[str, Dict[str, Any]] = dict()
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    loader = DataLoader(test_dataset, batch_size=eval_batch_size, drop_last=False,
                        pin_memory=True, shuffle=False)

    if not skip_LLMs:
        if tiny_llama:
            pure_llm = LLM(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
                num_params=1,
            )
        else:
            pure_llm = LLM(model_name="llama2-7b", num_params=7)

        if not path.exists("pure_llm_model_log.pt"):
            model_log["pure_llm"] = dict()
            
            pure_preds = []
            for batch in tqdm(loader):
                pure_llm_preds = pure_llm.inference(batch.question, batch.desc, max_tokens=256)
                pure_preds += pure_llm_preds
            pure_preds = [{"pred": pred} for pred in pure_preds]

            model_log["pure_llm"]["preds"] = pure_preds
            model_log["pure_llm"]["hallucinates_list"] = _eval_hallucinations_on_loader(pure_preds, loader, eval_batch_size)
            torch.save(model_log["pure_llm"], "pure_llm_model_log.pt")
        else:
            model_log["pure_llm"] = torch.load("pure_llm_model_log.pt")

        # LORA
        if not path.exists("tuned_llm_model_log.pt"):
            model_log["tuned_llm"] = dict()
            since = time.time()
            gc.collect()
            prep_time, _, lora_eval_outs = train(since, epochs, None, None, batch_size, eval_batch_size, lr, loss_fn, inference_fn, model=pure_llm, dataset=dataset)
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            gc.collect()
            e2e_time = round(time.time() - since, 2)
            model_log["tuned_llm"]["prep_time"] = prep_time
            model_log["tuned_llm"]["e2e_time"] = e2e_time
            model_log["tuned_llm"]["eval_output"] = lora_eval_outs
            print("E2E time (e2e_time) =", e2e_time, "seconds")
            print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")

            model_log["tuned_llm"]["hallucinates_list"] = _eval_hallucinations_on_loader(lora_eval_outs, loader, eval_batch_size)
            torch.save(model_log["tuned_llm"], "tuned_llm_model_log.pt")
        else:
            model_log["tuned_llm"] = torch.load("tuned_llm_model_log.pt")
        
        del pure_llm
        gc.collect()
     
    # All other models
    for name, Model, kwargs in zip(model_names, models, model_kwargs):
        model_log[name] = dict()
        train_model = True
        if path.exists(f"{name}.pt"):
            print(f"Model {name} appears to already exist.")
            print("Would you like to retrain?")
            train_model = str(input("(y/n):")).lower() == "y"
        
        if train_model:
            since = time.time()
            gc.collect()
            model = Model(**kwargs)
            prep_time, _, model_eval_outs = train(
                since=since, num_epochs=epochs, hidden_channels=None, num_gnn_layers=None,
                batch_size=batch_size, eval_batch_size=eval_batch_size, lr=lr, loss_fn=loss_fn,
                inference_fn=inference_fn, checkpointing=checkpointing,
                tiny_llama=tiny_llama, dataset=dataset, model_save_name=name, model=model)
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            gc.collect()
            e2e_time = round(time.time() - since, 2)
            model_log[name]["prep_time"] = prep_time
            model_log[name]["e2e_time"] = e2e_time
            model_log[name]["eval_output"] = model_eval_outs
            print("E2E time (e2e_time) =", e2e_time, "seconds")
            print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
            del model
            gc.collect()
        else:
            model_eval_outs = torch.load(f"{name}_eval_outs.pt")
        
        # Calculate Hallucinations
        skip_hallucination_detection = False

        if path.exists(f"{name}_model_log.pt"):
            print(f"Saved outputs for {name} have been found.")
            print("Would you like to redo?")
            user_input = str(input("(y/n):")).lower()
            skip_hallucination_detection = user_input != "y"


        if not skip_hallucination_detection:
            model_save_list = _eval_hallucinations_on_loader(model_eval_outs, loader, eval_batch_size)
 
            model_log[name]["hallucinates_list"] = model_save_list
            torch.save(model_log[name], f"{name}_model_log.pt")
        else:
            model_log[name]["hallucinates_list"] = torch.load(f"{name}_model_log.pt")["hallucinates_list"]
    
    hal_dict = {k: [tup[1] for tup in v["hallucinates_list"]] for (k, v) in model_log.items()}
    hallucinates_df = pd.DataFrame(hal_dict)
    for col in hallucinates_df.columns:
        print(f"Hallucinates for {col}:")
        print(hallucinates_df[col].value_counts())


def minimal_demo(gnn_llm_eval_outs, dataset, lr, epochs, batch_size,
                 eval_batch_size, loss_fn, inference_fn,
                 skip_pretrained_LLM=False, tiny_llama=False):
    if not skip_pretrained_LLM:
        print("First comparing against a pretrained LLM...")
    # Step 1: Define a single batch size test loader
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    # batch size 1 loader for simplicity
    loader = DataLoader(test_dataset, batch_size=eval_batch_size, drop_last=False,
                        pin_memory=True, shuffle=False)
    if tiny_llama:
        pure_llm = LLM(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
            num_params=1,
        )
    else:
        pure_llm = LLM(model_name="meta-llama/Llama-2-7b-chat-hf",
                       num_params=7)
    if path.exists("demo_save_dict.pt"):
        print("Saved outputs for the first step of the demo found.")
        print("Would you like to redo?")
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
        if skip_pretrained_LLM:
            print("Checking GNN+LLM for hallucinations...")
        else:
            print(
                "Checking pretrained LLM vs trained GNN+LLM for hallucinations..."  # noqa
            )
        for i, batch in enumerate(tqdm(loader)):
            question = batch.question
            correct_answer = batch.label
            if skip_pretrained_LLM:
                pure_llm_pred = None
                pure_llm_hallucinates = False
            else:
                # GNN+LLM only using 32 tokens to answer.
                # Allow more output tokens for untrained LLM
                pure_llm_pred = pure_llm.inference(batch.question, batch.desc,
                                                   max_tokens=256)
                pure_llm_hallucinates = detect_hallucinate(
                    pure_llm_pred, correct_answer)
            untuned_llm_save_list += [tup for tup in zip(pure_llm_pred, pure_llm_hallucinates)]

            gnn_llm_pred = gnn_llm_preds[i*eval_batch_size:(i+1)*eval_batch_size]
            gnn_llm_hallucinates = detect_hallucinate(gnn_llm_pred,
                                                      correct_answer)
            gnn_save_list += [tup for tup in zip(gnn_llm_pred, gnn_llm_hallucinates)]

            for gnn_llm_hal, pure_llm_hal in zip(gnn_llm_hallucinates, pure_llm_hallucinates):
                if gnn_llm_hal == "skip" or pure_llm_hal == "skip":  # noqa
                    # skipping when hallucination is hard to eval
                    continue
                gnn_llm_hallucin_sum += int(gnn_llm_hal)
                pure_llm_hallucin_sum += int(pure_llm_hal)
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
        _, _, pure_llm_eval_outputs = train(since, epochs, None, None,
                                            batch_size, eval_batch_size, lr,
                                            loss_fn, inference_fn,
                                            model=pure_llm, dataset=dataset)
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
    else:
        pure_llm_eval_outputs = torch.load("llm_eval_outs.pt")
    pure_llm_preds = []
    for out in pure_llm_eval_outputs:
        pure_llm_preds += out['pred']
    print("Final comparison between all models...")
    for i, batch in enumerate(tqdm(loader)):
        question = batch.question
        correct_answer = batch.label
        gnn_llm_pred, gnn_llm_hallucinates = list(zip(*gnn_save_list[i*eval_batch_size:(i+1)*eval_batch_size]))
        untuned_llm_pred, untuned_llm_hallucinates = list(zip(*untuned_llm_save_list[i*eval_batch_size:(i+1)*eval_batch_size]))
        if gnn_llm_hallucinates == "skip" or untuned_llm_hallucinates == "skip":  # noqa
            continue
        pure_llm_pred = pure_llm_preds[i*eval_batch_size:(i+1)*eval_batch_size]
        pure_llm_hallucinates = detect_hallucinate(pure_llm_pred,
                                                   correct_answer)
        for j in range(len(gnn_llm_pred)):
            if gnn_llm_hallucinates[j] == "skip" or untuned_llm_hallucinates[j] == "skip" or pure_llm_hallucinates[j] == "skip":
                continue
            trained_llm_hallucin_sum += int(pure_llm_hallucinates[j])
            if skip_pretrained_LLM:
                # we did not check the untrained LLM, so do not decide to demo
                # based on this.
                untuned_llm_hallucinates = True
            if untuned_llm_hallucinates[j] and pure_llm_hallucinates[j] and not gnn_llm_hallucinates[j]:  # noqa
                final_prnt_str += "Prompt: '" + question[j] + "'\n"
                final_prnt_str += "Label: '" + correct_answer[j] + "'\n"
                if not skip_pretrained_LLM:
                    final_prnt_str += "Untuned LLM Output: '" + untuned_llm_pred[j] + "'\n"  # noqa
                final_prnt_str += "Tuned LLM Output: '" + pure_llm_pred[j] + "'\n"
                final_prnt_str += "GNN+LLM Output: '" + gnn_llm_pred[j] + "'\n"
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
    parser.add_argument(
        "--tiny_llama", action="store_true",
        help="This example uses LLAMA2 (7B) by default. \
        This flag will run the example with TinyLLAMA (1B).")
    parser.add_argument(
        "--skip_pretrained_llm_eval", action="store_true",
        help="This flag will skip the evaluation of the pretrained LLM.")
    parser.add_argument(
        "--updated_qsp", action="store_true",
        help="This enables the updated and inproved WebQSP dataloader.")

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
        dataset = UpdatedWebQSPDataset if args.updated_qsp else None
        since = time.time()
        prep_time, dataset, gnn_llm_eval_outs = train(
            since, args.epochs, args.gnn_hidden_channels, args.num_gnn_layers,
            args.batch_size, args.eval_batch_size, args.lr, get_loss,
            inference_step, checkpointing=args.checkpointing,
            tiny_llama=args.tiny_llama, dataset=dataset)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
        print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
    else:
        gnn_llm_eval_outs = torch.load("gnn_llm_eval_outs.pt")
        dataset = WebQSPDataset(
        ) if not args.updated_qsp else UpdatedWebQSPDataset()
    print("Here's a demo showcasing how GNN reduces LLM hallucinations:")
    minimal_demo(gnn_llm_eval_outs, dataset, args.lr, args.epochs,
                 args.batch_size, args.eval_batch_size, get_loss,
                 inference_step,
                 skip_pretrained_LLM=args.skip_pretrained_llm_eval,
                 tiny_llama=args.tiny_llama)
