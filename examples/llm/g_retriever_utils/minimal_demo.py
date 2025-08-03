"""This example implements the G-Retriever model
(https://arxiv.org/abs/2402.07630) using PyG.

G-Retriever significantly reduces hallucinations by 54% compared to the
stand-alone LLM baseline.

Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`
"""
import argparse
import gc
import math
import multiprocessing as mp
import re
import sys
import time
from os import path
from typing import Any, Callable, Dict, List, Type

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM

# NOTE: This used to be merged in the G-Retriever example.
# FIXME: Getting the demos working like before is a WIP
sys.path.append('..')
from g_retriever import (  # noqa: E402 # isort:skip
    compute_metrics, load_params_dict, save_params_dict,
)


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


def compute_n_parameters(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


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


# TODO: Merge with G-Retriever example and make sure changes still work
def train(
    num_epochs,
    hidden_channels,
    num_gnn_layers,
    batch_size,
    eval_batch_size,
    lr,
    checkpointing=False,
    tiny_llama=False,
    model=None,
    dataset=None,
    model_save_name=None,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    if model is None:
        gc.collect()
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

    if model_save_name is None:
        model_save_name = 'gnn_llm' if num_gnn_layers is not None else 'llm'

    model_save_name = 'gnn_llm' if num_gnn_layers != 0 else 'llm'
    if model_save_name == 'llm':
        model = llm

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
            for batch in val_loader:
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
    torch.cuda.reset_peak_memory_stats()

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
    for batch in test_loader:
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

    # Step 6 Post-processing & compute metrics
    compute_metrics(eval_output)
    print(f"Total Training Time: {time.time() - start_time:2f}s")
    save_params_dict(model, f'{model_save_name}.pt')
    torch.save(eval_output, f'{model_save_name}_eval_outs.pt')
    print("Done!")
    return prep_time, dataset, eval_output  # noqa: F821


def _eval_hallucinations_on_loader(outs, loader, eval_batch_size):
    model_save_list = []
    model_preds = []
    for out in outs:
        model_preds += out['pred']
    for i, batch in enumerate(loader):
        correct_answer = batch.label

        model_pred = model_preds[i * eval_batch_size:(i + 1) * eval_batch_size]
        model_hallucinates = detect_hallucinate(model_pred, correct_answer)
        model_save_list += [tup for tup in zip(model_pred, model_hallucinates)]
    return model_save_list


def benchmark_models(models: List[Type[nn.Module]], model_names: List[str],
                     model_kwargs: List[Dict[str, Any]], dataset: Dataset,
                     lr: float, epochs: int, batch_size: int,
                     eval_batch_size: int, loss_fn: Callable,
                     inference_fn: Callable, skip_LLMs: bool = True,
                     tiny_llama: bool = False, checkpointing: bool = True,
                     force: bool = False, root_dir='.'):
    """Utility function for creating a model benchmark for GRetriever that
    grid searches over hyperparameters.  Produces a DataFrame containing
    metrics for each model.

    Args:
        models (List[Type[nn.Module]]): Models to be benchmarked.
        model_names (List[str]): Name of save files for model checkpoints
        model_kwargs (List[Dict[str, Any]]): Parameters to use for each
            particular model.
        dataset (Dataset): Input dataset to train on.
        lr (float): Learning rate
        epochs (int): Number of epochs
        batch_size (int): Batch size for training
        eval_batch_size (int): Batch size for eval. Also determines
            hallucination detection concurrancy.
        loss_fn (Callable): Loss function
        inference_fn (Callable): Inference function
        skip_LLMs (bool, optional): Whether to skip LLM-only runs.
            Defaults to True.
        tiny_llama (bool, optional): Whether to use tiny llama as LLM.
            Defaults to False.
        checkpointing (bool, optional): Whether to checkpoint models.
            Defaults to True.
        force (bool, optional): Whether to rerun already existing results.
            Defaults to False.
        root_dir (str, optional): Dir to save results and checkpoints in.
            Defaults to '.'.
    """
    model_log: Dict[str, Dict[str, Any]] = dict()
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                        drop_last=False, pin_memory=True, shuffle=False)

    if not skip_LLMs:
        if tiny_llama:
            pure_llm = LLM(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
                num_params=1,
            )
        else:
            pure_llm = LLM(model_name="meta-llama/Llama-2-7b-chat-hf",
                           num_params=7)

        if force or not path.exists(root_dir + "/pure_llm_model_log.pt"):
            model_log["pure_llm"] = dict()

            pure_preds = []
            for batch in tqdm(loader):
                pure_llm_preds = pure_llm.inference(batch.question, batch.desc,
                                                    max_tokens=256)
                pure_preds += pure_llm_preds
            pure_preds = [{"pred": pred} for pred in pure_preds]

            model_log["pure_llm"]["preds"] = pure_preds
            model_log["pure_llm"]["hallucinates_list"] = \
                _eval_hallucinations_on_loader(pure_preds, loader,
                                               eval_batch_size)
            model_log["pure_llm"]["n_params"] = compute_n_parameters(pure_llm)
            torch.save(model_log["pure_llm"],
                       root_dir + "/pure_llm_model_log.pt")
        else:
            model_log["pure_llm"] = \
                torch.load(root_dir+"/pure_llm_model_log.pt")

        # LORA
        if force or not path.exists(root_dir + "/tuned_llm_model_log.pt"):
            model_log["tuned_llm"] = dict()
            since = time.time()
            gc.collect()
            prep_time, _, lora_eval_outs = train(since, epochs, None, None,
                                                 batch_size, eval_batch_size,
                                                 lr, loss_fn, inference_fn,
                                                 model=pure_llm,
                                                 dataset=dataset)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            e2e_time = round(time.time() - since, 2)
            model_log["tuned_llm"]["prep_time"] = prep_time
            model_log["tuned_llm"]["e2e_time"] = e2e_time
            model_log["tuned_llm"]["eval_output"] = lora_eval_outs
            print("E2E time (e2e_time) =", e2e_time, "seconds")
            print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")

            model_log["tuned_llm"]["hallucinates_list"] = \
                _eval_hallucinations_on_loader(lora_eval_outs, loader,
                                               eval_batch_size)
            model_log["tuned_llm"]["n_params"] = compute_n_parameters(pure_llm)
            torch.save(model_log["tuned_llm"],
                       root_dir + "/tuned_llm_model_log.pt")
        else:
            model_log["tuned_llm"] = \
                torch.load(root_dir+"/tuned_llm_model_log.pt")

        del pure_llm
        gc.collect()

    # All other models
    for name, Model, kwargs in zip(model_names, models, model_kwargs):
        model_log[name] = dict()
        train_model = True
        if path.exists(root_dir + f"/{name}.pt") and not force:
            print(f"Model {name} appears to already exist.")
            print("Would you like to retrain?")
            train_model = str(input("(y/n):")).lower() == "y"

        if train_model:
            since = time.time()
            gc.collect()
            model = Model(**kwargs)
            prep_time, _, model_eval_outs = train(
                since=since, num_epochs=epochs, hidden_channels=None,
                num_gnn_layers=None, batch_size=batch_size,
                eval_batch_size=eval_batch_size, lr=lr, loss_fn=loss_fn,
                inference_fn=inference_fn, checkpointing=checkpointing,
                tiny_llama=tiny_llama, dataset=dataset,
                model_save_name=root_dir + '/' + name, model=model)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            e2e_time = round(time.time() - since, 2)
            model_log[name]["prep_time"] = prep_time
            model_log[name]["e2e_time"] = e2e_time
            model_log[name]["eval_output"] = model_eval_outs
            print("E2E time (e2e_time) =", e2e_time, "seconds")
            print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
            model_log[name]["n_params"] = compute_n_parameters(model)
            del model
            gc.collect()
        else:
            model_eval_outs = torch.load(root_dir + f"/{name}_eval_outs.pt")

        # Calculate Hallucinations
        skip_hallucination_detection = False

        if path.exists(root_dir + f"/{name}_model_log.pt") and not force:
            print(f"Saved outputs for {name} have been found.")
            print("Would you like to redo?")
            user_input = str(input("(y/n):")).lower()
            skip_hallucination_detection = user_input != "y"

        if not skip_hallucination_detection:
            model_save_list = _eval_hallucinations_on_loader(
                model_eval_outs, loader, eval_batch_size)

            model_log[name]["hallucinates_list"] = model_save_list
            torch.save(model_log[name], root_dir + f"/{name}_model_log.pt")
        else:
            model_log[name]["hallucinates_list"] = \
                torch.load(
                    root_dir+f"/{name}_model_log.pt"
                    )["hallucinates_list"]

    hal_dict = {
        k: [tup[1] for tup in v["hallucinates_list"]]
        for (k, v) in model_log.items()
    }
    hallucinates_df = pd.DataFrame(hal_dict).astype(str)
    hallucinates_df = hallucinates_df.apply(pd.Series.value_counts).transpose()
    hallucinates_df['e2e_time'] = pd.Series(
        {k: v.get('e2e_time')
         for (k, v) in model_log.items()})
    hallucinates_df['n_params'] = pd.Series(
        {k: v.get('n_params')
         for (k, v) in model_log.items()})
    print(hallucinates_df)
    hallucinates_df.to_csv(root_dir + "/hallucinates_df.csv", index=False)


def minimal_demo(gnn_llm_eval_outs, dataset, lr, epochs, batch_size,
                 eval_batch_size, loss_fn, inference_fn,
                 skip_pretrained_LLM=False, tiny_llama=False):
    if not skip_pretrained_LLM:
        print("First comparing against a pretrained LLM...")
    # Step 1: Define a single batch size test loader
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    # batch size 1 loader for simplicity
    loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                        drop_last=False, pin_memory=True, shuffle=False)
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

            gnn_llm_pred = gnn_llm_preds[i * eval_batch_size:(i + 1) *
                                         eval_batch_size]
            gnn_llm_hallucinates = detect_hallucinate(gnn_llm_pred,
                                                      correct_answer)
            gnn_save_list += [
                tup for tup in zip(gnn_llm_pred, gnn_llm_hallucinates)
            ]

            if not skip_pretrained_LLM:
                # GNN+LLM only using 32 tokens to answer.
                # Allow more output tokens for untrained LLM
                pure_llm_pred = pure_llm.inference(batch.question, batch.desc,
                                                   max_tokens=256)
                pure_llm_hallucinates = detect_hallucinate(
                    pure_llm_pred, correct_answer)
            else:
                pure_llm_pred = [''] * len(gnn_llm_hallucinates)
                pure_llm_hallucinates = [False] * len(gnn_llm_hallucinates)
            untuned_llm_save_list += [
                tup for tup in zip(pure_llm_pred, pure_llm_hallucinates)
            ]

            for gnn_llm_hal, pure_llm_hal in zip(gnn_llm_hallucinates,
                                                 pure_llm_hallucinates):
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
        gnn_llm_pred, gnn_llm_hallucinates = list(
            zip(*gnn_save_list[i * eval_batch_size:(i + 1) * eval_batch_size]))
        untuned_llm_pred, untuned_llm_hallucinates = list(
            zip(*untuned_llm_save_list[i * eval_batch_size:(i + 1) *
                                       eval_batch_size]))
        pure_llm_pred = pure_llm_preds[i * eval_batch_size:(i + 1) *
                                       eval_batch_size]
        pure_llm_hallucinates = detect_hallucinate(pure_llm_pred,
                                                   correct_answer)
        for j in range(len(gnn_llm_pred)):
            if skip_pretrained_LLM:
                # we did not check the untrained LLM, so do not decide to demo
                # based on this.
                # HACK
                untuned_llm_hallucinates = {j: True}
            if gnn_llm_hallucinates[j] == "skip" or untuned_llm_hallucinates[
                    j] == "skip" or pure_llm_hallucinates[j] == "skip":
                continue
            trained_llm_hallucin_sum += int(pure_llm_hallucinates[j])
            if untuned_llm_hallucinates[j] and pure_llm_hallucinates[
                    j] and not gnn_llm_hallucinates[j]:  # noqa
                final_prnt_str += "Prompt: '" + question[j] + "'\n"
                final_prnt_str += "Label: '" + correct_answer[j] + "'\n"
                if not skip_pretrained_LLM:
                    final_prnt_str += "Untuned LLM Output: '" \
                        + untuned_llm_pred[j] + "'\n"  # noqa
                final_prnt_str += "Tuned LLM Output: '" + pure_llm_pred[
                    j] + "'\n"
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
    parser.add_argument(
        "--skip_pretrained_llm_eval", action="store_true",
        help="This flag will skip the evaluation of the pretrained LLM.")
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
