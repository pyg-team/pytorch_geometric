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
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch_geometric
from torch_geometric import seed_everything
from torch_geometric.data import Batch, DataLoader
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.utils import scatter

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


def get_llm_kwargs():
    assert torch.cuda.is_available(), "GPU needed!"
    avail_gpus = torch.cuda.device_count()
    kwargs = {
        "revision": "main",
    }
    max_mem_dict = {}
    avail_mem_dict = {}
    mem_total = 0
    gpus_2_use_4_llm = 0
    for i in range(avail_gpus):
        available_mem = int(torch.cuda.mem_get_info(0)[0] // 1024**3)
        mem_total += available_mem
        avail_mem_dict[i] = available_mem
        gpus_2_use_4_llm += 1
        # We want to use the minimum number of GPUs that LLM can fit on
        # this is to minimize the need for interGPU communications
        # >= 75 GB VRAM in total is recommended
        if mem_total >= 75:
            break

    for i in range(gpus_2_use_4_llm):
        max_mem_dict[i] = str(avail_mem_dict[i]) + "GiB"
    kwargs["max_memory"] = max_mem_dict
    kwargs["device_map"] = "auto"
    return kwargs


class LLAMA2(nn.Module):
    # Pure LLAMA2 LLM module for demo
    def __init__(self):
        super().__init__()
        print('Loading LLAMA')
        kwargs = get_llm_kwargs()
        print("Setting up LLAMA w/ kwargs =", kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(llama2_str_name,
                                                       use_fast=False)
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.padding_side = padding_side
        self.llm = AutoModelForCausalLM.from_pretrained(
            llama2_str_name, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, **kwargs)
        self.llm_device = self.llm.device
        self.word_embedding = self.llm.model.get_input_embeddings()

    def encode_inputs(self, samples: Batch):
        batch_size = len(samples['question'])
        questions = self.tokenizer(samples["question"],
                                   add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"],
                                      add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False,
                           return_tensors='pt').input_ids[0].to(
                               self.llm_device))
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(
                self.llm_device)).unsqueeze(0)
        return batch_size, questions, descriptions, eos_user_tokens, bos_embeds, pad_embeds

    def inference(self, samples: Batch):
        batch_size, questions, descriptions, eos_user_tokens, bos_embeds, pad_embeds = self.encode_inputs(
            samples)
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.llm_device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }


class GAT_LLAMA(nn.Module):
    def __init__(self, hidden_channels: int, num_gnn_layers: int):
        super().__init__()

        self.llama2 = LLAMA2()

        print("Training LLAMA with LORA!")
        self.llm = self.llama2.llm
        self.llm_device = self.llama2.llm_device
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.tokenizer = self.llama2.tokenizer
        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, config)

        print('Finish loading LLAMA!')

        self.graph_encoder = torch_geometric.nn.models.GAT(
            in_channels=1024,
            out_channels=1024,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            heads=4,
        ).to(self.llm_device)
        self.projector = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.llm_device)

        self.word_embedding = self.llama2.word_embedding

    def encode_graphs(self, samples: Batch):
        x = samples.x.to(self.llm_device)
        edge_index = samples.edge_index.long().to(self.llm_device)
        edge_attr = samples.edge_attr.to(self.llm_device)
        n_embeds = self.graph_encoder(x, edge_index.long(), edge_attr)
        batch = samples.batch.to(self.llm_device)
        # mean pooling
        g_embeds = scatter(n_embeds, batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples: Batch):
        batch_size, questions, descriptions, eos_user_tokens, bos_embeds, pad_embeds = self.llama2.encode_inputs(
            samples)
        # encode labels
        labels = self.tokenizer(samples.label, add_special_tokens=False)
        # encode training specific special token
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        num_nodes_per_graph = samples.ptr[1:] - samples.ptr[:-1]
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[
                i][:max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            to_cat = [bos_embeds]
            if num_nodes_per_graph[i] != 0:
                to_cat.append(graph_embeds[i].unsqueeze(0))
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat(to_cat, dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX
                               ] * (inputs_embeds.shape[0] -
                                    len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[
                i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.llm_device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(
            self.llm_device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    def inference(self, samples: Batch):
        batch_size, questions, descriptions, eos_user_tokens, bos_embeds, pad_embeds = self.llama2.encode_inputs(
            samples)
        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            inputs_embeds = torch.cat(
                [bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds],
                dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.llm_device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


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
    model = GAT_LLAMA(hidden_channels, num_gnn_layers)

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
    torch.save(model, "gat_llama.pt")
    return prep_time, dataset, model


def minimal_demo(model, dataset):
    # Step 1: Define a single batch size test loader
    idx_split = dataset.split_idxs
    test_dataset = [dataset[i] for i in idx_split['test']]
    # batch size 1 loader for simplicity
    loader = DataLoader(test_dataset, batch_size=1, drop_last=False,
                        pin_memory=True, shuffle=False)
    # define the pure pretrained LLM
    pure_llm = LLAMA2()

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
            # skipping since hard to evaluate if the answer's are hallucinations
            continue
        gnn_llm_hallucin_sum += bool(gnn_llm_hallucinates)
        pure_llm_hallucin_sum += bool(pure_llm_hallucinates)
        # showcase LLM hallucinations solved by GNN
        if pure_llm_hallucinates and not gnn_llm_hallucinates:
            final_prnt_str += "Question: " + question + "\n"
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
    print("Instances where GNN solves the hallucinations of Pure LLMs:")
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
        print("E2E time (e2e_time) =", e2e_time)
        print("E2E time minus Prep Time =", e2e_time - prep_time)
    else:
        model = torch.load("gat_llama.pt")
        dataset = WebQSPDataset()
    print(
        "Would you like a minimal demo showcasing how GNN+LLM can solve LLM hallucinations?"
    )
    user_input = str(input("(y/n):")).lower()
    if user_input == "y":
        minimal_demo(model, dataset)
