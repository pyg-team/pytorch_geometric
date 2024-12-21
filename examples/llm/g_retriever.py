"""This example implements the G-Retriever model
(https://arxiv.org/abs/2402.07630) using PyG.
G-Retriever significantly reduces hallucinations by 54% compared to the
stand-alone LLM baseline.
Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`
Example repo for integration with Neo4j Graph DB:
https://github.com/neo4j-product-examples/neo4j-gnn-llm-example
"""
# Import necessary libraries and modules
import argparse
import gc
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

# Define a function to compute evaluation metrics
def compute_metrics(eval_output):
    """
    Compute evaluation metrics (Hit, Precision, Recall, F1) from the output of the inference step.

    Args:
        eval_output (list): List of dictionaries containing prediction and label information.
    """
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    all_hit = []
    all_precision = []
    all_recall = []
    all_f1 = []

    # Loop through each prediction and label pair
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        try:
            # Extract the first prediction (split by '[/s]') and label
            pred = pred.split('[/s]')[0].strip().split('|')
            label = label.split('|')

            # Compute hit, precision, recall, and F1 score
            hit = len(re.findall(pred[0], label)) > 0
            matches = set(pred).intersection(set(label))
            precision = len(matches) / len(set(pred))
            recall = len(matches) / len(set(label))
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            # Append computed scores to lists
            all_hit.append(hit)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        except Exception as e:
            # Handle exceptions (e.g., when label or prediction is empty)
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print(f'Exception: {e}')
            print('------------------')

    # Compute average scores
    hit = sum(all_hit) / len(all_hit)
    precision = sum(all_precision) / len(all_precision)
    recall = sum(all_recall) / len(all_recall)
    f1 = sum(all_f1) / len(all_f1)

    # Print average scores
    print(f'Hit: {hit:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

# Define a function to save model parameters to a file
def save_params_dict(model, save_path):
    """
    Save model parameters to a file.

    Args:
        model (torch.nn.Module): Model to save.
        save_path (str): Path to save the model parameters.
    """
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)

# Define a function to load model parameters from a file
def load_params_dict(model, save_path):
    """
    Load model parameters from a file.

    Args:
        model (torch.nn.Module): Model to load.
        save_path (str): Path to load the model parameters from.
    """
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model

# Define a function to compute the loss for a given model and batch
def get_loss(model, batch, model_save_name):
    """
    Compute the loss for a given model and batch.

    Args:
        model (torch.nn.Module): Model to compute loss for.
        batch (torch_geometric.data.Data): Batch to compute loss for.
        model_save_name (str): Name of the model (used to determine loss computation).

    Returns:
        loss (torch.Tensor): Computed loss.
    """
    if model_save_name == 'llm':
        return model(batch.question, batch.label, batch.desc)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)

# Define a function to perform inference for a given model and batch
def inference_step(model, batch, model_save_name):
    """
    Perform inference for a given model and batch.

    Args:
        model (torch.nn.Module): Model to perform inference with.
        batch (torch_geometric.data.Data): Batch to perform inference on.
        model_save_name (str): Name of the model (used to determine inference computation).

    Returns:
        pred (torch.Tensor): Predicted output.
    """
    if model_save_name == 'llm':
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)

# Define the training loop
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
    """
    Train the model for a specified number of epochs.

    Args:
        num_epochs (int): Number of epochs to train for.
        hidden_channels (int): Number of hidden channels for the GNN model.
        num_gnn_layers (int): Number of layers for the GNN model.
        batch_size (int): Batch size for training.
        eval_batch_size (int): Batch size for evaluation.
        lr (float): Learning rate for the optimizer.
        checkpointing (bool): Whether to save model checkpoints. Default: False.
        tiny_llama (bool): Whether to use the tiny LLaMA model. Default: False.
    """
    # Define the learning rate schedule
    def adjust_learning_rate(param_group, LR, epoch):
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

    # Initialize training start time
    start_time = time.time()

    # Load dataset and create data loaders
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', '..', 'data', 'WebQSPDataset')
    train_dataset = WebQSPDataset(path, split='train')
    val_dataset = WebQSPDataset(path, split='val')
    test_dataset = WebQSPDataset(path, split='test')

    # Set random seed for reproducibility
    seed_everything(42)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize GNN model
    gnn = GAT(
        in_channels=1024,
        hidden_channels=hidden_channels,
        out_channels=1024,
        num_layers=num_gnn_layers,
        heads=4,
    )

    # Initialize LLaMA model
    if tiny_llama:
        llm = LLM(
            model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
            num_params=1,
        )
        model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
    else:
        llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf', num_params=7)
        model = GRetriever(llm=llm, gnn=gnn)

    # Set model name for saving and loading
    model_save_name = 'gnn_llm' if num_gnn_layers != 0 else 'llm'
    if model_save_name == 'llm':
        model = llm

    # Initialize optimizer and learning rate schedule
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))
    grad_steps = 2

    # Initialize best epoch and validation loss
    best_epoch = 0
    best_val_loss = float('inf')

    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize epoch loss
        epoch_loss = 0

        # Print preparation time
        if epoch == 0:
            print(f"Total Preparation Time: {time.time() - start_time:2f}s")
            start_time = time.time()
            print("Training beginning...")

        # Create progress bar for training
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)

        # Train on each batch
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = get_loss(model, batch, model_save_name)
            loss.backward()

            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            # Update learning rate schedule
            if (step + 1) % grad_steps == 0:
                lr = adjust_learning_rate(optimizer.param_groups[0], lr,
                                         step / len(train_loader) + epoch)

            # Update optimizer
            optimizer.step()
            epoch_loss = epoch_loss + float(loss)

            # Update learning rate schedule
            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']

        # Print training loss
        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')

        # Evaluate on validation set
        val_loss = 0
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")

        # Save model checkpoint if validation loss improves
        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(model, f'{model_save_name}_best_val_loss_ckpt.pt')

    # Clean up memory
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Load best model checkpoint if checkpointing is enabled
    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{model_save_name}_best_val_loss_ckpt.pt',
        )

    # Evaluate on test set
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

    # Compute evaluation metrics
    compute_metrics(eval_output)

    # Print total training time
    print(f"Total Training Time: {time.time() - start_time:2f}s")

    # Save final model and evaluation output
    save_params_dict(model, f'{model_save_name}.pt')
    torch.save(eval_output, f'{model_save_name}_eval_outs.pt')

if __name__ == '__main__':
    # Parse command-line arguments
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

    # Print total time
    start_time = time.time()

    # Train the model
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

    # Print total time
    print(f"Total Time: {time.time() - start_time:2f}s")