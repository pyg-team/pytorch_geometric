"""This example implements the G-Retriever model
(https://arxiv.org/abs/2402.07630) using PyG.

G-Retriever significantly reduces hallucinations by 54% compared to the
stand-alone LLM baseline.

Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`

Example repo for integration with Neo4j Graph DB:
https://github.com/neo4j-product-examples/neo4j-gnn-llm-example
Example blog showing 2x accuracy over LLM on real medical data:
https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/
"""
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
from torch_geometric.datasets import CWQDataset, WebQSPDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM


def compute_metrics(eval_output):
    """Compute evaluation metrics (hit, precision, recall, F1).

    Parameters:
    eval_output (list): List of dictionaries containing prediction output.

    Returns:
    None (prints metrics to console)
    """
    # Concatenate prediction output into a single DataFrame
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # Initialize lists to store metrics
    all_hit = []  # Boolean values indicating whether prediction matches label
    all_precision = []  # List of precision values
    all_recall = []  # List of recall values
    all_f1 = []  # List of F1 values

    # Iterate over prediction-label pairs
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        try:
            # Preprocess prediction string
            pred = pred.split('[/s]')[0].strip().split('|')

            # Check if prediction matches label
            hit = re.findall(pred[0], label)
            all_hit.append(len(hit) > 0)

            # Compute precision, recall, and F1
            label = label.split('|')
            matches = set(pred).intersection(set(label))
            precision = len(matches) / len(set(pred))
            recall = len(matches) / len(set(label))

            # Handle division by zero
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            # Store metrics
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        except Exception as e:
            # Handle exceptions by printing error message and skipping
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print(f'Exception: {e}')
            print('------------------')

    # Compute average metrics
    hit = sum(all_hit) / len(all_hit)
    precision = sum(all_precision) / len(all_precision)
    recall = sum(all_recall) / len(all_recall)
    f1 = sum(all_f1) / len(all_f1)

    # Print metrics to console
    print(f'Hit@1: {hit:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')


def save_params_dict(model, save_path):
    """Saves a model's parameters, excluding non-trainable weights.

    Args:
        model (torch.nn.Module): The model to save parameters from.
        save_path (str): The path to save the parameters to.
    """
    # Get the model's state dictionary, which contains all its parameters
    state_dict = model.state_dict()

    # Create a dictionary mapping parameter names to their requires_grad status
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }

    # Remove non-trainable parameters from the state dictionary
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient

    # Save the filtered state dictionary to the specified path
    torch.save(state_dict, save_path)


def load_params_dict(model, save_path):
    # Load the saved model parameters from the specified file path
    state_dict = torch.load(save_path)

    # Update the model's parameters with the loaded state dictionary
    model.load_state_dict(state_dict)

    # Return the model with updated parameters
    return model


def get_loss(model, batch, model_save_name: str) -> Tensor:
    """Compute the loss for a given model and batch of data.

    Args:
        model: The model to compute the loss for.
        batch: The batch of data to compute the loss for.
        model_save_name: The name of the model being used (e.g. 'llm').

    Returns:
        Tensor: The computed loss.
    """
    # Check the type of model being used to determine the input arguments
    if model_save_name == 'llm':
        # For LLM models
        return model(batch.question, batch.label, batch.desc)
    else:  # (GNN+LLM)
        return model(
            batch.question,
            batch.x,  # node features
            batch.edge_index,  # edge indices
            batch.batch,  # batch indices
            batch.label,  # answers (labels)
            batch.edge_attr,  # edge attributes
            batch.desc  # description
        )


def inference_step(model, batch, model_save_name):
    """Performs inference on a given batch of data using the provided model.

    Args:
        model (nn.Module): The model to use for inference.
        batch: The batch of data to process.
        model_save_name (str): The name of the model (e.g. 'llm').

    Returns:
        The output of the inference step.
    """
    # Check the type of model being used to determine the input arguments
    if model_save_name == 'llm':
        # Perform inference on the question and textual graph description
        return model.inference(batch.question, batch.desc)
    else:  # (GNN+LLM)
        return model.inference(
            batch.question,
            batch.x,  # node features
            batch.edge_index,  # edge indices
            batch.batch,  # batch indices
            batch.edge_attr,  # edge attributes
            batch.desc  # description
        )


def train(
        num_epochs,  # Total number of training epochs
        hidden_channels,  # Number of hidden channels in GNN
        num_gnn_layers,  # Number of GNN layers
        batch_size,  # Training batch size
        eval_batch_size,  # Evaluation batch size
        lr,  # Initial learning rate
        llm_model_name,  # `transformers` model name
        checkpointing=False,  # Whether to checkpoint model
        cwq=False,  # Whether to train on the CWQ dataset
        tiny_llama=False,  # Whether to use tiny LLaMA model
):
    """Train a GNN+LLM model on WebQSP or CWQ dataset.

    Args:
        num_epochs (int): Total number of training epochs.
        hidden_channels (int): Number of hidden channels in GNN.
        num_gnn_layers (int): Number of GNN layers.
        batch_size (int): Training batch size.
        eval_batch_size (int): Evaluation batch size.
        lr (float): Initial learning rate.
        llm_model_name (str): The name of the LLM to use.
        checkpointing (bool, optional): Whether to checkpoint model.
            Defaults to False.
        cwq (bool, optional): Whether to train on the CWQ dataset
            instead of WebQSP.
        tiny_llama (bool, optional): Whether to use tiny LLaMA model.
            Defaults to False.

    Returns:
        None
    """
    def adjust_learning_rate(param_group, LR, epoch):
        """Decay learning rate with half-cycle cosine after warmup.

        Args:
            param_group (dict): Parameter group.
            LR (float): Learning rate.
            epoch (int): Current epoch.

        Returns:
            float: Adjusted learning rate.
        """
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

    # Start training time
    start_time = time.time()

    # Load dataset and create data loaders
    path = osp.dirname(osp.realpath(__file__))
    if not cwq:
        path = osp.join(path, '..', '..', 'data', 'WebQSPDataset')
        train_dataset = WebQSPDataset(path, split='train')
        val_dataset = WebQSPDataset(path, split='val')
        test_dataset = WebQSPDataset(path, split='test')
    else:
        path = osp.join(path, '..', '..', 'data', 'CWQDataset')
        train_dataset = CWQDataset(path, split='train')
        val_dataset = CWQDataset(path, split='val')
        test_dataset = CWQDataset(path, split='test')

    seed_everything(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Create GNN model
    gnn = GAT(
        in_channels=1024,
        hidden_channels=hidden_channels,
        out_channels=1024,
        num_layers=num_gnn_layers,
        heads=4,
    )

    # Create LLaMA model
    if tiny_llama:
        llm = LLM(model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1', )
    else:
        llm = LLM(model_name=llm_model_name)

    # Set model save name
    model_save_name = 'gnn_llm' if num_gnn_layers != 0 else 'llm'
    if model_save_name == 'llm':
        model = llm
    else:
        model = GRetriever(llm=llm, gnn=gnn,
                           mlp_out_channels=llm.word_embedding.embedding_dim)

    # Create optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))

    # Initialize best epoch and best validation loss
    best_epoch = 0
    best_val_loss = float('inf')

    # Train model
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

            if (step + 1) % 2 == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)

            optimizer.step()
            epoch_loss = epoch_loss + float(loss)

            if (step + 1) % 2 == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')

        # Evaluate model
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

    # Clean up memory
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Load best checkpoint if necessary
    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{model_save_name}_best_val_loss_ckpt.pt',
        )

    # Evaluate model on test set
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

    # Compute metrics
    compute_metrics(eval_output)

    # Print final training time
    print(f"Total Training Time: {time.time() - start_time:2f}s")

    # Save model and evaluation output
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
    parser.add_argument('--cwq', action='store_true')
    parser.add_argument('--tiny_llama', action='store_true')
    parser.add_argument('--llm_model_name', type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    start_time = time.time()
    train(
        args.epochs,
        args.gnn_hidden_channels,
        args.num_gnn_layers,
        args.batch_size,
        args.eval_batch_size,
        args.lr,
        args.llm_model_name,
        checkpointing=args.checkpointing,
        tiny_llama=args.tiny_llama,
        cwq=args.cwq,
    )
    print(f"Total Time: {time.time() - start_time:2f}s")
