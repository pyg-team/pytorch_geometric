"""This example provides helper functions for using the G-retriever model
(https://arxiv.org/abs/2402.07630) in PyG.

Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`


Example blog showing 2x accuracy over agentic graphRAG on real medical data
(integration with Neo4j Graph DB):
https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/

https://github.com/neo4j-product-examples/neo4j-gnn-llm-example

See examples/llm/txt2kg_rag.py for e2e pipeline in PyG including:
- KG Creation
- Subgraph Retrieval
- GNN+LLM Finetuning
- Testing
- LLM Judge Eval

"""
import math

import torch
from torch import Tensor


def adjust_learning_rate(param_group: dict, LR: float, epoch: int,
                         num_epochs: int):
    """Decay learning rate with half-cycle cosine after warmup.

    Args:
        param_group (dict): Parameter group.
        LR (float): Learning rate.
        epoch (int): current epoch
        num_epochs (int): total epochs
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


def get_loss(model, batch, model_save_name="gnn+llm") -> Tensor:
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
            batch.question,  # ["list", "of", "questions", "here"]
            batch.x,  # [num_nodes, num_features]
            batch.edge_index,  # [2, num_edges]
            batch.batch,  # which node belongs to which batch index
            batch.label,  # list answers (labels)
            batch.edge_attr,  # edge attributes
            batch.desc  # list of text graph descriptions
        )


def inference_step(model, batch, model_save_name="gnn+llm",
                   max_out_tokens=128):
    """Performs inference on a given batch of data using the provided model.

    Args:
        model (nn.Module): The model to use for inference.
        batch: The batch of data to process.
        model_save_name (str): The name of the model (e.g. 'llm').
        max_out_tokens (int): The maximum number of tokens
            for our model to output.

    Returns:
        The output of the inference step.
    """
    # Check the type of model being used to determine the input arguments
    if model_save_name == 'llm':
        # Perform inference on the question and textual graph description
        return model.inference(batch.question, batch.desc,
                               max_out_tokens=max_out_tokens)
    else:  # (GNN+LLM)
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc,
                               max_out_tokens=max_out_tokens)
