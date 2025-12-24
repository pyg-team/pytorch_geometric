#!/usr/bin/env python
import random

import numpy as np
import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer

from torch_geometric.contrib.nn.models.gprompt import GraphAdapter
from torch_geometric.data import Data

VERBOSE = True


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def build_toy_graph():
    # An undirected linear graph 0 - 1 - 2
    # (Two directed edges to represent each undirected edge)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],  # neighbors (dest)
            [1, 0, 2, 1],  # centers (src)
        ],
        dtype=torch.long,
    )

    texts = [
        "This is a course about graph neural networks.",
        "The goal is to better understand how to do machine learning over "
        "graphs.",
        "This project implements a paper to build better node features for "
        "text-attributed graphs.",
    ]

    data = Data(edge_index=edge_index)
    data.text = texts
    return data


def encode_texts_with_masks(tokenizer, model, texts, mlm_prob=0.15):
    """Randomly mask tokens with probability mlm_prob per node,
    store original token IDs in target_ids.

    Return:
        h_mask: [M, H]
        z: [N, H]  (CLS embeddings per node)
        target_ids: [M]
        node_ids: [M]
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_special_tokens_mask=True,
    )
    N, L = enc["input_ids"].shape
    mask_token_id = tokenizer.mask_token_id

    # Print out the decoded sequence as a sanity check
    # We see that the tokenizer automatically adds CLS and SEP, PAD tokens
    for i, seq in enumerate(enc["input_ids"]):
        vprint(f"Sanity check: decoded sequence {i} is", tokenizer.decode(seq))
    masked_input_ids, target_ids_list, node_ids_list = enc["input_ids"].clone(
    ), [], []
    special_mask = enc["special_tokens_mask"].to(torch.bool)

    masked_positions = (torch.bernoulli(
        torch.full(enc["input_ids"].shape, mlm_prob)).bool()) & ~special_mask
    for i in range(N):
        positions = masked_positions[i].nonzero(as_tuple=False).view(-1)
        for p in positions:
            target_ids_list.append(enc["input_ids"][i, p])
            node_ids_list.append(i)
            masked_input_ids[i, p] = mask_token_id

    # Print original tokens that are being masked.
    for i in range(enc["input_ids"].size(0)):
        vprint(f"Sentence {i}:")
        positions = masked_positions[i].nonzero(as_tuple=False).view(-1)
        for p in positions:
            orig_token_id = enc["input_ids"][i, p]
            vprint("  Masked token:", tokenizer.decode([orig_token_id]))

    if len(target_ids_list) == 0:
        raise ValueError(
            "MLM sampling produced zero masked tokens. Try a higher mlm_prob.")
    target_ids = torch.stack(target_ids_list, dim=0)  # [M]
    node_ids = torch.tensor(node_ids_list, dtype=torch.long)  # [M]
    with torch.no_grad():
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [N, L, H]

    # Get overall sentence embeddings (CLS tokens)
    z = last_hidden[:, 0, :]  # [N, H]

    # Collect hidden states at masked positions
    masked_states = []
    for i in range(N):
        positions = masked_positions[i].nonzero(as_tuple=False).view(-1)
        for p in positions:
            masked_states.append(last_hidden[i, p])

    h_mask = torch.stack(masked_states, dim=0)  # [M, H]

    return h_mask, z, target_ids, node_ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare input data (toy graph)
    data = build_toy_graph()
    edge_index = data.edge_index.to(device)
    vprint("Graph edges are", edge_index)

    # Load a tiny version of BERT (from https://arxiv.org/abs/1908.08962)
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name).to(device)
    lm.eval()

    # Get the hidden token embeddings and also the sentence (CLS) embeddings
    h_mask, z, target_ids, node_ids = encode_texts_with_masks(
        tokenizer, lm, data.text)
    h_mask = h_mask.to(device)  # [M, H]
    z = z.to(device)  # [N, H]
    target_ids = target_ids.to(device)
    node_ids = node_ids.to(device)

    hidden_channels = h_mask.size(-1)
    node_channels = z.size(-1)

    #    linear layer that maps adapter outputs to vocab logits.
    vocab_size = tokenizer.vocab_size

    adapter = GraphAdapter(
        hidden_channels=hidden_channels,
        node_channels=node_channels,
    ).to(device)

    # Note: We do NOT use the language model's final prediction layer (denoted
    # by f_LM in the paper). We instead define a fresh one for simplicity. In
    # the paper, this final prediction layer is also not explicitly fine-tuned
    # since it already carries a lot of pre-trained knowledge.
    predictor = nn.Linear(hidden_channels, vocab_size).to(device)
    optimizer = optim.Adam(
        list(adapter.parameters()) + list(predictor.parameters()),
        lr=1e-3,
    )

    # A short training loop to demonstrate the E2E flow. We will observe that
    # the loss quickly goes to 0, which we expect given that it is a toy
    # dataset and it should overfit quickly.
    adapter.train()
    predictor.train()

    num_steps = 10000
    for step in range(num_steps):
        optimizer.zero_grad()
        edge_logits, masked_idx = adapter(
            h_mask=h_mask,
            z=z,
            edge_index=edge_index,
            node_ids=node_ids,
            predictor=predictor,
        )

        loss = GraphAdapter.loss(
            edge_logits=edge_logits,
            masked_idx=masked_idx,
            target_ids=target_ids,
        )

        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            vprint(f"Step {step:02d} | loss = {loss.item():.4f}")

    print(f"Success! Final loss {loss} should be nearly 0")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    main()
