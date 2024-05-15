from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F

from .llm import get_llm_kwargs, get_mem_needed_for_llm


class SentenceTransformer(torch.nn.Module):
    def __init__(self, pretrained_repo: str, device: Optional[torch.device] = None, autocast_dtype: Optional[torch.dtype] = None, num_params: int = 7) -> None:
        super().__init__()
        print(f"inherit model weights from {pretrained_repo}")
        from transformers import AutoModel, AutoTokenizer
        self.autocast_dtype = autocast_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)
        if device is not None:
            self.llm = AutoModel.from_pretrained(pretrained_repo).to(device)
        else:
            if autocast_dtype is None:
                autocast_dtype = torch.bfloat16
            self.mem_needed = get_mem_needed_for_llm(num_params)
            kwargs = get_llm_kwargs(self.mem_needed)
            self.llm = AutoModel.from_pretrained(
            pretrained_repo, torch_dtype=autocast_dtype,
            low_cpu_mem_usage=True, **kwargs)
            self.llm_device = self.llm.device

    def mean_pooling(self, token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids: torch.Tensor,
                att_mask: torch.Tensor) -> torch.Tensor:
        with nullcontext() if autocast_dtype is None else torch.cuda.amp.autocast(dtype=self.llm_dtype):
            bert_out = self.llm(input_ids=input_ids,
                                       attention_mask=att_mask)

        # First element of model_output contains all token embeddings
        token_embeddings = bert_out[0]
        sentence_embeddings = self.mean_pooling(token_embeddings, att_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def text2embedding(model: SentenceTransformer,
                   text: List[str],
                   batch_size: Optional[int] = 256,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    try:
        encoding = model.tokenizer(text, padding=True, truncation=True,
                                   return_tensors="pt")
        data_len = encoding.input_ids.size(0)
        num_full_batches = data_len // batch_size
        all_embeddings_list = []

        # Iterate through batches
        if device is None:
            device = model.llm_device
        with torch.no_grad():
            left_ptr = 0
            for i in range(num_full_batches):
                # Forward pass
                embeddings = model(
                    input_ids=encoding.input_ids[left_ptr:left_ptr +
                                                 batch_size].to(device),
                    att_mask=encoding.attention_mask[left_ptr:left_ptr +
                                                     batch_size].to(device))
                left_ptr += batch_size
                # Append the embeddings to the list
                all_embeddings_list.append(embeddings)
            # final batch if len not divisible by batch_size
            if data_len % batch_size != 0:
                embeddings = model(
                    input_ids=encoding.input_ids[left_ptr:].to(device),
                    att_mask=encoding.attention_mask[left_ptr:].to(device))
                all_embeddings_list.append(embeddings)
        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings_list, dim=0).cpu()
    except:  # noqa
        print("text embedding failed, returning torch.zeros((0, 1024))...")
        return torch.zeros((0, 1024))

    return all_embeddings
