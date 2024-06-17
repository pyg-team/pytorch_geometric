from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class PoolingStrategy(Enum):
    MEAN = 'mean'
    LAST = 'last'
    CLS = 'cls'


class SentenceTransformer(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling_strategy: Union[PoolingStrategy, str] = 'mean',
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = PoolingStrategy(pooling_strategy)

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        emb = out[0]  # First element contains all token embeddings.
        if self.pooling_strategy == PoolingStrategy.MEAN:
            emb = mean_pooling(emb, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.LAST:
            emb = last_pooling(emb, attention_mask)
        else:
            assert self.pooling_strategy == PoolingStrategy.CLS
            emb = emb[:, 0, :]

        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @property
    def device(self) -> torch.device:
        return next(iter(self.model.parameters())).device

    @torch.no_grad()
    def encode(
        self,
        text: List[str],
        batch_size: Optional[int] = None,
        output_device: Optional[torch.device] = None,
    ) -> Tensor:
        batch_size = len(text) if batch_size is None else batch_size

        embs: List[Tensor] = []
        for start in range(0, len(text), batch_size):
            token = self.tokenizer(
                text[start:start + batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt',
            )

            emb = self(
                input_ids=token.input_ids.to(self.device),
                attention_mask=token.attention_mask.to(self.device),
            ).to(output_device or 'cpu')

            embs.append(emb)

        return torch.cat(embs, dim=0) if len(embs) > 1 else embs[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(model_name={self.model_name})'


def mean_pooling(emb: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).expand(emb.size()).to(emb.dtype)
    return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def last_pooling(emb: Tensor, attention_mask: Tensor) -> Tensor:
    # Check whether language model uses left padding,
    # which is always used for decoder LLMs
    left_padding = attention_mask[:, -1].sum() == attention_mask.size(0)
    if left_padding:
        return emb[:, -1]

    seq_indices = attention_mask.sum(dim=1) - 1
    return emb[torch.arange(emb.size(0), device=emb.device), seq_indices]
