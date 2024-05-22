from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, emb: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).expand(emb.size()).to(emb.dtype)
        return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        emb = out[0]  # First element contains all token embeddings.
        emb = self.mean_pooling(emb, attention_mask)
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
