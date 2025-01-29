from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class PoolingStrategy(Enum):
    MEAN = 'mean'
    LAST = 'last'
    CLS = 'cls'
    LAST_HIDDEN_STATE = 'last_hidden_state'


class NIMSentenceTransformer():
    """Class for converting strings into vectors for NLP tasks.
    Uses NVIDIA Inference MicroServices.
    """
    def __init__(
        self,
        model_name: str,
        NIM_KEY: str,
    ) -> None:
        """Initializes the class with the given parameters.

        Args:
        - model_name: Name of the machine learning model
        - NIM_KEY: NIM API Key.
        """
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NIM_KEY,
        )
        self.embedding_model_name = model_name

    def encode(
        self,
        text: List[str],
        batch_size: Optional[int] = None,
        output_device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """Converts list of strings into tensor of vectors

        Args:
        - text: List of strings to encode
        - batch_size: optional batch size for encoding
            Default: None
        - output_device: optional device for output tensor
            Default: None

        Returns: a PyTorch tensor containing the encoded data
        """
        is_empty = len(text) == 0
        # for downstream ease, embed empty text list as the "dummy" vector
        text = ['dummy'] if is_empty else text

        batch_size = len(text) if batch_size is None else batch_size

        embs: List[Tensor] = []
        for start in range(0, len(text), batch_size):
            batch = text[start:start + batch_size]
            print("batch=", batch)
            #batch = ["list", "of", "dummytext"]
            response = self.client.embeddings.create(
                input=batch, model=self.embedding_model_name,
                encoding_format="float", extra_body={
                    "input_type": "passage",
                    "truncate": "END"
                })
            embs.append(
                torch.tensor([[d.embedding for d in response.data]
                              ]).to(torch.float32).to(output_device))

        out = torch.cat(embs, dim=0) if len(embs) > 1 else embs[0]
        out = out[:0] if is_empty else out
        return out


class SentenceTransformer(torch.nn.Module):
    """Class for converting strings into vectors for NLP tasks.
    Generally freeze the weights
    Uses local LM via HuggingFace.
    """
    def __init__(
            self,
            model_name: str,  # Name of the machine learning model
            pooling_strategy: Union[
                PoolingStrategy,
                str] = 'mean',  # Strategy to apply to pool features
    ) -> None:
        """Initializes the class with the given parameters.

        Args:
        - model_name: Name of the machine learning model
        - pooling_strategy: Strategy to apply to pool features.
            (default: 'mean')
        """
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = PoolingStrategy(pooling_strategy)

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        emb = out[0]  # First element contains all token embeddings.
        if self.pooling_strategy == PoolingStrategy.MEAN:
            emb = mean_pooling(emb, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.LAST:
            emb = last_pooling(emb, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.LAST_HIDDEN_STATE:
            emb = out.last_hidden_state
        else:
            assert self.pooling_strategy == PoolingStrategy.CLS
            emb = emb[:, 0, :]

        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def get_input_ids(
        self,
        text: List[str],
        batch_size: Optional[int] = None,
        output_device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        is_empty = len(text) == 0
        text = ['dummy'] if is_empty else text

        batch_size = len(text) if batch_size is None else batch_size

        input_ids: List[Tensor] = []
        attention_masks: List[Tensor] = []
        for start in range(0, len(text), batch_size):
            token = self.tokenizer(
                text[start:start + batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            input_ids.append(token.input_ids.to(self.device))
            attention_masks.append(token.attention_mask.to(self.device))

        def _out(x: List[Tensor]) -> Tensor:
            out = torch.cat(x, dim=0) if len(x) > 1 else x[0]
            out = out[:0] if is_empty else out
            return out.to(output_device)

        return _out(input_ids), _out(attention_masks)

    @property
    def device(self) -> torch.device:
        return next(iter(self.model.parameters())).device

    @torch.no_grad()
    def encode(
        self,
        text: List[str],
        batch_size: Optional[int] = None,
        output_device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:  #
        """Converts list of strings into tensor of vectors

        Args:
        - text: List of strings to encode
        - batch_size: optional batch size for encoding
            Default: None
        - output_device: optional device for output tensor
            Default: None

        Returns: a PyTorch tensor containing the encoded data
        """
        is_empty = len(text) == 0
        # for downstream ease, embed empty text list as the "dummy" vector
        text = ['dummy'] if is_empty else text

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
            ).to(output_device)

            embs.append(emb)

        out = torch.cat(embs, dim=0) if len(embs) > 1 else embs[0]
        out = out[:0] if is_empty else out
        return out

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
