from typing import Optional, Union

import torch
from torch import Tensor


class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        from transformers import SwinConfig, SwinModel

        self.config = SwinConfig.from_pretrained(model_name)
        self.model = SwinModel(self.config)

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        output_device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        return self.model(images).last_hidden_state.to(output_device)

    @property
    def device(self) -> torch.device:
        return next(iter(self.model.parameters())).device

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(model_name={self.model_name})'
