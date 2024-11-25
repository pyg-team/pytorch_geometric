from typing import List, Optional

import torch
from torch import Tensor

from torch_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS
from torch_geometric.utils import scatter


class GRetriever(torch.nn.Module):
    r"""The G-Retriever model from the `"G-Retriever: Retrieval-Augmented
    Generation for Textual Graph Understanding and Question Answering"
    <https://arxiv.org/abs/2402.07630>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        gnn (torch.nn.Module): The GNN to use.
        use_lora (bool, optional): If set to :obj:`True`, will use LORA from
            :obj:`peft` for training the LLM, see
            `here <https://huggingface.co/docs/peft/en/index>`_ for details.
            (default: :obj:`False`)
        mlp_out_channels (int, optional): The size of each graph embedding
            after projection. (default: :obj:`4096`)
        mlp_out_tokens (int, optional): Number of LLM prefix tokens to
            reserve for GNN output. (default: :obj:`1`)

    .. warning::
        This module has been tested with the following HuggingFace models

        * :obj:`llm_to_use="meta-llama/Llama-2-7b-chat-hf"`
        * :obj:`llm_to_use="google/gemma-7b"`

        and may not work with other models. See other models at `HuggingFace
        Models <https://huggingface.co/models>`_ and let us know if you
        encounter any issues.

    .. note::
        For an example of using :class:`GRetriever`, see
        `examples/llm/g_retriever.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/g_retriever.py>`_.
    """
    def __init__(
        self,
        llm: LLM,
        gnn: torch.nn.Module,
        use_lora: bool = False,
        mlp_out_channels: int = 4096,
        mlp_out_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.gnn = gnn.to(self.llm.device)

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm
        if use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            self.llm_generator = prepare_model_for_kbit_training(
                self.llm_generator)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ['q_proj', 'v_proj']
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias='none',
                task_type='CAUSAL_LM',
            )
            self.llm_generator = get_peft_model(self.llm_generator, config)

        mlp_hidden_channels = self.gnn.out_channels
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            torch.nn.Sigmoid(),
            torch.nn.Linear(mlp_hidden_channels,
                            mlp_out_channels * mlp_out_tokens),
            torch.nn.Unflatten(-1, (mlp_out_tokens, mlp_out_channels)),
        ).to(self.llm.device)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)

        out = self.gnn(x, edge_index, edge_attr=edge_attr)
        return scatter(out, batch, dim=0, reduce='mean')

    def forward(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        label: List[str],
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
    ):
        r"""The forward pass.

        Args:
            question (List[str]): The questions/prompts.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            label (List[str]): The answers/labels.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
                (default: :obj:`None`)
        """
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        # Handle case where theres more than one embedding for each sample
        xs = [x.squeeze(0) for x in xs]

        # Handle questions without node features:
        batch_unique = batch.unique()
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(question, additional_text_context, xs, label)

        with self.llm.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    @torch.no_grad()
    def inference(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
        max_out_tokens: Optional[int] = MAX_NEW_TOKENS,
    ):
        r"""The inference pass.

        Args:
            question (List[str]): The questions/prompts.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
                (default: :obj:`None`)
            max_out_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        # Handle case where theres more than one embedding for each sample
        xs = [x.squeeze(0) for x in xs]

        # Handle questions without node features:
        batch_unique = batch.unique()
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, additional_text_context, xs)

        bos_token = self.llm.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.llm.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                bos_token_id=bos_token,
                use_cache=True  # Important to set!
            )

        return self.llm.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  gnn={self.gnn},\n'
                f')')
