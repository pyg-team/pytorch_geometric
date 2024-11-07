import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    BatchEncoding = Dict

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '[/s]'
IGNORE_INDEX = -100
MAX_TXT_LEN = 512
MAX_NEW_TOKENS = 32
PAD_TOKEN_ID = 0
PADDING_SIDE = 'left'


def get_llm_kwargs(required_memory: int, dtype=torch.dtype) -> Dict[str, Any]:
    torch.cuda.empty_cache()

    gpu_memory: List[int] = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.mem_get_info(i)[0] // 1024**3)
        # Use the minimum number of GPUs to fit the LLM on.
        if sum(gpu_memory) >= required_memory:
            break

    if sum(gpu_memory) < required_memory:
        gpu_memory = []  # If not enough VRAM, use pure CPU.

    kwargs = dict(revision='main')
    if len(gpu_memory) > 0:
        kwargs['max_memory'] = {
            i: f'{memory}GiB'
            for i, memory in enumerate(gpu_memory)
        }
        kwargs['low_cpu_mem_usage'] = True
        kwargs['device_map'] = 'auto'
        kwargs['torch_dtype'] = dtype

    return kwargs


class LLM(torch.nn.Module):
    r"""A wrapper around a Large Language Model (LLM) from HuggingFace.

    model_name (str): The HuggingFace model name, *e.g.*, :obj:`"llama2"` or
        :obj:`"gemma"`.
    num_params (int): An integer representing how many parameters the
        HuggingFace model has, in billions. This is used to automatically
        allocate the correct number of GPUs needed, given the available GPU
        memory of your GPUs.
    dtype (torch.dtype, optional): The data type to use for the LLM.
        (default :obj: `torch.bloat16`)
    """
    def __init__(
        self,
        model_name: str,
        num_params: int,
        dtype=torch.bfloat16,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # A rough heuristic on GPU memory requirements, e.g., we found that
        # LLAMA2 (7B parameters) fits on a 85GB GPU.
        required_memory = 85 * num_params / 7
        kwargs = get_llm_kwargs(required_memory, dtype)

        print(f"Setting up '{model_name}' with configuration: {kwargs}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        self.tokenizer.pad_token_id = PAD_TOKEN_ID
        self.tokenizer.padding_side = PADDING_SIDE
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.word_embedding = self.llm.model.get_input_embeddings()

        if 'max_memory' not in kwargs:  # Pure CPU:
            warnings.warn("LLM is being used on CPU, which may be slow")
            self.device = torch.device('cpu')
            self.autocast_context = nullcontext()
        else:
            self.device = self.llm.device
            self.autocast_context = torch.amp.autocast('cuda', dtype=dtype)

    def _encode_inputs(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
    ) -> tuple:
        batch_size = len(question)
        questions = self.tokenizer(question, add_special_tokens=False)
        if context is not None:
            context = self.tokenizer(context, add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_token = self.tokenizer(
            BOS,
            add_special_tokens=False,
            return_tensors='pt',
        ).input_ids[0].to(self.device)
        bos_embeds = self.word_embedding(bos_token)
        pad_token = torch.tensor(self.tokenizer.pad_token_id,
                                 device=self.device)
        pad_embeds = self.word_embedding(pad_token).unsqueeze(0)
        return (batch_size, questions, context, eos_user_tokens, bos_embeds,
                pad_embeds)

    def _label_input_ids(
        self,
        i: int,
        label: BatchEncoding,
        eos_tokens: BatchEncoding,
    ) -> List[int]:
        label_input_ids = label.input_ids[i][:MAX_NEW_TOKENS]
        label_input_ids = label_input_ids + eos_tokens.input_ids
        return label_input_ids

    def _input_ids(
        self,
        i: int,
        context: BatchEncoding,
        question: BatchEncoding,
        eos_user_tokens: BatchEncoding,
    ) -> List[int]:
        input_ids: List[int] = []
        if context is not None:
            input_ids += context.input_ids[i][:MAX_TXT_LEN]
        input_ids += question.input_ids[i]
        input_ids += eos_user_tokens.input_ids
        return input_ids

    def _inputs_embeds(
        self,
        i: int,
        input_ids: List[int],
        bos_embeds: Tensor,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:
        inputs_embeds = self.word_embedding(
            torch.tensor(input_ids, device=self.device))

        to_cat = [bos_embeds]
        if embedding is not None and embedding[i] is not None:
            to_cat.append(embedding[i])
        to_cat.append(inputs_embeds)
        return torch.cat(to_cat, dim=0).to(self.device)

    def _append_embeds(
        self,
        inputs_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        label_input_ids: List[int] = None,
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        batch_inputs_embeds.append(inputs_embeds)
        batch_attention_mask.append([1] * inputs_embeds.size(0))
        if label_input_ids is not None:
            pad = inputs_embeds.size(0) - len(label_input_ids)
            label_input_ids = [IGNORE_INDEX] * pad + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        return batch_inputs_embeds, batch_attention_mask, batch_label_input_ids

    def _pad_embeds(
        self,
        pad_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        max_length = max([x.size(0) for x in batch_inputs_embeds])
        batch_size = len(batch_inputs_embeds)
        for i in range(batch_size):
            pad = max_length - batch_inputs_embeds[i].size(0)
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad, 1),
                batch_inputs_embeds[i],
            ])
            batch_attention_mask[i] = [0] * pad + batch_attention_mask[i]
            if batch_label_input_ids is not None:
                tmp = [IGNORE_INDEX] * pad + batch_label_input_ids[i]
                batch_label_input_ids[i] = tmp
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        label_input_ids = None
        if batch_label_input_ids is not None:
            label_input_ids = torch.tensor(batch_label_input_ids,
                                           device=self.device)
        return inputs_embeds, attention_mask, label_input_ids

    def _get_embeds(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        answer: Optional[List[str]] = None,
    ) -> tuple:
        (batch_size, question, context, eos_user_tokens, bos_embeds,
         pad_embeds) = self._encode_inputs(question, context)

        batch_label_input_ids = None
        if answer is not None:
            label = self.tokenizer(answer, add_special_tokens=False)
            eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
            batch_label_input_ids = []

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = self._input_ids(i, context, question, eos_user_tokens)
            if answer is not None:
                label_input_ids = self._label_input_ids(i, label, eos_tokens)
                input_ids += label_input_ids
            else:
                label_input_ids = None

            inputs_embeds = self._inputs_embeds(i, input_ids, bos_embeds,
                                                embedding)

            (
                batch_inputs_embeds,
                batch_attention_mask,
                batch_label_input_ids,
            ) = self._append_embeds(
                inputs_embeds,
                batch_inputs_embeds,
                batch_attention_mask,
                label_input_ids,
                batch_label_input_ids,
            )

        inputs_embeds, attention_mask, label_input_ids = self._pad_embeds(
            pad_embeds, batch_inputs_embeds, batch_attention_mask,
            batch_label_input_ids)

        return inputs_embeds, attention_mask, label_input_ids

    def forward(
        self,
        question: List[str],
        answer: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:
        r"""The forward pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
        """
        inputs_embeds, attention_mask, label_input_ids = self._get_embeds(
            question, context, embedding, answer)

        with self.autocast_context:
            outputs = self.llm(
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
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        max_tokens: Optional[int] = MAX_NEW_TOKENS,
    ) -> List[str]:
        r"""The inference pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
            max_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        inputs_embeds, attention_mask, _ = self._get_embeds(
            question, context, embedding)

        bos_token = self.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.autocast_context:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                bos_token_id=bos_token,
                max_new_tokens=max_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'
