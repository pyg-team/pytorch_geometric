import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

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

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_name == 'llama2-7b':
            pretty_model_name = 'LLAMA2'
            model_name = 'meta-llama/Llama-2-7b-chat-hf'
        elif model_name == 'gemma':
            pretty_model_name = 'GEMMA'
            model_name = 'google/gemma-7b'
        else:
            pretty_model_name = model_name

        # A rough heuristic on GPU memory requirements, e.g., we found that
        # LLAMA2 (7B parameters) fits on a 85GB GPU.
        required_memory = 85 * num_params / 7
        kwargs = get_llm_kwargs(required_memory, dtype)

        print(f"Setting up '{pretty_model_name}' with configuration: {kwargs}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        self.tokenizer.pad_token_id = PAD_TOKEN_ID
        self.tokenizer.padding_side = PADDING_SIDE
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.word_embedding = self.llm.model.get_input_embeddings()

        if 'max_memory' not in kwargs:  # Pure CPU:
            self.llm_device = torch.device('cpu')
            self.autocast_context = nullcontext()
        else:
            self.llm_device = self.llm.device
            self.autocast_context = torch.cuda.amp.autocast(dtype=dtype)

    def _encode_inputs(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
    ) -> None:
        batch_size = len(question)
        questions = self.tokenizer(question, add_special_tokens=False)
        if context is not None:
            context = self.tokenizer(context, add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_token = self.tokenizer(
            BOS,
            add_special_tokens=False,
            return_tensors='pt',
        ).input_ids[0].to(self.llm_device)
        bos_embeds = self.word_embedding(bos_token)
        pad_token = torch.tensor(self.tokenizer.pad_token_id,
                                 device=self.llm_device)
        pad_embeds = self.word_embedding(pad_token).unsqueeze(0)
        return (batch_size, questions, context, eos_user_tokens, bos_embeds,
                pad_embeds)

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
                :obj:`context` or :obj:`rag_embeddings` should be used, not
                both. (default: :obj:`None`)
        """
        if context is not None and embedding is not None:
            warnings.warn("Using both 'context' and 'embedding' is a waste of "
                          "compute and memory")

        (batch_size, question, context, eos_user_tokens, bos_embeds,
         pad_embeds) = self._encode_inputs(question, context)

        label = self.tokenizer(answer, add_special_tokens=False)
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = label.input_ids[i][:MAX_NEW_TOKENS]
            label_input_ids += eos_tokens.input_ids  # Add EOS token.

            input_ids: List[int] = []
            if context is not None:
                input_ids += context.input_ids[i][:MAX_TXT_LEN]
            input_ids += question.input_ids[i]
            input_ids += eos_user_tokens.input_ids
            input_ids += label_input_ids

            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids, device=self.llm_device))

            to_cat = [bos_embeds]
            if embedding is not None:
                to_cat.append(embedding[i])
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat(to_cat, dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.size(0))
            label_input_ids = [IGNORE_INDEX] * (
                inputs_embeds.size(0) - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # Pad input embeddings:
        max_length = max([x.size(0) for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad = max_length - batch_inputs_embeds[i].size(0)
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad, 1),
                batch_inputs_embeds[i],
            ])
            batch_attention_mask[i] = [0] * pad + batch_attention_mask[i]
            batch_label_input_ids[i] = ([IGNORE_INDEX] * pad +
                                        batch_label_input_ids[i])

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask,
                                      device=self.llm_device)
        label_input_ids = torch.tensor(batch_label_input_ids,
                                       device=self.llm_device)

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
                :obj:`context` or :obj:`rag_embeddings` should be used, not
                both. (default: :obj:`None`)
            max_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        if context is not None and embedding is not None:
            warnings.warn("Using both 'context' and 'embedding' is a waste of "
                          "compute and memory")

        (batch_size, question, context, eos_user_tokens, bos_embeds,
         pad_embeds) = self._encode_inputs(question, context)

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids: List[int] = []
            if context is not None:
                input_ids = context.input_ids[i][:MAX_TXT_LEN]
            input_ids += question.input_ids[i]
            input_ids += eos_user_tokens.input_ids

            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids, device=self.llm_device))

            to_cat = [bos_embeds]
            if embedding is not None:
                to_cat.append(embedding[i])
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat(to_cat, dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.size(0))

        # Pad input embeddings:
        max_length = max([x.size(0) for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad = max_length - batch_inputs_embeds[i].size(0)
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad, 1),
                batch_inputs_embeds[i],
            ])
            batch_attention_mask[i] = [0] * pad + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask,
                                      device=self.llm_device)

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
