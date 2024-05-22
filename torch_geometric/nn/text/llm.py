from typing import List, Optional

import torch
import torch.nn as nn

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '[/s]'
IGNORE_INDEX = -100
llama2_str_name = "meta-llama/Llama-2-7b-chat-hf"
gemma_str_name = "google/gemma-7b"
max_txt_len = 512
max_new_tokens = 32
pad_token_id = 0
padding_side = 'left'


def get_llm_kwargs(mem_needed, autocast_dtype=torch.bfloat16):
    avail_gpus = torch.cuda.device_count()
    kwargs = {
        "revision": "main",
    }
    max_mem_dict = {}
    avail_mem_dict = {}
    mem_total = 0
    gpus_2_use_4_llm = 0
    for i in range(avail_gpus):
        available_mem = int(torch.cuda.mem_get_info(i)[0] // 1024**3)
        mem_total += available_mem
        avail_mem_dict[i] = available_mem
        gpus_2_use_4_llm += 1
        # We want to use the minimum number of GPUs that LLM can fit on
        # this is to minimize the need for interGPU communications
        if mem_total >= mem_needed:
            break
    cpu_offload = mem_total < mem_needed
    if not cpu_offload:
        for i in range(gpus_2_use_4_llm):
            max_mem_dict[i] = str(avail_mem_dict[i]) + "GiB"
        kwargs["max_memory"] = max_mem_dict
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = autocast_dtype
        kwargs["low_cpu_mem_usage"] = True

    return kwargs, cpu_offload


class LLM(nn.Module):
    r"""This module wraps a HuggingFace Transformer based model.

    model_name (str): A string representing the huggingface model you
        want to use. This module has been tested for 'llama2' and 'gemma'.
        Other huggingface transformer models should work if you pass the
        correct name, see huggingface.co for details. If any issues occur
        please file an issue on
        https://github.com/pyg-team/pytorch_geometric
        and assign to puririshi98. (default: :obj:'llama2')
    dtype (torch.dtype): The dtype to use for the LLM.
            (default :obj: `torch.bloat16`)
    num_params (int): An integer representing how many params your
        huggingface transformer model has, in billions. This is used to
        automatically allocate the number of gpus needed, given the
        available GPU memory of your GPUs (default :obj:`7`)
    """
    def __init__(self, model_name: str = "llama2", dtype=torch.bfloat16,
                 num_params: int = 7):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if model_name == "llama2":
            self.printable_llm_name = "LLAMA2"
            self.huggingface_str = llama2_str_name
        elif model_name == "gemma":
            self.printable_llm_name = "GEMMA"
            self.huggingface_str = gemma_str_name
        else:
            self.printable_llm_name = model_name
            self.huggingface_str = model_name
        """
        This is a rough hueristic:
        We found that LLAMA2 (7B) + GAT hits OOM
        on a single 80GB GPU, but can fit on a single
        GPU that is slightly larger GPU.
        """
        self.mem_needed = 85 * num_params / 7
        self.llm_dtype = dtype
        print('Loading ' + str(self.printable_llm_name))
        kwargs, cpu_offload = get_llm_kwargs(self.mem_needed, self.llm_dtype)
        print("Setting up " + self.printable_llm_name + " w/ kwargs =", kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_str,
                                                       use_fast=False)
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.padding_side = padding_side
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.huggingface_str, **kwargs)

        if cpu_offload:
            self.llm_device = torch.device("cpu")
            if torch.cuda.is_available():
                import accelerate
                self.exec_device = torch.device("cuda:0")
                self.llm = accelerate.cpu_offload(
                    self.llm, execution_device=self.exec_device,
                    offload_buffers=True)
            else:
                self.exec_device = torch.device("cpu")
            from contextlib import nullcontext
            self.autocast_context = nullcontext()
        else:
            self.llm_device = self.llm.device
            self.exec_device = self.llm_device
            self.autocast_context = torch.cuda.amp.autocast(
                dtype=self.llm_dtype)
        self.word_embedding = self.llm.model.get_input_embeddings()

    def encode_inputs(self, question, additional_context=None):
        batch_size = len(question)
        questions = self.tokenizer(question, add_special_tokens=False)
        if additional_context is not None:
            additional_context = self.tokenizer(additional_context,
                                                add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False,
                           return_tensors='pt').input_ids[0].to(
                               self.llm_device))
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(
                self.llm_device)).unsqueeze(0)
        return (batch_size, questions, additional_context, eos_user_tokens,
                bos_embeds, pad_embeds)

    def forward(self, question: List[str], label: List[str],
                additional_context: Optional[List[str]] = None):
        r"""Forward pass.

        Args:
            question (List[str]): The questions/prompts.
            label (List[str]): The answers/labels.
            additional_context (List[str], optional): Additional context to
                give to the LLM, such as textified knowledge graphs.
        """
        batch_size, questions, context, eos_user_tokens, \
            bos_embeds, pad_embeds = self.encode_inputs(question, additional_context) # noqa
        # encode labels
        labels = self.tokenizer(label, add_special_tokens=False)
        # encode training specific special token
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[
                i][:max_new_tokens] + eos_tokens.input_ids
            if context is not None:
                input_ids = context.input_ids[
                    i][:max_txt_len] + questions.input_ids[
                        i] + eos_user_tokens.input_ids + label_input_ids
            else:
                input_ids = questions.input_ids[
                    i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            to_cat = [bos_embeds]
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat(to_cat, dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX
                               ] * (inputs_embeds.shape[0] -
                                    len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[
                i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.llm_device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(
            self.llm_device)

        with self.autocast_context:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    @torch.no_grad()
    def inference(self, question: List[str],
                  additional_context: Optional[List[str]] = None,
                  max_out_tokens: Optional[int] = max_new_tokens):
        r"""Inference.

        Args:
            question (List[str]): The questions/prompts.
            additional_context (List[str], optional): Additional context to
                give to the LLM, such as textified knowledge graphs.
            max_out_tokens (int, optional): How many tokens for the LLM to
                generate. (default: {32})
        """
        batch_size, questions, context, eos_user_tokens, \
            bos_embeds, pad_embeds = self.encode_inputs(question, additional_context) # noqa
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            if context is not None:
                input_ids = context.input_ids[
                    i][:max_txt_len] + questions.input_ids[
                        i] + eos_user_tokens.input_ids
            else:
                input_ids = questions.input_ids[i] + eos_user_tokens.input_ids

            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.llm_device)

        with self.autocast_context:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'question': question,
            'desc': additional_context,
        }
