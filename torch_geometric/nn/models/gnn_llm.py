import torch
import torch.nn as nn

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    WITH_PEFT = True
except ImportError as e:  # noqa
    WITH_PEFT = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    WITH_TRANSFORMERS = True
except ImportError as e:  # noqa
    WITH_TRANSFORMERS = False

from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT
from torch_geometric.utils import scatter

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


def get_llm_kwargs(mem_needed):
    assert torch.cuda.is_available(), "GPU needed to run LLMs efficiently!"
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

    for i in range(gpus_2_use_4_llm):
        max_mem_dict[i] = str(avail_mem_dict[i]) + "GiB"
    kwargs["max_memory"] = max_mem_dict
    kwargs["device_map"] = "auto"
    return kwargs


class LLM(nn.Module):
    def __init__(self, llm_name: str = "llama2", llm_dtype=torch.bfloat16,
                 num_params: int = 7):
        super().__init__()
        if llm_name == "llama2":
            self.printable_llm_name = "LLAMA2"
            self.huggingface_str = llama2_str_name
        elif llm_name == "gemma":
            self.printable_llm_name = "GEMMA"
            self.huggingface_str = gemma_str_name
        else:
            self.printable_llm_name = llm_name
            self.huggingface_str = llm_name
        self.mem_needed = 75 * num_params / 7
        self.llm_dtype = llm_dtype
        print('Loading ' + str(self.printable_llm_name))
        kwargs = get_llm_kwargs(self.mem_needed)
        print("Setting up " + self.printable_llm_name + " w/ kwargs =", kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_str,
                                                       use_fast=False)
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.padding_side = padding_side
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.huggingface_str, torch_dtype=self.llm_dtype,
            low_cpu_mem_usage=True, **kwargs)
        self.llm_device = self.llm.device
        self.word_embedding = self.llm.model.get_input_embeddings()

    def encode_inputs(self, samples: Batch):
        batch_size = len(samples['question'])
        questions = self.tokenizer(samples["question"],
                                   add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"],
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
        return (batch_size, questions, descriptions, eos_user_tokens,
                bos_embeds, pad_embeds)

    def inference(self, samples: Batch):
        # this function is for comparing a pretrained LLM to a trained GNN_LLM
        batch_size, questions, descriptions, eos_user_tokens, \
            bos_embeds, pad_embeds = self.encode_inputs(samples)
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids
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

        with torch.cuda.amp.autocast(dtype=self.llm_dtype):
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }


class GNN_LLM(nn.Module):
    r"""This GNN+LLM implementation is based on G-retriever.
    Original Paper: <https://arxiv.org/abs/2402.07630>`_.
    See `examples/llm_plus_gnn/g_retriever.py` for example usage.

    Args:
        llm_to_use (str): A string representing the huggingface model you
            want to use. This module has been tested for 'llama2' and 'gemma'.
            Other huggingface transformer models should work if you pass the
            correct name, see huggingface.co for details. If any issues occur
            please file an issue on
            https://github.com/pyg-team/pytorch_geometric
            and assign to puririshi98. (default: :obj:'llama2')
        llm_use_lora (bool): use LORA from peft for training the LLM. see
            https://huggingface.co/docs/peft/en/index for details.
            llm_dtype (torch.dtype): The lower precision dtype to use for the
            LLM. (default :obj: `torch.bloat16`)
        num_llm_params (int): An integer representing how many params your
            huggingface transformer model has, in billions. This is used to
            automatically allocate the number of gpus needed, given the
            available GPU memory of your GPUs (default :obj:`7`)
        gnn_to_use (BasicGNN): Please pass a valid model that extends
            torch_geometric.nn.models.basic_gnn.BasicGNN. (default: :obj:`GAT`)
        gnn_in_channels (int): (default: 1024)
        gnn_hidden_channels (int): (default: 1024)
        gnn_out_channels (int): (default: 1024)
        num_gnn_layers (int): (default: 4)
        num_gnn_heads (int): Number of heads to use for BasicGNNs with the
        `heads` kwarg. (default: 4)
        mlp_hidden_dim (int): (default: 2048)
        mlp_out_dim (int): (default: 4096)
    """
    def __init__(
        self,
        llm_to_use='llama2',
        llm_use_lora: bool = True,
        llm_dtype=torch.bfloat16,
        num_llm_params: int = 7,
        gnn_to_use=GAT,
        gnn_in_channels: int = 1024,
        gnn_hidden_channels: int = 1024,
        gnn_out_channels: int = 1024,
        num_gnn_layers: int = 4,
        num_gnn_heads: int = 4,
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 4096,
    ):
        super().__init__()
        if not WITH_TRANSFORMERS:
            raise ImportError(
                "To use GNN_LLM, please `pip install transformers`.")
        if 'llama' in llm_to_use.lower():
            self.llm_to_use = LLM('llama2', llm_dtype)
        elif 'gemma' in llm_to_use.lower():
            self.llm_to_use = LLM('gemma', llm_dtype)
        else:
            self.llm_to_use = LLM(llm_to_use, llm_dtype)
        self.llm = self.llm_to_use.llm
        self.llm_dtype = llm_dtype
        if llm_use_lora:
            if not WITH_PEFT:
                raise ImportError("To use LORA, please `pip install peft`.")
            print("Training our LLM with LORA!")
            self.llm = prepare_model_for_kbit_training(self.llm)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, config)
        self.llm_device = self.llm_to_use.llm_device
        self.tokenizer = self.llm_to_use.tokenizer
        print('Finished loading LLAMA!')

        self.graph_encoder = gnn_to_use(
            in_channels=gnn_in_channels,
            out_channels=gnn_out_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=num_gnn_layers,
            heads=num_gnn_heads,
        ).to(self.llm_device)
        # For the MLP Projection
        mlp_hidden_dim = gnn_out_channels
        self.projector = nn.Sequential(
            nn.Linear(gnn_out_channels, mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim),
        ).to(self.llm_device)

        self.word_embedding = self.llm_to_use.word_embedding

    def encode_graphs(self, samples: Batch):
        x = samples.x.to(self.llm_device)
        edge_index = samples.edge_index.long().to(self.llm_device)
        edge_attr = samples.edge_attr.to(self.llm_device)
        n_embeds = self.graph_encoder(x, edge_index.long(), edge_attr)
        batch = samples.batch.to(self.llm_device)
        # mean pooling
        g_embeds = scatter(n_embeds, batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples: Batch):
        batch_size, questions, descriptions, eos_user_tokens, \
            bos_embeds, pad_embeds = self.llm_to_use.encode_inputs(samples)
        # encode labels
        labels = self.tokenizer(samples.label, add_special_tokens=False)
        # encode training specific special token
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        num_nodes_per_graph = samples.ptr[1:] - samples.ptr[:-1]
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[
                i][:max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            to_cat = [bos_embeds]
            if num_nodes_per_graph[i] != 0:
                to_cat.append(graph_embeds[i].unsqueeze(0))
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

        with torch.cuda.amp.autocast(dtype=self.llm_dtype):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    def inference(self, samples: Batch):
        batch_size, questions, descriptions, eos_user_tokens, \
            bos_embeds, pad_embeds = self.llm_to_use.encode_inputs(samples)
        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[
                i][:max_txt_len] + questions.input_ids[
                    i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            inputs_embeds = torch.cat(
                [bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds],
                dim=0)
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

        with torch.cuda.amp.autocast(dtype=self.llm_dtype):
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
