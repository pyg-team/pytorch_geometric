# flake8: noqa
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, LayerNorm, Linear, ReLU, Sequential

from torch_geometric.nn import GINEConv
from torch_geometric.nn.attention.gitformer import BertConfig, BertLMHeadModel
from torch_geometric.nn.cv import SwinTransformer
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.utils import to_dense_batch


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.gnns.append(
                GINEConv(
                    nn=Sequential(
                        Linear(in_channels, in_channels * 2),
                        ReLU(),
                        Linear(in_channels * 2, in_channels),
                    ),
                    train_eps=True,
                    edge_dim=in_channels,
                ))
            self.batch_norms.append(BatchNorm1d(in_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            x = gnn(x, edge_index, edge_attr)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x, mask = to_dense_batch(x, batch)
        return x, mask


class GITFormer(torch.nn.Module):
    def __init__(self, num_query_token, vision_graph_width,
                 cross_attention_freq=2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained(
            "allenai/scibert_scivocab_uncased")
        encoder_config.encoder_width = vision_graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=encoder_config)
        self.query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0,
                                       std=encoder_config.initializer_range)


class GITMol(torch.nn.Module):
    r"""Assume pretrain task = image + graph + smiles --> caption."""
    def __init__(self, ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(num_layers=2, in_channels=16)
        self.graph_proj = Linear(16, 768)
        self.ln_graph = LayerNorm(768)
        self.text_encoder = SentenceTransformer(
            model_name='allenai/scibert_scivocab_uncased',
            pooling_strategy='last_hidden_state',
        )
        self.ln_text = LayerNorm(768)
        self.vision_encoder = SwinTransformer()
        self.vision_proj = Linear(1536, 768)
        self.ln_vision = LayerNorm(768)

        self.gitformer = GITFormer(384, 768)

        self.xtm_head = {
            'image': Linear(self.gitformer.Qformer.config.hidden_size, 2),
            'graph': Linear(self.gitformer.Qformer.config.hidden_size, 2),
            'cs_text': Linear(self.gitformer.Qformer.config.hidden_size, 2),
        }

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
        captions: List[str],
        images: Tensor,
    ) -> Tensor:
        batch_size = len(smiles)

        x_vision = self.vision_encoder(images)
        x_vision = self.vision_proj(x_vision)
        x_vision = self.ln_vision(x_vision)  # [bs, patch_len, d]
        # vision_atts = torch.ones(x_vision.size()[:-1],
        #                          dtype=torch.long).to(x_vision.device)
        torch.arange(batch_size).to(x_vision.device)

        # TODO: add atom and bond embedding
        x_graph, graph_atts = self.graph_encoder(x, edge_index, batch,
                                                 edge_attr)
        x_graph = self.graph_proj(x_graph)
        x_graph = self.ln_graph(x_graph)  # [bs, node_len, d]
        torch.arange(batch_size).to(x_graph.device)

        x_smiles = self.text_encoder.encode(smiles)  # [bs, seq_len, d]
        # smiles_atts = torch.ones(x_smiles.size()[:-1],
        #                          dtype=torch.long).to(x_smiles.device)
        torch.arange(batch_size).to(x_smiles.device)

        x_captions = self.text_encoder.encode(captions)  # [bs, seq_len, d]
        caption_input_ids, caption_attention_masks = self.text_encoder.get_input_ids(
            captions)
        torch.arange(batch_size).to(x_captions.device)

        print(x_graph.size(), x_smiles.size(), x_captions.size(),
              x_vision.size())
        loss = 0
        for x_embed, modal in zip([x_graph, x_smiles, x_vision],
                                  ['graph', 'cs_text', 'image']):
            loss += self._calc_xtm_loss(x_embed, caption_input_ids,
                                        caption_attention_masks, modal)

        return loss

    def _calc_xtm_loss(
        self,
        x_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        modal: str,
    ) -> Tensor:
        # Initializing lists to hold the original and negative samples
        x_embeds_list = []
        text_input_ids_list = []
        text_attention_mask_list = []

        batch_size = x_embeds.size(0)
        for i in range(batch_size):
            # Original samples
            x_embeds_list.append(x_embeds[i])
            text_input_ids_list.append(input_ids[i, :])
            text_attention_mask_list.append(attention_mask[i, :])

            if batch_size > 1:
                # Negative samples (neg_text_input_ids corresponds to x_embeds)
                neg_text_input_ids = input_ids[i - 1 if i == batch_size -
                                               1 else i + 1, :]
                neg_text_attention_mask = attention_mask[i -
                                                         1 if i == batch_size -
                                                         1 else i + 1, :]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                x_embeds_list.append(x_embeds[i, :])

                # Negative samples (text_input_ids corresponds to neg_x_embeds)
                neg_x_embeds = x_embeds[i - 1 if i == batch_size - 1 else i +
                                        1, :]
                x_embeds_list.append(neg_x_embeds)
                text_input_ids_list.append(input_ids[i, :])
                text_attention_mask_list.append(attention_mask[i, :])

        # Stack all samples into two large tensors
        x_embeds_all = torch.stack(x_embeds_list, dim=1) \
            .reshape(-1, x_embeds.size(1), x_embeds.size(2))
        text_input_ids_all = torch.stack(text_input_ids_list, dim=1) \
            .reshape(-1, input_ids.size(1))
        text_attention_mask_all = torch.stack(text_attention_mask_list, dim=1) \
            .reshape(-1, attention_mask.size(1))
        # Create image attention masks for the concatenated tensor
        image_atts_all = torch.ones(x_embeds_all.size()[:-1], dtype=torch.long) \
            .to(x_embeds_all.device)
        query_tokens_xtm = self.gitformer.query_tokens.expand(
            text_input_ids_all.shape[0], -1, -1)
        query_atts_xtm = torch.ones(query_tokens_xtm.size()[:-1], dtype=torch.long) \
            .to(x_embeds_all.device)
        attention_mask_all = torch.cat(
            [query_atts_xtm, text_attention_mask_all], dim=1)

        output_xtm = self.gitformer.Qformer.bert(
            text_input_ids_all,
            query_embeds=query_tokens_xtm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=x_embeds_all,
            encoder_attention_mask=image_atts_all,
            modal=modal,
            return_dict=True,
        ).last_hidden_state
        xtm_embeddings = output_xtm[:, :query_tokens_xtm.size(1), :]

        xtm_logit = self.xtm_head[modal](xtm_embeddings).mean(dim=1)
        # Create labels: 1 for the original samples, 0 for the negative samples
        if batch_size > 1:
            labels = torch.cat(
                [torch.ones(batch_size),
                 torch.zeros(batch_size * 2)], dim=0)
        else:
            labels = torch.ones(batch_size)
        labels = labels.long().to(xtm_logit.device)

        # Calculate cross entropy loss
        return F.cross_entropy(xtm_logit, labels)

    def _calc_itc_loss(self, ) -> Tensor:
        pass

    def _calc_gtc_loss(self, ) -> Tensor:
        pass

    def _calc_ctc_loss(self, ) -> Tensor:
        pass

    def pretrain(
        self,
        task: str,
    ) -> None:
        pass

    def finetune(
        self,
        task: str,
    ) -> None:
        pass

    def inference(self, ) -> Tensor:
        pass
