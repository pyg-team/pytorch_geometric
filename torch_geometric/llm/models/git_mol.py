from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, LayerNorm, Linear, ReLU, Sequential

from torch_geometric.nn import GINEConv
from torch_geometric.nn.nlp import SentenceTransformer, VisionTransformer
from torch_geometric.utils import add_self_loops, to_dense_batch


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        dropout: float = 0.,
        num_atom_type: int = 120,
        num_chirality_tag: int = 3,
        num_bond_type: int = 6,
        num_bond_direction: int = 3,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.x_embed1 = torch.nn.Embedding(num_atom_type, in_channels)
        self.x_embed2 = torch.nn.Embedding(num_chirality_tag, in_channels)
        self.edge_embed1 = torch.nn.Embedding(num_bond_type, in_channels)
        self.edge_embed2 = torch.nn.Embedding(num_bond_direction, in_channels)

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
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embed1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embed2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embed1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embed2.weight.data)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        x = self.x_embed1(x[:, 0].long()) + self.x_embed2(x[:, 1].long())
        edge_index, edge_attr = add_self_loops(
            edge_index,
            edge_attr,
            fill_value=0,
            num_nodes=x.size(0),
        )
        edge_attr = self.edge_embed1(edge_attr[:, 0]) + self.edge_embed2(
            edge_attr[:, 1])
        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            x = gnn(x, edge_index, edge_attr)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x, mask = to_dense_batch(x, batch)
        return x, mask


class GITFormer(torch.nn.Module):
    def __init__(
        self,
        num_query_token: int,
        vision_graph_width: int,
        cross_attention_freq: int = 2,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
        config.encoder_width = vision_graph_width
        # insert cross-attention layer every other block
        config.add_cross_attention = True
        config.is_decoder = True
        config.cross_attention_freq = cross_attention_freq
        config.query_length = num_query_token
        self.Qformer = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=config)
        self.query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)


class GITMol(torch.nn.Module):
    r"""The GITMol model from the `"GIT-Mol: A Multi-modal Large Language
    Model for Molecular Science with Graph, Image, and Text"
    <https://arxiv.org/pdf/2308.06911>`_ paper.

    .. note::
        For an example of using :class:`GITMol`, see
        `examples/llm/git_mol.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/git_mol.py>`_.
    """
    def __init__(self) -> None:
        super().__init__()
        # graph
        self.graph_encoder = GraphEncoder(num_layers=2, in_channels=16)
        self.graph_proj = Linear(16, 768)
        self.ln_graph = LayerNorm(768)
        # text
        self.text_encoder = SentenceTransformer(
            model_name='allenai/scibert_scivocab_uncased',
            pooling_strategy='last_hidden_state',
        )
        self.text_proj = Linear(768, 768)
        self.ln_text = LayerNorm(768)
        # vision
        self.vision_encoder = VisionTransformer(
            model_name='microsoft/swin-base-patch4-window7-224', )
        self.vision_proj = Linear(1024, 768)
        self.ln_vision = LayerNorm(768)
        # cross-attention
        self.gitformer = GITFormer(384, 768)

        self.xtm_head = torch.nn.ModuleDict({
            'image':
            Linear(self.gitformer.Qformer.config.hidden_size, 2),
            'graph':
            Linear(self.gitformer.Qformer.config.hidden_size, 2),
            'cs_text':
            Linear(self.gitformer.Qformer.config.hidden_size, 2),
        })

        self.xtc_proj = torch.nn.ModuleDict({
            'image':
            Linear(self.gitformer.Qformer.config.hidden_size, 768),
            'graph':
            Linear(self.gitformer.Qformer.config.hidden_size, 768),
            'cs_text':
            Linear(self.gitformer.Qformer.config.hidden_size, 768),
        })
        self.temp = torch.nn.Parameter(0.07 * torch.ones([]))
        self.model_freeze()

    def model_freeze(self) -> None:
        for param in self.graph_encoder.parameters():
            param.requires_grad = False

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
        images: Tensor,
        captions: List[str],
    ) -> Tensor:
        batch_size = len(smiles)

        x_vision = self.vision_encoder(images)
        x_vision = self.vision_proj(x_vision)
        x_vision = self.ln_vision(x_vision)  # [bs, patch_len, d]
        vision_atts = torch.ones(x_vision.size()[:-1],
                                 dtype=torch.long).to(x_vision.device)
        vision_targets = torch.arange(batch_size).to(x_vision.device)

        x_graph, graph_atts = self.graph_encoder(x, edge_index, batch,
                                                 edge_attr)
        x_graph = self.graph_proj(x_graph)
        x_graph = self.ln_graph(x_graph)  # [bs, node_len, d]
        graph_targets = torch.arange(batch_size).to(x_graph.device)

        x_smiles = self.text_encoder.encode(smiles)  # [bs, seq_len, d]
        smiles_atts = torch.ones(x_smiles.size()[:-1],
                                 dtype=torch.long).to(x_smiles.device)
        smiles_targets = torch.arange(batch_size).to(x_smiles.device)

        caption_input_ids, caption_attention_masks = self.text_encoder.get_input_ids(  # noqa: E501
            captions)

        text_output = self.gitformer.Qformer(
            caption_input_ids,
            attention_mask=caption_attention_masks,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        loss = 0
        for x_embed, x_atts, x_targets, modal in zip(
            [x_graph, x_smiles, x_vision],
            [graph_atts, smiles_atts, vision_atts],
            [graph_targets, smiles_targets, vision_targets],
            ['graph', 'cs_text', 'image'],
        ):
            loss += self._calc_xtc_loss(x_embed, x_atts, x_targets, text_feat,
                                        modal)
            loss += self._calc_xtm_loss(x_embed, caption_input_ids,
                                        caption_attention_masks, modal)

        return loss / 6

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
        # Create image attention masks for the concatenated tensor
        image_attns_all = torch.ones(x_embeds_all.size()[:-1],
                                     dtype=torch.long).to(x_embeds_all.device)
        query_tokens_xtm = self.gitformer.query_tokens.expand(
            text_input_ids_all.shape[0], -1, -1)
        query_attns_xtm = torch.ones(query_tokens_xtm.size()[:-1],
                                     dtype=torch.long).to(x_embeds_all.device)

        output_xtm = self.gitformer.Qformer(
            inputs_embeds=query_tokens_xtm,
            attention_mask=query_attns_xtm,
            encoder_hidden_states=x_embeds_all,
            encoder_attention_mask=image_attns_all,
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

    def _calc_xtc_loss(
        self,
        x_embeds: Tensor,
        x_atts: Tensor,
        x_targets: Tensor,
        text_feat: Tensor,
        modal: str,
    ) -> Tensor:
        query_tokens = self.gitformer.query_tokens.expand(
            x_embeds.shape[0], -1, -1)

        query_output = self.gitformer.Qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=x_embeds,
            encoder_attention_mask=x_atts,
            return_dict=True,
        ).last_hidden_state

        x_feats = F.normalize(self.xtc_proj[modal](query_output), dim=-1)

        sim_q2t = torch.matmul(
            x_feats.unsqueeze(1),
            text_feat.unsqueeze(-1),
        ).squeeze(-1)

        # modal-text similarity: aggregate across all query tokens
        sim_x2t, _ = sim_q2t.max(-1)
        sim_x2t = sim_x2t / self.temp

        # text-query similarity
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1),
            x_feats.permute(0, 2, 1),
        ).squeeze(-2)

        # text-modal similarity: aggregate across all query tokens
        sim_t2x, _ = sim_t2q.max(-1)
        sim_t2x = sim_t2x / self.temp

        loss_itc = (
            F.cross_entropy(sim_x2t, x_targets, label_smoothing=0.1) +
            F.cross_entropy(sim_t2x, x_targets, label_smoothing=0.1)) / 2

        return loss_itc
