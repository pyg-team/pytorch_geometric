import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertLMHeadModel

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import add_self_loops


class GITFormer(nn.Module):
    """A class representing a modified version of the BERT architecture that includes cross-attention layers.

    Args:
        num_query_token (int): Number of query tokens.
        vision_graph_width (int): Width of the vision and graph models.
        cross_attention_freq (int, optional): Frequency of cross-attention layers. Default is 2.

    Attributes:
        Qformer (nn.Module): The BERT-based model with added cross-attention layers.
        query_tokens (torch.nn.Parameter): Learnable query tokens for the model.
    """
    def __init__(self, num_query_token, vision_graph_width,
                 cross_attention_freq=2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained(
            "../../ckpts/text_ckpts/scibert_scivocab_uncased")
        encoder_config.encoder_width = vision_graph_width

        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        self.Qformer = BertLMHeadModel.from_pretrained(
            "../../ckpts/text_ckpts/scibert_scivocab_uncased",
            config=encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0,
                                       std=encoder_config.initializer_range)


class GraphEncoder(nn.Module):
    """A Graph Neural Network (GNN) encoder using GINConv layers for graph-based feature extraction.

    Args:
        hidden_size (int): Size of the hidden layer.
        graph_config (dict, optional): Configuration for the graph network. Defaults is None.

    Attributes:
        conv1 (GINConv): First GIN convolution layer.
        conv2 (GINConv): Second GIN convolution layer.
        fc_hidden (nn.Linear): Fully connected layer to produce the hidden representation.
    """
    def __init__(self, hidden_size, graph_config=None):
        super().__init__()

        self.num_features = graph_config.get("graph_num_features", 300)
        self.hidden_size = hidden_size

        nn = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 2 * self.num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.num_features, self.hidden_size))

        self.conv1 = GINConv(nn)
        self.conv2 = GINConv(nn)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, mol):
        x, edge_index, edge_attr, batch = (mol["graph2d"].x,
                                           mol["graph2d"].edge_index,
                                           mol["graph2d"].edge_attr,
                                           mol["graph2d"].batch)

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr,
                                               fill_value=0,
                                               num_nodes=x.size(0))
        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.conv1(x, edge_embeddings)
        x = F.relu(x)
        x = self.conv2(x, edge_embeddings)
        x = F.relu(x)

        graph_feats = global_add_pool(x, batch)

        node_feats = self.fc_hidden(graph_feats)

        return node_feats


def generate_prompt(inputs, outputs):
    """Generates a descriptive prompt string based on input and output modalities.

    Args:
        inputs (list): List of input modality names.
        outputs (list): List of output modality names.

    Returns:
        prompt: Generated descriptive prompt.
    """
    modalities = {
        'image2d': 'image',
        'SMILES': 'SMILES representation',
        'isoSMILES': 'isoSMILES representation',
        'caption': 'textual description',
        'graph2d': 'structure graph'
    }

    inputs_desc = [modalities[i] for i in inputs]
    outputs_desc = [modalities[o] for o in outputs]

    inputs_str = ' and '.join(inputs_desc)
    outputs_str = ', '.join(outputs_desc)

    prompt = f"Given the provided {inputs_str}, generate the corresponding {outputs_str}."
    return prompt


def prepare_samples(embed_list, text_input_list, text_attn_list, embeds, text,
                    index, is_image=True):
    """Prepare original and negative samples for image/graph-text matching tasks.

    Args:
        embed_list : List to store embeddings.
        text_input_list : List to store text input IDs.
        text_attn_list : List to store text attention masks.
        embeds : Embeddings to process (image/graph embeddings).
        text : Dictionary containing text input details.
        index : Index of the current sample.
        is_image : True if processing image, False if processing graph.

    """
    # Original samples
    embed_list.append(embeds[index])
    text_input_list.append(text['input_ids'][index])
    text_attn_list.append(text['attention_mask'][index])

    # Negative samples (neg_text_input_ids corresponds to embed)
    neg_text_input_ids = text['input_ids'][
        index -
        1] if index == embeds.shape[0] - 1 else text['input_ids'][index + 1]
    neg_text_attention_mask = text['attention_mask'][index - 1] if index == embeds.shape[0] - 1 else \
    text['attention_mask'][index + 1]

    text_input_list.append(neg_text_input_ids)
    text_attn_list.append(neg_text_attention_mask)
    embed_list.append(embeds[index])

    # Negative samples (text_input_ids corresponds to neg_embed)
    neg_embeds = embeds[index -
                        1] if index == embeds.shape[0] - 1 else embeds[index +
                                                                       1]
    embed_list.append(neg_embeds)
    text_input_list.append(text['input_ids'][index])
    text_attn_list.append(text['attention_mask'][index])


def stack_and_prepare_tensors(embeds_list, text_input_list, text_attn_list):
    """Stack samples into larger tensors for batch processing.

    Args:
        embeds_list: List of embeddings.
        text_input_list : List of text input IDs.
        text_attn_list: List of text attention masks.

    Returns:
        embeds_all, text_input_all, text_attn_all: Stacked embeddings, input IDs, and attention masks.
    """
    embeds_all = torch.stack(embeds_list,
                             dim=1).reshape(-1, embeds_list[0].size(0),
                                            embeds_list[0].size(1))
    text_input_all = torch.stack(text_input_list,
                                 dim=1).reshape(-1, text_input_list[0].size(0))
    text_attn_all = torch.stack(text_attn_list,
                                dim=1).reshape(-1, text_attn_list[0].size(0))

    return embeds_all, text_input_all, text_attn_all


def calculate_loss(logits, batch_size, device):
    labels = torch.cat([torch.ones(batch_size),
                        torch.zeros(batch_size * 2)], dim=0).long().to(device)
    return F.cross_entropy(logits, labels)


def compute_similarity(encoder_feats, text_feats, temp):
    """Computes the similarity scores between encoder features and text features.

    Args:
        encoder_feats: Encoded features from the model (image, graph, or text).
        text_feats: Text features.
        temp: Temperature parameter for scaling.

    Returns:
        sim_encoder_to_text, sim_text_to_encoder
    """
    # Encoder-to-text similarity
    sim_q2t = torch.matmul(encoder_feats.unsqueeze(1),
                           text_feats.unsqueeze(-1)).squeeze()
    sim_e2t, _ = sim_q2t.max(-1)
    sim_e2t = sim_e2t / temp

    # Text-to-encoder similarity
    sim_t2q = torch.matmul(
        text_feats.unsqueeze(1).unsqueeze(1),
        encoder_feats.permute(0, 2, 1)).squeeze()
    sim_t2e, _ = sim_t2q.max(-1)
    sim_t2e = sim_t2e / temp

    return sim_e2t, sim_t2e
