import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTModel,
    ViTSelfAttention,
)

from sparse_transformer.sparse_messages import *


def kron(a, b, axis):
    axis.sort()
    asize = a.shape
    bsize = b.shape

    assert len(asize) == len(bsize)
    for i in range(len(asize)):
        assert (asize[i] == bsize[i]) or (i in axis) or (asize[i]
                                                         == 1) or (bsize[i]
                                                                   == 1)

    cnt = 0
    for i in axis:
        a = a.unsqueeze(i + cnt + 1)
        b = b.unsqueeze(i + cnt)
        cnt += 1

    result = a * b
    res_size = torch.tensor(asize)
    for i in axis:
        res_size[i] = asize[i] * bsize[i]
    result = result.reshape(torch.Size(res_size))
    return result


class ViTForImageClassification_kronsp(ViTForImageClassification):
    def __init__(self, config):
        super(ViTForImageClassification_kronsp, self).__init__(config)
        self.vit = ViTModel_kronsp(config)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViTModel_kronsp(ViTModel):
    def __init__(self, config):
        super(ViTModel_kronsp, self).__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.attention = ViTSelfAttention_kronsp(config)


class ViTSelfAttention_kronsp(ViTSelfAttention):
    def __init__(self, config):
        super(ViTSelfAttention_kronsp, self).__init__(config)
        self.Message = Kronecker(self.dropout)
        self.config = config
        self.cond = config.cond
        self.order = config.order
        self.block_list = config.block_list
        assert len(self.block_list) == self.order
        self.num_attention_heads = config.num_attention_heads

        num_patches = (self.config.image_size // self.config.patch_size)**2
        self.num_tokens = num_patches + 1
        self.nz_eles = int(self.num_tokens * self.num_tokens *
                           (1 - config.sparsity * 0.01))

        assert np.prod(self.block_list) == self.num_tokens - 1
        if self.cond:
            self.proj = nn.Linear(config.hidden_size, config.hidden_size,
                                  bias=config.qkv_bias)
        else:
            self.param_masks = nn.ParameterList([
                nn.Parameter(
                    torch.randn(config.num_attention_heads, block_size,
                                block_size), requires_grad=True)
                for block_size in self.block_list
            ])

        self.num_nzeles_list = [
            int(
                np.ceil(block**2 *
                        ((1 - config.sparsity * 0.01)**(1 / self.order))))
            for block in self.block_list
        ]

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        B, N, D = hidden_states.shape
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(
            self.query(hidden_states)) / math.sqrt(
                self.attention_head_size)  #B,H,N,Dh

        if self.config.sparsity > 0:
            for order, param_mask in enumerate(self.param_masks):
                logits = param_mask
                res = torch.sigmoid(logits)
                split_num, _ = torch.kthvalue(
                    -1 * res.reshape(res.shape[0], -1),
                    self.num_nzeles_list[order] + 1, dim=1, keepdim=False)
                mask_layer = ((res >= (-split_num).unsqueeze(1).unsqueeze(1)
                               ).type_as(res) - res).detach() + res
                if order > 0:
                    mask = kron(mask, mask_layer, axis=[1, 2])
                else:
                    mask = mask_layer
            mask = F.pad(mask, (1, 0, 1, 0), "constant", 1)
            self.mask_np = mask.detach().cpu().numpy()
            mask = mask.unsqueeze(0)
            mask = mask.transpose(2, 3)
            mask = mask[0]
            mask = mask.to(torch.bool)

        query_layer = query_layer.float()
        key_layer = key_layer.float()
        value_layer = value_layer.float()

        context_layer, att = self.Message(key=key_layer, query=query_layer,
                                          value=value_layer, mask=mask,
                                          B=B)  #B,H,N,D

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )  #B,N,D
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   att) if output_attentions else (context_layer, )
        return outputs
