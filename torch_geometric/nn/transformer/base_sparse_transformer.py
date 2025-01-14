import torch
from base_model_ViT import ViTForImageClassification_kronsp
from transformers import AutoConfig


class SequenceModel(torch.nn.Module):
    def __init__(self, model, config: AutoConfig, load_from_pretrained: bool,
                 messages="Kroneckor", self_define=False):
        """
        Base Container for (Attention-based) Sequence Models
        Args:
            model (huggingface based): the base sequence model
                (default: ViT)
            config (Optional): configuration files for the base sequence model
                (To use the default ViT Transformer, please see ~/base_model_ViT.py for reference)
            messages (Graph-based self-attention message-passing schemes): the designed functions of
                graph-based message-passing schemes for self-attention mechanisms
                (default: Kroneckor)
            self_define (Boolean): specify if overwrite this Base Class
                If self_defined is set to True, then only provide the model and config
                (if necessary), and overrite the forward function to customize your input

        The default model could be integrated with Pytorch Lightening. To use it, please
        specify your own arguments and pass ~.model to the argument of pl trainer.
        See ./example for referece
        """
        super(SequenceModel, self).__init__()
        if not self_define:
            assert model in [
                "ViT"
            ], f"Please enter available model backbones: [ViT]"
            assert messages in [
                "Kroneckor"
            ], f"Please enter available messages, examples: [Kronector]"
            if model == "ViT" and messages == "Kroneckor":
                if load_from_pretrained:
                    self._model = ViTForImageClassification_kronsp.from_pretrained(
                        'google/vit-base-patch16-224-in21k', config=config)
                else:
                    self._model = ViTForImageClassification_kronsp(config)
            else:
                raise NotImplementedError(
                    "Please enter available model backbones, examples: [ViT]")
        else:
            self._model = model

    @property
    def model(self):
        return self._model

    def forward(
        self,
        sequence_input=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        """
        Forward function of the sparse transformer, expect inputs to follow standard
        conventions as the Transformer Module.

        @Overrite this method for your own data input and customized it according to your
        own model
        """
        return self._model.forward(sequence_input, head_mask, labels,
                                   output_attentions, output_hidden_states,
                                   interpolate_pos_encoding, return_dict)
