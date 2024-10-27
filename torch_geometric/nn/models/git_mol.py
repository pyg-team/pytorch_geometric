import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2ForConditionalGeneration, Blip2Config
from transformers import AutoModel, SwinConfig, SwinModel
from transformers import Blip2Processor, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput
from torch_geometric.nn.models.git_mol_utils import GraphEncoder, GITFormer, generate_prompt, prepare_samples, \
    calculate_loss, stack_and_prepare_tensors, compute_similarity


class GitMol(torch.nn.Module):
    r"""The GitMol model from the `"GIT-Mol: A Multi-modal
    Large Language Model for Molecular Science with Graph
    , Image, and Text"
        <https://arxiv.org/pdf/2308.06911>`_ paper.
        The code for this model is heavily influenced by the paper's implementation
        <https://github.com/AI-HPC-Research-Team/GIT-Mol>
        Args:
            mode : Mode to decide Pretrain or Finetune
            graph_config : Config of the GIN architecture
            modal : Config containing input and output types to use GIT-Mol model.
            num_query_token : Number of query tokens, optional
            vision_graph_width : Width of vision and graph encoder initial layer, optional
            swin_model : Swin Transformer model path, optional,
            text_model : Text encoder model path, optional,
            t5_model : T5 model path, optional,
            blip2_processor : BLIP-2 processor, optional
            device : device, optional
        """

    def __init__(
            self,
            mode,
            graph_config,
            modal,
            num_query_token=384,
            vision_graph_width=768,
            hidden_size=256,
            swin_model='microsoft/swin-base-patch4-window7-224',
            text_model="allenai/scibert_scivocab_uncased",
            t5_model="ChemistryTreeHouse/molT5-base",
            blip2_processor="Salesforce/blip2-flan-t5-xl",
            device='cuda'
    ) -> None:
        super(GitMol, self).__init__()

        self.tokenizer = None
        self.processor = None
        self.device = device
        self.mode = mode
        self.graph_config = graph_config

        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)

        self.text_encoder = AutoModel.from_pretrained(text_model)

        self.setup_tokenizers(text_model, t5_model, blip2_processor)

        swin_config = SwinConfig.from_pretrained(swin_model)
        self.model.vision_model = SwinModel(swin_config)

        self.model.graph_encoder = GraphEncoder(self.hidden_size, graph_config)

        gitformer = GITFormer(num_query_token, vision_graph_width)

        self.model.git_former = gitformer.Qformer
        self.model.query_tokens = gitformer.query_tokens

        # Layer normalization
        self.model.ln_text = nn.LayerNorm(vision_graph_width)
        self.model.ln_vision = nn.LayerNorm(vision_graph_width)
        self.model.ln_graph = nn.LayerNorm(vision_graph_width)

        # Projection layers
        self.text_proj = nn.Linear(vision_graph_width, hidden_size)
        self.vision_proj = nn.Linear(vision_graph_width, hidden_size)
        self.graph_proj = nn.Linear(vision_graph_width, hidden_size)

        if self.mode == "pretrain":
            # Task-specific heads
            self.itm_head = nn.Linear(hidden_size, 2)
            self.gtm_head = nn.Linear(hidden_size, 2)
            self.ctm_head = nn.Linear(hidden_size, 2)

            self.temp = nn.Parameter(0.07 * torch.ones([]))

        # Freeze graph and vision layers
        self._freeze_layers()

        self.modal = modal

        self.task = []
        self.input_modal = modal['inputs_modal']
        self.output_modal = modal['outputs_modal']
        if 'isoSMILES' in self.output_modal:
            if 'image2d' in self.input_modal:
                self.task.append('itm')
                self.task.append('itc')
            if 'caption' in self.input_modal:
                self.task.append('ctm')
                self.task.append('ctc')
        if 'caption' in self.output_modal:
            if 'image2d' in self.input_modal:
                self.task.append('itm')
                self.task.append('itc')
            if 'isoSMILES' in self.input_modal:
                self.task.append('ctm')
                self.task.append('ctc')
            if 'graph2d' in self.input_modal:
                self.task.append('gtm')
                self.task.append('gtc')

    def setup_tokenizers(self, text_model, t5_model, blip2_processor):
        if self.mode == "pretrain":
            self.tokenizer = BertTokenizer.from_pretrained(text_model, truncation_side='right')
            self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        else:
            self.model.language_model = T5ForConditionalGeneration.from_pretrained(t5_model)
            self.processor = Blip2Processor.from_pretrained(blip2_processor)
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model, model_max_length=512)
            self.processor = Blip2Processor(self.processor.current_processor, self.tokenizer)

    def _freeze_layers(self):
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.graph_encoder.parameters():
            param.requires_grad = False

    def forward(self, mol):
        if self.mode == "pretrain":
            self.pretrain_forward(mol)
        else:
            self.finetune_forward(mol)

    def pretrain_forward(self, mol):
        """
        Executes the pretraining forward pass for the GitMol model. This involves processing multiple modalities
        (text, image, graph) and calculating loss for different pretraining tasks.

        Args:
            mol : A dictionary containing various modalities of molecular data.

        Returns:
            loss: Computed loss during pretraining.
        """
        loss = 0
        text_modal = self.output_modal[0]
        text = mol[text_modal]

        batch_size = text['input_ids'].size(0)

        #Process image input
        if ('image2d' in self.input_modal):
            image_embeds = self.model.ln_vision(self.model.vision_model(mol))
            image_embeds = image_embeds.float()
            image_attrs = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
            image_targets = torch.arange(batch_size).to(image_embeds.device)

        #Process graph input
        if ('graph2d' in self.input_modal):
            graph_embeds = self.model.ln_graph(self.model.graph_encoder(mol))
            graph_attrs = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(graph_embeds.device)
            graph_targets = torch.arange(batch_size).to(graph_embeds.device)

        #Process text (caption) input
        if ('caption' in self.input_modal):
            cs_text = mol['Caption']
            cs_text_embeds = self.model.git_former.bert(cs_text['input_ids'], attention_mask=cs_text['attention_mask'],
                return_dict=True,
            ).last_hidden_state
            cs_text_attrs = torch.ones(cs_text_embeds.size()[:-1], dtype=torch.long).to(cs_text_embeds.device)
            cs_text_targets = torch.arange(batch_size).to(cs_text_embeds.device)

        #Process text (SMILE) input
        if ('isoSMILES' in self.input_modal):
            cs_text = mol['isoSMILES']
            cs_text_embeds = self.model.git_former.bert(cs_text['input_ids'], attention_mask=cs_text['attention_mask'],
                return_dict=True,
            ).last_hidden_state
            cs_text_attrs = torch.ones(cs_text_embeds.size()[:-1], dtype=torch.long).to(cs_text_embeds.device)
            cs_text_targets = torch.arange(batch_size).to(cs_text_embeds.device)

        text_output = self.model.git_former.bert(text['input_ids'], attention_mask=text['attention_mask'],
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        if ("itm" in self.task):

            # Initializing lists to hold the original and negative samples
            image_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(image_embeds.shape[0]):
                prepare_samples(image_embeds_list, text_input_ids_list, text_attention_mask_list, image_embeds, text, i)

            # Stack all samples into two large tensors
            image_embeds_all, text_input_ids_all, text_attention_mask_all = stack_and_prepare_tensors(
                image_embeds_list, text_input_ids_list, text_attention_mask_list
            )

            # Create image attention masks for the concatenated tensor
            image_attrs_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )

            query_tokens_itm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_attrs_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )
            attention_mask_all = torch.cat([query_attrs_itm, text_attention_mask_all], dim=1)

            output_itm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_attrs_all,
                modal='image',
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]

            itm_logit = self.itm_head(itm_embeddings).mean(dim=1)

            loss_itm = calculate_loss(itm_logit, batch_size, itm_logit.device)
            loss = loss + loss_itm

        if ("gtm" in self.task):

            # Initializing lists to hold the original and negative samples
            graph_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(graph_embeds.shape[0]):
                prepare_samples(graph_embeds_list, text_input_ids_list, text_attention_mask_list, graph_embeds, text, i,
                                is_image=False)

            graph_embeds_all, text_input_ids_all, text_attention_mask_all = stack_and_prepare_tensors(
                graph_embeds_list, text_input_ids_list, text_attention_mask_list
            )

            # Create image attention masks for the concatenated tensor
            graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(graph_embeds_all.device)
            query_tokens_gtm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_gtm = torch.ones(query_tokens_gtm.size()[:-1], dtype=torch.long).to(graph_embeds_all.device)
            attention_mask_all = torch.cat([query_atts_gtm, text_attention_mask_all], dim=1)

            output_gtm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_gtm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                modal='graph',
                return_dict=True,
            )

            gtm_embeddings = output_gtm.last_hidden_state[:, : query_tokens_gtm.size(1), :]
            gtm_logit = self.gtm_head(gtm_embeddings).mean(dim=1)

            loss_gtm = calculate_loss(gtm_logit, batch_size, gtm_logit.device)
            loss = loss + loss_gtm

        if ("ctm" in self.task):
            # Initializing lists to hold the original and negative samples
            cs_text_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(cs_text_embeds.shape[0]):
                prepare_samples(cs_text_embeds_list, text_input_ids_list, text_attention_mask_list, cs_text_embeds,
                                text, i)

            # Stack all samples into two large tensors
            cs_text_embeds_all, text_input_ids_all, text_attention_mask_all = stack_and_prepare_tensors(
                cs_text_embeds_list, text_input_ids_list, text_attention_mask_list
            )

            # Create image attention masks for the concatenated tensor
            cs_text_attrs_all = torch.ones(cs_text_embeds_all.size()[:-1], dtype=torch.long).to(cs_text_embeds_all.device)
            query_tokens_ctm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_attrs_ctm = torch.ones(query_tokens_ctm.size()[:-1], dtype=torch.long).to(cs_text_embeds_all.device)
            attention_mask_all = torch.cat([query_attrs_ctm, text_attention_mask_all], dim=1)

            output_ctm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_ctm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=cs_text_embeds_all,
                encoder_attention_mask=cs_text_attrs_all,
                modal='cs_text',
                return_dict=True,
            )

            ctm_embeddings = output_ctm.last_hidden_state[:, : query_tokens_ctm.size(1), :]
            ctm_logit = self.ctm_head(ctm_embeddings).mean(dim=1)

            loss_ctm = calculate_loss(ctm_logit, batch_size, ctm_logit.device)
            loss = loss + loss_ctm

        if ("itc" in self.task):
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attrs,
                modal='image',
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            sim_i2t, sim_t2i = compute_similarity(image_feats, text_feat, self.temp)

            loss_itc = (F.cross_entropy(sim_i2t, image_targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2i, image_targets, label_smoothing=0.1)) / 2
            loss = loss + loss_itc

        if ("gtc" in self.task):
            query_tokens = self.model.query_tokens.expand(graph_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_attrs,
                modal='graph',
                return_dict=True,
            )

            graph_feats = F.normalize(
                self.graph_proj(query_output.last_hidden_state), dim=-1
            )

            sim_g2t, sim_t2g = compute_similarity(graph_feats, text_feat, self.temp)

            loss_gtc = (F.cross_entropy(sim_g2t, graph_targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2g, graph_targets, label_smoothing=0.1)) / 2
            loss = loss + loss_gtc

        if ("ctc" in self.task):
            query_tokens = self.model.query_tokens.expand(cs_text_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cs_text_embeds,
                encoder_attention_mask=cs_text_attrs,
                modal='cs_text',
                return_dict=True,
            )

            cs_text_feats = F.normalize(
                self.cs_text_proj(query_output.last_hidden_state), dim=-1
            )

            sim_c2t, sim_t2c = compute_similarity(cs_text_feats, text_feat, self.temp)

            loss_ctc = (F.cross_entropy(sim_c2t, cs_text_targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2c, cs_text_targets, label_smoothing=0.1)) / 2
            loss = loss + loss_ctc

        loss = loss / len(self.task)
        return loss

    def finetune_forward(self, mol):
        """
        Executes the finetuning forward pass for the GitMol model, processing different tasks and calculating loss.

        Args:
            mol : A dictionary containing various modalities of molecular data.

        Returns:
            loss: The computed loss during finetuning.
        """
        loss_list = []
        for task in self.task_list:
            inputs_modal = task['inputs_modal']
            outputs_modal = task['outputs_modal']
            qformer_outputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)
            if 'SMILES' in outputs_modal or 'caption' in outputs_modal or 'isoSMILES' in outputs_modal:
                loss_text = self.loss_text(mol, qformer_outputs, outputs_modal)
                loss_list.append(loss_text)

        loss_tensor = torch.stack(loss_list)
        loss = torch.mean(loss_tensor)
        return loss

    def get_git_former_outputs(self, mol, inputs_modal, outputs_modal):
        """
            Generates GitFormer outputs by processing input modalities and computing embeddings.

            Args:
                mol : Molecular data containing various input modalities.
                inputs_modal : List of input modalities.
                outputs_modal : List of output modalities.

            Returns:
                qformer_outputs, mol: A tuple containing the generated embeddings and updated molecular data.
            """
        input_tensors = []
        batch_size = len(mol['smiles'])

        prompt = generate_prompt(inputs_modal, outputs_modal)
        prompt = self.processor(text=prompt, return_tensors="pt")
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask']
        mol['input_ids'] = input_ids.repeat(batch_size, 1).to(self.device)
        mol['attention_mask'] = attention_mask.repeat(batch_size, 1).to(self.device)
        if 'image2d' in inputs_modal:
            language_model_inputs_image = self.get_image_git_former_features(mol)
            input_tensors.append(language_model_inputs_image)

        if 'SMILES' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)

        if 'isoSMILES' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)

        if 'caption' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)

        if 'graph2d' in inputs_modal:
            language_model_inputs_graph = self.get_graph_git_former_features(mol)
            input_tensors.append(language_model_inputs_graph)

        qformer_outputs = torch.stack(input_tensors).mean(dim=0)
        return qformer_outputs, mol

    def get_graph_git_former_features(self, mol):
        graph_embeds = self.model.ln_graph(self.model.graph_encoder(mol))
        graph_attrs = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )
        query_tokens = self.model.query_tokens.expand(graph_embeds.shape[0], -1, -1)

        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_attrs,
            modal='graph',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_graph = query_output
        return language_model_inputs_graph

    def get_text_git_former_features(self, mol, inputs_modal):
        if 'isoSMILES' in inputs_modal:
            text = mol['isoSMILES']
        if 'SMILES' in inputs_modal:
            text = mol['SMILES']
        if 'caption' in inputs_modal:
            text = mol['Caption']
        text_embeds = self.model.ln_text(self.model.git_former.bert(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            return_dict=True,
        ).last_hidden_state)
        text_attention_mask = torch.ones(text_embeds.size()[:-1], dtype=torch.long, device=text_embeds.device)
        query_tokens = self.model.query_tokens.expand(text_embeds.shape[0], -1, -1)

        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_attention_mask,
            modal='cs_text',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_text = query_output
        return language_model_inputs_text

    def get_image_git_former_features(self, mol):
        image_embeds = self.model.ln_vision(self.model.vision_model(mol))

        image_embeds = image_embeds.float()
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        image_embeds = self.model.vision_model(mol)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            modal='image',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_image = query_output

        return language_model_inputs_image

    def loss_text(self, mol, qformer_outputs, outputs_modal):

        language_model_inputs = qformer_outputs

        if ('SMILES' in outputs_modal):
            labels = mol['smiles_labels']
        elif ('isoSMILES' in outputs_modal):
            labels = mol['isosmiles_labels']
        elif ('caption' in outputs_modal):
            labels = mol['caption_labels']

        # Use decoder of MolT5
        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        outputs = self.model.language_model(
            encoder_outputs=h,
            labels=labels
        )

        loss_text = outputs['loss']
        return loss_text

    @torch.no_grad()
    def generate_text(self, mol, inputs_modal, outputs_modal):
        """
            Generates text from the input molecular data using the GitMol model. This method can handle
             multiple input modalities and produces text outputs based on the provided configuration.

            Args:
                mol : The input molecular data, including various modalities (e.g., image, text, graph).
                inputs_modal : List of input modalities (e.g., 'SMILES', 'image2d').
                outputs_modal : List of output modalities specifying the desired text outputs.

            Returns:
                generated_texts: A list of generated texts corresponding to the input data.
            """
        generated_ids = self.generate_language(mol, inputs_modal, outputs_modal)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def generate_language(self, mol, inputs_modal, outputs_modal):
        """
            Generates text outputs by processing the input data using the language generation
             capabilities of the GitMol model.

            Args:
                mol : The input molecular data, including encoded embeddings and other features.
                inputs_modal (list): List of input modalities (e.g., 'SMILES', 'image2d').
                outputs_modal (list): List of output modalities specifying the desired text outputs.

            Returns:
                torch.Tensor: A tensor containing the generated token IDs.
            """
        language_model_inputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        input_ids = mol['input_ids']
        attention_mask = mol['attention_mask']

        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        outputs = self.model.language_model.generate(
            encoder_outputs=h,
            num_beams=5,
            max_length=512
        )
        return outputs