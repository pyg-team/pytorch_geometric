import torch
import torch.nn as nn
from torch.autograd import Variable

from torch_geometric.nn import GlobalAttention
from torch_geometric.utils.reason import Fusion, QueryReform, TypeLayer


class ReaRev(torch.nn.Module):
    """Recurrent Attention and Reasoning model (ReaRev) for iterative reasoning
    over knowledge graphs using graph neural networks (GNNs) and
    query-driven instructional guidance.

    Args:
        args (dict): Configuration dictionary containing hyperparameters.
        num_entity (int): Total number of entities in the knowledge graph.
        num_relation (int): Total number of relations in the knowledge graph.
        num_word (int): Vocabulary size or maximum word index for queries.
    """
    def __init__(self, args, num_entity, num_relation, num_word):
        super().__init__()
        self.norm_rel = args['norm_rel']
        self._init_model_params(args)
        self._define_layers(args)
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3 * self.entity_dim, self.entity_dim)
        self.fusion = Fusion(self.entity_dim)

        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))

    def _init_model_params(self, args):
        """Initialize model hyperparameters."""
        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

    def _define_layers(self, args):
        """Define primary layers and encoders."""
        self.layers(args)

    def layers(self, args):
        """Define layers for embedding transformations and attention encoders."""
        self.word_dim
        self.kg_dim
        entity_dim = self.entity_dim

        self.linear_dropout = args['linear_dropout']

        self.entity_linear = nn.Linear(in_features=self.ent_dim,
                                       out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim,
                                         out_features=entity_dim)

        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(
                in_features=entity_dim,
                out_features=entity_dim,
                linear_drop=self.linear_drop,
                device=self.device,
                norm_rel=self.norm_rel,
            )

        self.self_att_r = GlobalAttention(nn.Linear(self.entity_dim, 1))

        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def private_module_def(self, args, num_entity, num_relation):
        """Defines private modules: LMs, GNN layers, etc.
        Initializes modules needed by the reasoning pipeline.
        """
        # Define ReasonGNNLayer if not already imported
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation,
                                        self.entity_dim, self.alg)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        """Computes the initial entity embeddings.
        """
        if self.encode_type:
            local_entity_emb = self.type_layer(
                local_entity=local_entity,
                edge_list=kb_adj_mat,
                rel_features=rel_features,
            )
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)
        return local_entity_emb

    def get_rel_feature(self):
        """Gets relation features from embeddings or encoded texts.

        Returns:
            tuple: rel_features, rel_features_inv.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(
                self.rel_features_inv)

            rel_features = self.self_att_r(rel_features)
            rel_features_inv = self.self_att_r(rel_features_inv)

        return rel_features, rel_features_inv

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input,
                    query_entities):
        """Initializes reasoning procedure by preparing instructions, entity
        embeddings, and GNN layers.
        """
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat,
                                                  rel_features)
        self.init_entity_emb = self.local_entity_emb

        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist

        self.reasoning.init_reason(
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            local_entity_emb=self.local_entity_emb,
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities,
        )

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        """Computes the label loss based on current predictions vs. teacher distribution.
        """
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist,
                                reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, training=False):
        """Standard forward pass: perform reasoning steps and produce predictions and loss.
        """
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, _, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).long().to(self.device)
        query_entities = torch.from_numpy(query_entities).float().to(
            self.device)
        answer_dist = torch.from_numpy(answer_dist).float().to(self.device)
        seed_dist = torch.from_numpy(seed_dist).float().to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input = torch.from_numpy(query_text).long().to(self.device)

        self.init_reason(
            curr_dist=current_dist,
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            q_input=q_input,
            query_entities=query_entities,
        )

        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, _ = self.instruction.get_instruction(
                self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        self.dist_history.append(self.curr_dist)

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(
                    self.curr_dist, relation_ins, step=j)
            self.dist_history.append(self.curr_dist)

            qs = []
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(
                    self.instruction.instructions[j].squeeze(1),
                    global_rep,
                    query_entities,
                    local_entity,
                )
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)

        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist,
                                    teacher_dist=answer_dist,
                                    label_valid=case_valid)

        pred = torch.max(pred_dist, dim=1)[1]
        tp_list = None
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]

        return loss, pred, pred_dist, tp_list
