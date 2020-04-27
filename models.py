"""
    A file which contains all relevant models for this work.
    Most of these have been taken, reappropriated from [PyKeen](https://github.com/SmartDataAnalytics/PyKEEN/)
"""

# Torch imports
import torch
from torch import nn
import torch.autograd
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import numpy as np
from typing import List, Optional, Dict, Tuple

# Local imports
from utils_gcn import get_param, scatter_add, MessagePassing, ccorr, rotate
from utils import *
from utils_mytorch import compute_mask
from gnn_encoder import CompQGCNEncoder


class BaseModule(nn.Module):
    """A base class for all of the models."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.margin_ranking_loss_size_average: bool = None
        self.entity_embedding_max_norm: Optional[int] = None
        self.entity_embedding_norm_type: int = 2
        self.hyper_params = [config['EMBEDDING_DIM'],
                             config['MARGIN_LOSS'],
                             config['LEARNING_RATE']]

        # Device selection
        self.device: torch.device = config['DEVICE']

        # Loss
        self.margin_loss = config['MARGIN_LOSS']
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            reduction='mean' if self.margin_ranking_loss_size_average else 'sum'
        )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = config['NUM_ENTITIES']
        #: The number of unique relation types in the knowledge graph
        self.num_relations = config['NUM_RELATIONS']
        #: The dimension of the embeddings to generate
        self.embedding_dim = config['EMBEDDING_DIM']

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
            padding_idx=0
        )

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _compute_loss(self, positive_scores: torch.Tensor,
                      negative_scores: torch.Tensor) -> torch.Tensor:
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss


def slice_triples(triples: torch.Tensor, slices: int):
    """ Slice in 3 or 5 as needed """
    if slices == 5:
        return triples[:, 0], triples[:, 1], triples[:, 2], triples[:, 3], triples[:, 4]
    elif slices == 3:
        return triples[:, 0], triples[:, 1], triples[:, 2]
    else:
        return triples[:, 0], triples[:, 2::2], triples[:,
                                                1::2]  # subject, all other entities, all relations


class TransE(BaseModule):
    """
    An implementation of TransE [borders2013]_.
     This model considers a relation as a translation from the head to the tail entity.
    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.
    .. seealso::
       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py


        Modifications to use this to work with quints can be turned on/off with a flag.
    """

    model_name = 'TransE MM'

    def __init__(self, config) -> None:

        self.margin_ranking_loss_size_average: bool = True
        self.entity_embedding_max_norm: Optional[int] = None
        self.entity_embedding_norm_type: int = 2
        self.model_name = 'TransE MM'
        super().__init__(config)
        self.statement_len = config['STATEMENT_LEN']

        # Embeddings
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.relation_embeddings = nn.Embedding(config['NUM_RELATIONS'], config['EMBEDDING_DIM'],
                                                padding_idx=0)

        self.config = config

        if self.config['PROJECT_QUALIFIERS']:
            self.proj_mat = nn.Linear(2 * self.embedding_dim, self.embedding_dim, bias=False)

        self._initialize()

        # Make pad index zero. # TODO: Should pad index be configurable? Probably not, right? Cool? Cool.
        # self.entity_embeddings.weight.data[0] = torch.zeros_like(self.entity_embeddings.weight[0], requires_grad=True)
        # self.relation_embeddings.weight.data[0] = torch.zeros_like(self.relation_embeddings.weight[0], requires_grad=True)

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.config['EMBEDDING_DIM'])
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight,
                           p=self.config['NORM_FOR_NORMALIZATION_OF_RELATIONS'], dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

        self.relation_embeddings.weight.data[0] = torch.zeros(1, self.embedding_dim)
        self.entity_embeddings.weight.data[0] = torch.zeros(1,
                                                            self.embedding_dim)  # zeroing the padding index

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores

    def forward(self, batch_positives, batch_negatives) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))
        self.entity_embeddings.weight.data[0] = torch.zeros(1,
                                                            self.embedding_dim)  # zeroing the padding index

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return (positive_scores, negative_scores), loss

    def _score_triples(self, triples) -> torch.Tensor:
        """ Get triple/quint embeddings, and compute scores """
        scores = self._compute_scores(*self._get_triple_embeddings(triples))
        return scores

    def _self_attention_2d(self, head_embeddings, relation_embeddings, tail_embeddings,
                           scale=False):
        """ Simple self attention """
        # @TODO: Add scaling factor

        triple_vector = head_embeddings + relation_embeddings[:, 0, :] - tail_embeddings[:, 0, :]
        qualifier_vectors = relation_embeddings[:, 1:, :] - tail_embeddings[:, 1:, :]
        ct = torch.cat((triple_vector.unsqueeze(1), qualifier_vectors), dim=1)
        score = torch.bmm(ct, ct.transpose(1, 2))
        mask = compute_mask(score, padding_idx=0)
        score = masked_softmax(score, mask)
        return torch.sum(torch.bmm(score, ct), dim=1)

    def _self_attention_1d(self, head_embedding, relation_embedding, tail_embedding, scale=False):
        """
            Self attention but 1D instead of n x n .
            So, weighing all qualifiers wrt the triple scores
        """

        triple_vector = head_embedding + relation_embedding[:, 0, :] - tail_embedding[:, 0, :]
        qualifier_vectors = relation_embedding[:, 1:, :] - tail_embedding[:, 1:, :]
        # ct = torch.cat((triple_vector.unsqueeze(1), qualifier_vectors), dim=1)
        score = torch.bmm(triple_vector.unsqueeze(1), qualifier_vectors.transpose(1, 2)).squeeze(1)
        mask = compute_mask(score, padding_idx=0)
        score = masked_softmax(score, mask)
        weighted_qualifier_vectors = score.unsqueeze(-1) * qualifier_vectors

        return torch.sum(
            torch.cat((triple_vector.unsqueeze(1), weighted_qualifier_vectors), dim=1), dim=1)

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings,
                        qual_relation_embeddings=None, qual_entity_embeddings=None):
        """
            Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: embeddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param qual_entity_embeddings: embeddings of qualifier relations of dimensinos batchsize x embeddig_dim
        :param qual_relation_embeddings: embeddings of qualifier entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        if self.statement_len != -1:
            sum_res = head_embeddings + relation_embeddings - tail_embeddings
            if qual_relation_embeddings is not None and qual_entity_embeddings is not None:
                sum_res = sum_res + qual_relation_embeddings - qual_entity_embeddings
            # Add the vector element wise
        else:
            # use formula head + sum(relations - tails)

            # Project or not
            if self.config['PROJECT_QUALIFIERS']:
                # relation_embeddings[:,1:,:] = self.proj_mat(relation_embeddings[:,1:,:])
                # tail_embeddings[:,1:,:] = self.proj_mat(tail_embeddings[:,1:,:])

                quals = torch.sum(relation_embeddings[:, 1:, :] - tail_embeddings[:, 1:, :], dim=1)
                new_rel = torch.cat((relation_embeddings[:, 0, :], quals), dim=1)
                p_proj = self.proj_mat(new_rel)
                sum_res = head_embeddings + p_proj - tail_embeddings[:, 0, :]
            elif self.config['SELF_ATTENTION'] == 1:
                sum_res = self._self_attention_1d(head_embeddings, relation_embeddings,
                                                  tail_embeddings)
            elif self.config['SELF_ATTENTION'] == 2:
                sum_res = self._self_attention_2d(head_embeddings, relation_embeddings,
                                                  tail_embeddings)
            else:
                sum_res = head_embeddings + torch.sum(relation_embeddings - tail_embeddings, dim=1)
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        if self.statement_len == 5:
            heads, relations, tails, qual_relations, qual_entities = slice_triples(triples, 5)
            return (
                self._get_entity_embeddings(heads),
                self._get_relation_embeddings(relations),
                self._get_entity_embeddings(tails),
                self._get_relation_embeddings(qual_relations),
                self._get_entity_embeddings(qual_entities)
            )

        elif self.statement_len == 3:
            heads, relations, tails = slice_triples(triples, 3)
            return (
                self._get_entity_embeddings(heads),
                self._get_relation_embeddings(relations),
                self._get_entity_embeddings(tails)
            )
        else:
            head, statement_entities, statement_relations = slice_triples(triples, -1)
            return (
                self._get_entity_embeddings(head),
                self.relation_embeddings(statement_relations),
                self.entity_embeddings(statement_entities)
            )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)


class ConvKB(BaseModule):
    """
    An implementation of ConvKB.

    A Novel Embedding Model for Knowledge Base CompletionBased on Convolutional Neural Network.
    """

    model_name = 'ConvKB'

    def __init__(self, config) -> None:
        self.margin_ranking_loss_size_average: bool = True
        self.entity_embedding_max_norm: Optional[int] = None
        self.entity_embedding_norm_type: int = 2
        self.model_name = 'ConvKB'
        super().__init__(config)
        self.statement_len = config['STATEMENT_LEN']

        # Embeddings
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.relation_embeddings = nn.Embedding(config['NUM_RELATIONS'], config['EMBEDDING_DIM'],
                                                padding_idx=0)

        self.config = config

        self.criterion = nn.SoftMarginLoss(
            reduction='sum'
        )

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=config['NUM_FILTER'],
                              kernel_size=(config['MAX_QPAIRS'], 1),
                              bias=True)

        self.fc = nn.Linear(config['NUM_FILTER'] * self.embedding_dim, 1, bias=False)

        self._initialize()

        # Make pad index zero. # TODO: Should pad index be configurable? Probably not, right? Cool? Cool.
        # self.entity_embeddings.weight.data[0] = torch.zeros_like(self.entity_embeddings.weight[0], requires_grad=True)
        # self.relation_embeddings.weight.data[0] = torch.zeros_like(self.relation_embeddings.weight[0], requires_grad=True)

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.config['EMBEDDING_DIM'])
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight,
                           p=self.config['NORM_FOR_NORMALIZATION_OF_RELATIONS'], dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

        self.relation_embeddings.weight.data[0] = torch.zeros(1, self.embedding_dim)
        self.entity_embeddings.weight.data[0] = torch.zeros(1,
                                                            self.embedding_dim)  # zeroing the padding index

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores

    def _compute_loss(self, positive_scores: torch.Tensor,
                      negative_scores: torch.Tensor) -> torch.Tensor:
        # Let n items in pos score.
        y = np.repeat([-1], repeats=positive_scores.shape[0])  # n item here (all -1)
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        pos_loss = self.criterion(positive_scores, -1 * y)
        neg_loss = self.criterion(negative_scores, y)
        return pos_loss + neg_loss

    def forward(self, batch_positives, batch_negatives) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data

        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        self.entity_embeddings.weight.data[0] = torch.zeros(1,
                                                            self.embedding_dim)  # zeroing the padding index

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return (positive_scores, negative_scores), loss

    def _score_triples(self, triples) -> torch.Tensor:
        """ Get triple/quint embeddings, and compute scores """
        scores = self._compute_scores(*self._get_triple_embeddings(triples))
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings,
                        qual_relation_embeddings=None, qual_entity_embeddings=None):
        """
            Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: embeddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param qual_entity_embeddings: embeddings of qualifier relations of dimensinos batchsize x embeddig_dim
        :param qual_relation_embeddings: embeddings of qualifier entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        statement_emb = torch.zeros(head_embeddings.shape[0],
                                    relation_embeddings.shape[1] * 2 + 1,
                                    head_embeddings.shape[1],
                                    device=self.config['DEVICE'],
                                    dtype=head_embeddings.dtype)  # 1 for head embedding

        # Assignment
        statement_emb[:, 0] = head_embeddings
        statement_emb[:, 1::2] = relation_embeddings
        statement_emb[:, 2::2] = tail_embeddings

        # Convolutional operation
        statement_emb = F.relu(self.conv(statement_emb.unsqueeze(1))).squeeze(
            -1)  # bs*number_of_filter*emb_dim
        statement_emb = statement_emb.view(statement_emb.shape[0], -1)
        score = self.fc(statement_emb)

        return score.squeeze()

    def _get_triple_embeddings(self, triples):
        head, statement_entities, statement_relations = slice_triples(triples, -1)
        return (
            self._get_entity_embeddings(head),
            self.relation_embeddings(statement_relations),
            self.entity_embeddings(statement_entities)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)


class DenseClf(nn.Module):

    def __init__(self, inputdim, hiddendim, outputdim):
        """
            This class has a two layer dense network of changable dims.
            Intended use case is that of
                - *bidir dense*:
                    give it [v_q, v_p] and it gives a score.
                    in this case, have outputdim as 1
                - * bidir dense dot*
                    give it v_q and it gives a condensed vector
                    in this case, have any outputdim, preferably outputdim < inputdim
        :param inputdim: int: #neurons
        :param hiddendim: int: #neurons
        :param outputdim: int: #neurons
        """

        super(DenseClf, self).__init__()

        self.inputdim = int(inputdim)
        self.hiddendim = int(hiddendim)
        self.outputdim = int(outputdim)
        self.hidden = nn.Linear(self.inputdim, self.outputdim)
        # self.output = nn.Linear(self.hiddendim, self.outputdim)

    def forward(self, x):
        """
        :param x: bs*sl
        :return:
        """
        _x = F.sigmoid(self.hidden(x))

        # if self.outputdim == 1:
        #     return F.sigmoid(self.output(_x))
        #
        # else:
        #     return F.sigmoid(self.output(_x))

        return _x

    def evaluate(self, y_pred, y_true):
        """
        :param x: bs*nc
        :param y: bs*nc (nc is number of classes)
        :return: accuracy (torch tensor)
        """
        y_pred, y_true = torch.argmax(y_pred), torch.argmax(y_true)
        final_score = torch.mean((y_pred == y_true).to(torch.float))
        return final_score


class GraphAttentionLayerMultiHead(nn.Module):

    def __init__(self, config: dict, residual_dim: int = 0, final_layer: bool = False):

        super().__init__()

        # Parse params
        ent_emb_dim, rel_emb_dim = config['EMBEDDING_DIM'], config['EMBEDDING_DIM']
        out_features = config['KBGATARGS']['OUT']
        num_head = config['KBGATARGS']['HEAD']
        alpha_leaky = config['KBGATARGS']['ALPHA']

        self.w1 = nn.Linear(2 * ent_emb_dim + rel_emb_dim + residual_dim, out_features)
        self.w2 = nn.Linear(out_features, num_head)
        self.relu = nn.LeakyReLU(alpha_leaky)

        self.final = final_layer

        # Why copy un-necessary stuff
        self.heads = num_head

        # Not initializing here. Should be called by main module

    def initialize(self):
        nn.init.xavier_normal_(self.w1.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.w2.weight.data, gain=1.414)

    def forward(self, data: torch.Tensor, mask: torch.Tensor = None):
        """
            data: size (batchsize, num_neighbors, 2*ent_emb+rel_emb) or (bs, n, emb)
            mask: size (batchsize, num_neighbors)

            PS: num_neighbors is padded either with max neighbors or with a limit
        """
        # data: bs, n, emb
        bs, _, _ = data.shape

        c = self.w1(data)  # c: bs, n, out_features
        b = self.relu(self.w2(c)).squeeze()  # b: bs, n, num_heads
        m = mask.unsqueeze(-1).repeat(1, 1, self.heads)  # m: bs, n, num_heads
        alphas = masked_softmax(b, m, dim=1)  # α: bs, n, num_heads

        # BMM simultaneously weighs the triples and sums across neighbors
        h = torch.bmm(c.transpose(1, 2), alphas)  # h: bs, out_features, num_heads

        if self.final:
            h = torch.mean(h, dim=-1)  # h: bs, out_features
        else:
            # Their implementation uses ELU instead of RELU :/
            h = F.elu(h).view(bs, -1)  # h: bs, out_features*num_heads

        return h


class KBGat(BaseModule):
    """
        Untested.
        And Depreciated
    """
    model_name = 'KBGAT'

    def __init__(self, config: dict, pretrained_embeddings=None) -> None:

        self.margin_ranking_loss_size_average: bool = True
        self.entity_embedding_max_norm: Optional[int] = None
        self.entity_embedding_norm_type: int = 2
        self.model_name = 'KBGAT'
        super().__init__(config)
        self.statement_len = config['STATEMENT_LEN']

        # Embeddings
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.relation_embeddings = nn.Embedding(config['NUM_RELATIONS'], config['EMBEDDING_DIM'],
                                                padding_idx=0)

        self.config = config

        if self.config['PROJECT_QUALIFIERS']:
            self.proj_mat = nn.Linear(2 * self.embedding_dim, self.embedding_dim, bias=False)

        self.gat1 = GraphAttentionLayerMultiHead(self.config, final_layer=False)
        self.gat2 = GraphAttentionLayerMultiHead(self.config,
                                                 residual_dim=self.config['EMBEDDING_DIM'],
                                                 final_layer=True)

        # Note: not initing them
        self.wr = nn.Linear(config['EMBEDDING_DIM'], config['KBGATARGS']['OUT'])
        self.we = nn.Linear(config['EMBEDDING_DIM'], config['KBGATARGS']['OUT'])

        # Put in weights
        self._initialize(pretrained_embeddings)

    def _initialize(self, pretrained_embeddings):
        if pretrained_embeddings is None:
            embeddings_init_bound = 6 / np.sqrt(self.config['EMBEDDING_DIM'])
            nn.init.uniform_(
                self.entity_embeddings.weight.data,
                a=-embeddings_init_bound,
                b=+embeddings_init_bound,
            )
            nn.init.uniform_(
                self.relation_embeddings.weight.data,
                a=-embeddings_init_bound,
                b=+embeddings_init_bound,
            )

            norms = torch.norm(self.relation_embeddings.weight,
                               p=self.config['NORM_FOR_NORMALIZATION_OF_RELATIONS'], dim=1).data
            self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
                norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

            self.relation_embeddings.weight.data[0] = torch.zeros(1, self.embedding_dim)
            self.entity_embeddings.weight.data[0] = torch.zeros(1,
                                                                self.embedding_dim)  # zeroing the padding index

        else:
            raise NotImplementedError("Haven't wired in the mechanism to load weights yet fam")

        # Also init the GATs with bacteria and tapeworms
        self.gat1.initialize(), self.gat2.initialize()

    def predict(self, triples_hops) -> torch.Tensor:
        scores = self._score_triples_(triples_hops)
        return scores

    def normalize(self) -> None:

        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data

        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        # zeroing the padding index
        self.entity_embeddings.weight.data[0] = torch.zeros(1, self.embedding_dim)

    def forward(self, pos: List, neg: List) -> (tuple, torch.Tensor):
        """
            triples of size: (bs, 3)    (s and r1 and  o)
               hop1 of size: (bs, n, 2) (s and r1)
               hop2 of size: (bs, n, 3) (s and r1 and r2)
            (here n -> num_neighbors)
            (here hop2 has for bc it is <s r1 r2 o> )
            (pos has pos_triples, pos_hop1, pos_hop2. neg has same.)
        """
        pos_triples, pos_hop1, pos_hop2 = pos
        neg_triples, neg_hop1, neg_hop2 = neg

        self.normalize()

        positive_scores = self._score_triples_(pos_triples, pos_hop1, pos_hop2)
        negative_scores = self._score_triples_(neg_triples, neg_hop1, neg_hop2)

        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return (positive_scores, negative_scores), loss

    def _score_triples_(self,
                        triples: torch.Tensor,
                        hop1: torch.Tensor,
                        hop2: torch.Tensor) -> torch.Tensor:
        """
            triples of size: (bs, 3)
            hop1 of size: (bs, n, 2) (s, p) (o is same as that of triples)
            hop2 of size: (bs, n, 3) (s, p1, p2) (o is same as that of triples)
            1. Embed all things so triples (bs, 3, emb), hop1 (bs, n, 3, emb), hop2 (bs, n, 4, emb)
            2. Concat hop1, hop2 to be (bs, n, 3*emb) and (bs, n, 4*emb) each
            3. Pass the baton to some other function.
        """
        s, p, o, h1_s, h1_p, h2_s, h2_p1, h2_p2 = self._embed_(triples, hop1, hop2)

        hf = self._score_o_(s, p, o, h1_s, h1_p, h2_s, h2_p1, h2_p2)
        sum_res = s + p - hf
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _score_o_(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor,
                  h1_s: torch.Tensor, h1_p: torch.Tensor,
                  h2_s: torch.Tensor, h2_p1: torch.Tensor, h2_p2: torch.Tensor) -> torch.Tensor:
        """
            Expected embedded tensors: following

            s: (bs, emb)
            p: (bs, emb)
            o: (bs, emb)
            h1_s: (bs, n, emb)
            h1_p: (bs, n, emb)
            h2_s: (bs, n, emb)
            h2_p1: (bs, n, emb)
            h2_p2: (bs, n, emb)

            Next:
              -> compute mask1, cat o to it, and push to gat 1
              -> compute mask2, cat gat1op to it, and push to gat 2
              -> do all the residual connections
              -> return final score
        """

        # Compute Masks
        mask1 = compute_mask(h1_s)[:, :, 0]  # m1   : (bs, n)
        mask2 = compute_mask(h2_s)[:, :, 0]  # m2   : (bs, n)

        # Cat `o` in in h1
        h1_o = o.repeat(1, h1_s.shape[1], 1)  # h1_o : (bs, n, emb)
        h1_o = h1_o * mask1.unsqueeze(-1)  # h1_o : (bs, n, emb)
        h1 = torch.cat((h1_s, h1_p, h1_o), dim=-1)  # h1   : (bs, n, 3*emb)

        # Pass to first graph attn layer
        gat1_op = self.gat1(h1, mask1)  # op   : (bs, num_head*out_dim)
        self.normalize()

        # Do the G` = G*W thing here
        gat1_p = self.wr(p.squeeze(1))  # rels : (bs, emb')
        gat1_op_concat = torch.cat((gat1_op, gat1_p), dim=-1)  # op   : (bs, emb'+num_head*out_dim)

        # Average h2_p1, h2_p2
        h2_p = (h2_p1 + h2_p2) / 2.0  # h2_p : (bs, n, emb)

        # Treat this as the new "o", and throw in h2 data as well.
        h2_o = gat1_op_concat.unsqueeze(1).repeat(1, h2_s.shape[1],
                                                  1)  # h2_o : (bs, n, num_head*out_dim + emb')
        h2_o = h2_o * mask2.unsqueeze(-1)  # h2_o : (bs, n, num_head*out_dim + emb')
        h2 = torch.cat((h2_s, h2_p, h2_o),
                       dim=-1)  # h2   : (bs, n, 2*emb + num_head*out_dim + emb')

        # Pass to second graph attn layer
        hf = self.gat2(h2, mask2)  # hf   : (bs, out_dim)
        self.normalize()

        # Eq. 12 (H'' = W^EH^T + H^F)
        # @TODO: Should we add or concat?
        hf = hf + self.we(o.squeeze(1))  # hf   : (bs, out_dim)
        # hf = torch.cat((hf, self.we(o.squeeze(1))), dim=-1)           # hf   : (bs, out_dim*2)

        return hf

    def _embed_(self, tr, h1, h2):
        """ The obj is to pass things through entity and rel matrices as needed """
        # Triple
        s, p, o = slice_triples(tr, 3)  # *    : (bs, 1)

        s = self.entity_embeddings(s).unsqueeze(1)
        p = self.relation_embeddings(p).unsqueeze(1)
        o = self.entity_embeddings(o).unsqueeze(1)  # o    : (bs, 1, emb)

        # Hop1
        h1_s, h1_p = h1[:, :, 0], h1[:, :, 1]  # h1_* : (bs, n)

        h1_s = self.entity_embeddings(h1_s)  # h1_s : (bs, n, emb)
        h1_p = self.relation_embeddings(h1_p)  # h1_p : (bs, n, emb)

        # Hop2
        h2_s, h2_p1, h2_p2 = h2[:, :, 0], h2[:, :, 1], h2[:, :, 2]  # h2_* : (bs, n)

        h2_s = self.entity_embeddings(h2_s)  # h2_s : (bs, n, emb)
        h2_p1 = self.relation_embeddings(h2_p1)  # h2_p1: (bs, n, emb)
        h2_p2 = self.relation_embeddings(h2_p2)  # h2_p2: (bs, n, emb)

        return s, p, o, h1_s, h1_p, h2_s, h2_p1, h2_p2

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)




class CompGCNTransEStatements(CompQGCNEncoder):
    """ Merging from class TransE code """

    model_name = 'CompGCN TransE Statements MM'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, pretrained=None):

        # @TODO: What to do of pretrained embeddings
        if pretrained:
            raise NotImplementedError

        super(self.__class__, self).__init__(kg_graph_repr, config)
        self.drop = torch.nn.Dropout(self.hid_drop)
        self.gamma = config['COMPGCNARGS']['GAMMA']

    def forward(self, sub, rel, quals):
        """ quals is (bs, 2*maxQpairs) """
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.drop, self.drop, quals, True)

        obj_emb = sub_emb + rel_emb + torch.sum(qual_rel_emb - qual_obj_emb, dim=1)

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score

class CompGCNTransE(CompQGCNEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, pretrained=None):

        # @TODO: What to do of pretrained embeddings
        if pretrained:
            raise NotImplementedError

        super(self.__class__, self).__init__(kg_graph_repr, config)
        self.drop = torch.nn.Dropout(self.hid_drop)
        self.gamma = config['COMPGCNARGS']['GAMMA']

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)

        obj_emb = sub_emb + rel_emb

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class CompGCNDistMultStatement(CompQGCNEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):

        super().__init__(kg_graph_repr, config)
        self.drop = torch.nn.Dropout(self.hid_drop)

    def forward(self, sub, rel, quals):

        """
            quals is (bs, 2*maxQpairs)
        """
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.drop, self.drop, quals, True)

        obj_emb = sub_emb * rel_emb * torch.prod(qual_rel_emb * qual_obj_emb, 1)

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCNDistMult(CompQGCNEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):

        super().__init__(kg_graph_repr, config)
        self.drop = torch.nn.Dropout(self.hid_drop)
        # self.bias = config['COMPGCNARGS']['BIAS']

        # TODO: Will raise error since bias is not defined.
        # raise AssertionError("self.bias is not defined.")

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCNConvE(CompQGCNEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']
        self.k_w = config['COMPGCNARGS']['K_W']
        self.k_h = config['COMPGCNARGS']['K_H']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(self.kernel_sz, self.kernel_sz), stride=1,
                                       padding=0, bias=config['COMPGCNARGS']['BIAS'])

        assert self.emb_dim == self.k_w * self.k_h, "Incorrect combination of conv params and emb dim " \
                                                    " ConvE decoder will not work properly, " \
                                                    " should be emb_dim == k_w * k_h"

        flat_sz_h = int(2 * self.k_w) - self.kernel_sz + 1
        flat_sz_w = self.k_h - self.kernel_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.n_filters
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        # TODO check that reshaping against the batch size - apparently emb_dim  = k_w * k_h
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop,
                                                      self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCNConvEStatement(CompQGCNEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']
        self.k_w = config['COMPGCNARGS']['K_W']
        self.k_h = config['COMPGCNARGS']['K_H']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(self.kernel_sz, self.kernel_sz), stride=1,
                                       padding=0, bias=config['COMPGCNARGS']['BIAS'])
        assert 2 * self.k_w > self.kernel_sz and self.k_h > self.kernel_sz, "kernel size is incorrect"
        assert self.emb_dim * (config['MAX_QPAIRS'] - 1) == 2 * self.k_w * self.k_h, "Incorrect combination of conv params and emb dim " \
                                                    " ConvE decoder will not work properly, " \
                                                    " should be emb_dim * (pairs - 1) == 2* k_w * k_h"

        flat_sz_h = int(2 * self.k_w) - self.kernel_sz + 1
        flat_sz_w = self.k_h - self.kernel_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.n_filters
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1)  # [bs, 2 + num_qual_pairs, emb_dim]
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score




class CompGCN_ConvKB(CompQGCNEncoder):

    model_name = 'CompGCN_ConvKB'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_ConvKB'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']

        self.entity_embedding_norm_type: int = 2
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(2, self.kernel_sz), stride=1,
                                       padding=0, bias=config['COMPGCNARGS']['BIAS'])
        self.flat_sz = self.n_filters * (self.emb_dim - self.kernel_sz + 1)
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        #self._initialize()


    # def _initialize(self):
    #     embeddings_init_bound = 6 / np.sqrt(self.config['EMBEDDING_DIM'])
    #     nn.init.uniform_(
    #         self.entity_embeddings.weight.data,
    #         a=-embeddings_init_bound,
    #         b=+embeddings_init_bound,
    #     )
    #     nn.init.uniform_(
    #         self.relation_embeddings.weight.data,
    #         a=-embeddings_init_bound,
    #         b=+embeddings_init_bound,
    #     )
    #
    #     norms = torch.norm(self.relation_embeddings.weight,
    #                        p=self.config['NORM_FOR_NORMALIZATION_OF_RELATIONS'], dim=1).data
    #     self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
    #         norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))
    #
    #     self.relation_embeddings.weight.data[0] = torch.zeros(1, self.embedding_dim)
    #     self.entity_embeddings.weight.data[0] = torch.zeros(1,
    #                                                         self.embedding_dim)  # zeroing the padding index


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1).unsqueeze(1)
        return stack_inp


    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop,
                                                      self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        #  x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ConvKB_Statement(CompQGCNEncoder):
    model_name = 'CompGCN_ConvKB_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_ConvKB_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']

        self.entity_embedding_norm_type: int = 2
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.pooling = config['COMPGCNARGS']['POOLING']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(config['MAX_QPAIRS']-1, self.kernel_sz), stride=1,
                                       padding=0, bias=config['COMPGCNARGS']['BIAS'])
        self.flat_sz = self.n_filters * (self.emb_dim - self.kernel_sz + 1)
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        # self._initialize()

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).unsqueeze(1)  # [bs, 1, 2 + num_qual_pairs, emb_dim]
        #stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        #  x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ConvKB_Hinge_Statement(CompQGCNEncoder):
    model_name = 'CompGCN_ConvKB_Hinge_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_ConvKB_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']

        self.entity_embedding_norm_type: int = 2
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.pooling = config['COMPGCNARGS']['POOLING']
        self.multi_convs = config['COMPGCNARGS']['MULTI_CONVS']
        self.pooling = config['COMPGCNARGS']['POOLING']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)


        if self.multi_convs:
            self.m_convs = nn.ModuleList()
            for i in range((config['MAX_QPAIRS']-3)//2):
                self.m_convs.append(torch.nn.Conv2d(1, out_channels=self.n_filters,
                            kernel_size=(4, self.kernel_sz), stride=1,
                            padding=0, bias=config['COMPGCNARGS']['BIAS']))
        else:
            self.m_convs = torch.nn.Conv2d(1, out_channels=self.n_filters,
                            kernel_size=(4, self.kernel_sz), stride=1,
                            padding=0, bias=config['COMPGCNARGS']['BIAS'])


        self.flat_sz = self.n_filters * (self.emb_dim - self.kernel_sz + 1)
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        print("in pooling flatten is not supported and would correspond to average")
        # self._initialize()

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):


        stack_inp = [torch.cat([e1_embed.unsqueeze(1), rel_embed.unsqueeze(1), qual_rel_embed[:, i, :].unsqueeze(1),
                   qual_obj_embed[:, i, :].unsqueeze(1)], 1) for i in range(qual_obj_embed.shape[1])]
        stack_inp = torch.stack(stack_inp,dim=1)
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        iterator_size = stk_inp.shape[1]
        x = [self.bn0(stk_inp[:,i,:,:].unsqueeze(1)) for i in range(iterator_size)]

        if self.multi_convs:
            x = [self.m_convs[i](x[i]) for i in range(iterator_size)]
        else:
            x = [self.m_convs(x[i]) for i in range(iterator_size)]

        x = [self.bn1(x[i]) for i in range(iterator_size)]

        x = [x[i].squeeze().view(x[i].shape[0],-1) for i in range(iterator_size)]
        x = torch.stack(x,1)

        if self.pooling == 'min':
            x = torch.min(x,1)[0]
        else:
            x = torch.mean(x,1)
        # x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        # x = self.hidden_drop2(x)
        x = self.bn2(x)
        #  x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

class CompGCN_Transformer_Triples(CompQGCNEncoder):
    model_name = 'CompGCN_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.num_transformer_layers = config['COMPGCNARGS']['T_LAYERS']
        self.num_heads = config['COMPGCNARGS']['T_N_HEADS']
        self.num_hidden = config['COMPGCNARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['COMPGCNARGS']['POSITIONAL']
        self.time = config['COMPGCNARGS']['TIME']  # treat qual values as numbers and pass them through the t_enc
        self.pooling = config['COMPGCNARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['COMPGCNARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['COMPGCNARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)
        # self._initialize()

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)

        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel):
        '''

        :param sub: bs
        :param rel: bs
        :return:

        '''
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        # stk_inp = self.layer_norm(stk_inp)
        # stk_inp = self.hidden_drop2(stk_inp)
        x = self.encoder(stk_inp)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)
        # x = self.hidden_drop2(x)
        # x = self.bn2(x)
        # x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        # x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score