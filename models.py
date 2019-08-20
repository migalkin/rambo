"""
    A file which contains all relevant models for this work.
    Most of these have been taken, reappropriated from [PyKeen](https://github.com/SmartDataAnalytics/PyKEEN/)
"""

# Torch imports
import torch
from torch import nn
import torch.autograd

import numpy as np
from typing import List, Optional, Dict, Tuple

# Local imports
from utils import *


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
        )

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss


def slice_triples(triples: torch.Tensor, slices=5) :
    """ Slice in 3 or 5 as needed """
    if slices == 5:
        return triples[:, 0], triples[:, 1], triples[:, 2], triples[:, 3], triples[:, 4]
    elif slices == 3:
        return triples[:, 0], triples[:, 1], triples[:, 2]
    else:
        raise UnknownSliceLength


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

    def __init__(self, config) -> None:
        super().__init__(config)

        self.margin_ranking_loss_size_average: bool = True
        self.entity_embedding_max_norm: Optional[int] = None
        self.entity_embedding_norm_type: int = 2
        self.model_name = 'TransE MM'
        self.quints = config['IS_QUINTS']

        # Embeddings
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']
        self.relation_embeddings = nn.Embedding(config['NUM_RELATIONS'], config['EMBEDDING_DIM'])

        self.config = config

        self._initialize()

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

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores

    def forward(self, batch_positives, batch_negatives) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return (positive_scores, negative_scores), loss

    def _score_triples(self, triples) -> torch.Tensor:
        """ Get triple/quint embeddings, and compute scores """
        scores = self._compute_scores(self._get_triple_embeddings(triples))
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings,
                        qual_relation_embeddings=None, qual_entity_embeddings=None):
        """
            Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param qual_entity_embeddings: embeddings of qualifier relations of dimensinos batchsize x embeddig_dim
        :param qual_relation_embeddings: embeddings of qualifier entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        if qual_relation_embeddings is not None and qual_entity_embeddings is not None:
            sum_res = sum_res + qual_relation_embeddings - qual_entity_embeddings
        # Add the vector element wise
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        if self.quints:
            heads, relations, tails, qual_relations, qual_entities = slice_triples(triples, 5)
            return (
                self._get_entity_embeddings(heads),
                self._get_relation_embeddings(relations),
                self._get_entity_embeddings(tails),
                self._get_relation_embeddings(qual_relations),
                self._get_entity_embeddings(qual_entities)
            )

        else:
            heads, relations, tails = slice_triples(triples, 3 )
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)