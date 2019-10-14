#!/usr/bin/env python
# coding: utf-8

# # TransE
# 
# **Data**: s, p, o, qp, qe
# 
# $s,p,o,qe,qp \in R^{d}$

# ## Challenge
# 1. Model
# 2. Sampling Code
# 
# 
# **Model** from [pyKeen](https://github.com/SmartDataAnalytics/PyKEEN/blob/master/src/pykeen/kge_models/trans_e.py)

# In[1]:


import logging
# from dataclasses import dataclass
from typing import Dict

from tqdm import tqdm_notebook as tqdm
import pandas as  pd
import numpy as np
import traceback
import warnings
import random
import pickle

import torch
import torch.autograd
from torch import nn

from functools import partial
from typing import Optional, Union, List, Callable

# MyTorch imports
from mytorch.utils.goodies import *
from mytorch import dataiters

import wandb


# In[2]:


# Local imports
# from raw_parser import *
from utils import *
from corruption import sample_negatives


# In[3]:


np.random.seed(42)
random.seed(42)


# ## Basic Stats
# n_entities, n_relations etc
# Load data from disk
with open('./parsed_raw_data.pkl', 'rb') as f:
    raw_data = pickle.load(f)entities, predicates = [], []

for quint in raw_data:
    entities += [quint[0], quint[2]]
    if quint[4]:
        entities.append(quint[4])
        
    predicates.append(quint[1])
    if quint[3]: 
        predicates.append(quint[3])
    
entities = list(set(entities))
predicates = list(set(predicates))
        
print(len(entities), len(predicates))
# ## TransE Neural Model
# Code from PyKeen

# In[4]:


EXPERIMENT_CONFIG = {
    'NUM_ENTITIES': len(entities),
    'NUM_RELATIONS': len(predicates),
    'EMBEDDING_DIM': 200,
    'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    'SCORING_FUNCTION_NORM': 1,
    'MARGIN_LOSS': 4,
    'LEARNING_RATE': 0.001,
    'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    'NEGATIVE_SAMPLING_TIMES': 10,
    'BATCH_SIZE': 256
}


# In[5]:


def slice_triples(triples: torch.Tensor) -> List[torch.Tensor]:
    """ Slice in 3 or 5 as needed """
    return triples[:,0], triples[:,1], triples[:,2], triples[:,3], triples[:, 4]
    

# Test slice_triples
t = torch.randint(0, 100, (10, 5))
t, slice_triples(t)
# In[6]:


class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = None
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EXPERIMENT_CONFIG['EMBEDDING_DIM'], 
                    EXPERIMENT_CONFIG['MARGIN_LOSS'], 
                    EXPERIMENT_CONFIG['LEARNING_RATE']]

    def __init__(self, config: Dict) -> None:
        super().__init__()

        # Device selection
        self.device = config['DEVICE']

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

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss


# In[7]:


class TransE(BaseModule):
    """An implementation of TransE [borders2013]_.
     This model considers a relation as a translation from the head to the tail entity.
    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.
    .. seealso::
       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = 'TransE MM'
    margin_ranking_loss_size_average: bool = True
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    
    def __init__(self, config) -> None:
        super().__init__(config)

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
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        
        head_embeddings, relation_embeddings, tail_embeddings, qual_relation_embeddings, qual_entity_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings, qual_relation_embeddings, qual_entity_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings, qual_relation_embeddings, qual_entity_embeddings):
        """
            Compute the scores based on the head, relation, and tail embeddings.
        
        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param qual_relation_embeddings: embeddings of qualifier relation of dimension batchsize x embedding_dim
        :param qual_entity_embeddings: embeddings of qualifier entity of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings                                         + qual_relation_embeddings - qual_entity_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        heads, relations, tails, qual_relations, qual_entities = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
            self._get_relation_embeddings(qual_relations),
            self._get_entity_embeddings(qual_entities)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)


# ## Data Sampling etc
# 
# TODO: Figuring out 
# 1. How to perturb data
# 2. What ratios
# 3. Train/Test splits
# 4. Test task
# 
# **CODE ORG**
# Instead of perturbing on the fly, we can pre-perturb it and then do splits and batches.
# 
# **How to deal with cases w no qualifiers**
#     - treat them differently?
for datum in data:
    case 1 (enter with p) -> fuck s
    case 2 (enter with p) -> fuck o
    case 3 (enter with p) -> fuck r
    case 3 (enter with p)
        -> either qe, qp exist
            -> either you can remove qe, qp | or replace qe & qp | or replace qe | or replace qp
        -> they dont exist
            -> either you add qe qp
    check if corrupted thing doesnt exist in dataset.
# In[8]:


# TODO: Play with probs
# [ p(s), p(r), p(o), p(q) ]
# prob = [0.3, 0.0, 0.2, 0.5]


# In[9]:


sample_negatives(raw_data[0], EXPERIMENT_CONFIG['NEGATIVE_SAMPLING_PROBS']), raw_data[0]

negatives = [sample_negatives(datum, prob) for datum in raw_data]
# In[10]:


def generate_negatives(positive: List[Quint], probs: List[float], times: int):
    """
        :param postive: List of the raw data
        :param probs: List of probabilities to generate neg data following [ p(s), p(r), p(o), p(q) ]
        :param times: how many negative samples per positive sample.
    """
    negatives = []
    for pos in tqdm(positive):
        negatives_per_pos = [sample_negatives(pos, probs) for _ in range(times)]
        negatives.append(negatives_per_pos)
        
    return negatives


# In[11]:


try:
    negatives = pickle.load(open(PRETRAINING_DATA_DIR / 'negatives.pkl', 'rb'))
except (FileNotFoundError, IOError) as e:
    # Generate it again
    warnings.warn("Negative data not pre-generating. Takes three minutes.")
    negatives = generate_negatives(raw_data, 
                                   probs = EXPERIMENT_CONFIG['NEGATIVE_SAMPLING_PROBS'],
                                   times = EXPERIMENT_CONFIG['NEGATIVE_SAMPLING_TIMES'])

    # Dump this somewhere
    with open(PRETRAINING_DATA_DIR / 'negatives.pkl', 'wb+') as f:
        pickle.dump(negatives, f)


# In[12]:


_positives, _negatives = [], []
for quint in raw_data:
    _pos = [
        entoid[quint[0]], 
        prtoid[quint[1]], 
        entoid[quint[2]], 
        prtoid[quint[3] if quint[3] else '__na__'], 
        entoid[quint[4] if quint[4] else '__na__']
    ]
    _positives.append(_pos)
    
for negative in negatives:
    _negative = []
    for quint in negative:
        _neg = [
            entoid[quint[0]], 
            prtoid[quint[1]], 
            entoid[quint[2]], 
            prtoid[quint[3] if quint[3] else '__na__'], 
            entoid[quint[4] if quint[4] else '__na__']
        ] 
        _negative.append(_neg)
    _negatives.append(_negative)

# _positives, _negatives = np.array(_positives), np.array(_negatives)


# In[13]:


_positives[:10], _negatives[0]

# Convert raw_data (pos) and negatives to int lists/arrays
_positives = [[uritoid[uri] if uri else uritoid['__na__'] for uri in quint] for quint in raw_data]
_negatives = [[[uritoid[uri] if uri else uritoid['__na__'] for uri in quint] for quint in negative] for negative in negatives]_positives[:10], _negatives[0]
# In[14]:


# Try sampling (test), skip in prod
sampler = QuintRankingSampler({"pos": _positives, "neg": _negatives }, bs=4000)
for x in tqdm(sampler):
    pass


# ## Training
# 
# Make a model.
# 
# See if we can use something for loop.

# In[18]:


config = EXPERIMENT_CONFIG.copy()
config['DEVICE'] = torch.device('cpu')
model = TransE(config)
optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])


wandb.init(project="wikidata-embeddings")
for k, v in config.items():
    wandb.config[k] = v


# In[16]:





# In[52]:


_positives.__class__


# In[53]:


# Split data in train and valid
index = np.arange(len(_positives))
np.random.shuffle(index)
train_index, valid_index = index[:int(index.shape[0]*0.8)], index[int(index.shape[0]*0.8):]
train_pos = [_positives[i] for i in train_index]
valid_pos = [_positives[i] for i in valid_index]
train_neg = [_negatives[i] for i in train_index]
valid_neg = [_negatives[i] for i in valid_index]
data = {'train': {'pos': train_pos, 'neg': train_neg}, 'valid': {'pos': valid_pos, 'neg': valid_neg}}


# In[54]:


def evaluate(*args, **kwargs):
    ...


# In[19]:


# Make a loop fn
def simplest_loop(epochs: int,
                  data: dict,
                  opt: torch.optim,
                  train_fn: Callable,
                  predict_fn: Callable,
                  device: torch.device = torch.device('cpu'),
                  data_fn: classmethod = dataiters.SimplestSampler,
                  eval_fn: Callable = default_eval) -> (list, list, list):
    """
        A fn which can be used to train a language model.

        The model doesn't need to be an nn.Module,
            but have an eval (optional), a train and a predict function.

        Data should be a dict like so:
            {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        Train_fn must return both loss and y_pred

        :param epochs: number of epochs to train for
        :param data: a dict having keys train_x, test_x, train_y, test_y
        :param device: torch device to create new tensor from data
        :param opt: optimizer
        :param loss_fn: loss function
        :param train_fn: function to call with x and y
        :param predict_fn: function to call with x (test)
        :param data_fn: a class to which we can pass X and Y, and get an iterator.
        :param eval_fn: (optional) function which when given pred and true, returns acc
        :return: traces
    """

    train_loss = []
    train_acc = []
    valid_acc = []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
#         per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])

            for pos, neg in tqdm(trn_dl):
                opt.zero_grad()

                _pos = torch.tensor(pos, dtype=torch.long, device=device)
                _neg = torch.tensor(neg, dtype=torch.long, device=device)

                loss = train_fn(_pos, _neg)
#                 loss = loss_fn(y_pred, _y)

#                 per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()
                opt.step()

#         # Val
#         with torch.no_grad():

#             per_epoch_vl_acc = []
#             for x, y in tqdm(val_dl):
#                 _x = torch.tensor(x, dtype=torch.long, device=device)
#                 _y = torch.tensor(y, dtype=torch.long, device=device)

#                 y_pred = predict_fn(_x)

#                 per_epoch_vl_acc.append(eval_fn(y_pred, _y).item())

        # Bookkeep
#         train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
#         valid_acc.append(np.mean(per_epoch_vl_acc))

#         print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
#               % {'epo': e,
#                  'loss': float(np.mean(per_epoch_loss)),
#                  'tracc': float(np.mean(per_epoch_tr_acc)),
#                  'vlacc': float(np.mean(per_epoch_vl_acc)),
#                  'time': timer.interval / 60.0})
        print("Epoch: %(epo)03d | Loss: %(loss).5f | Time: %(time).3f min"
              % {'epo': e,
                 'loss': float(np.mean(per_epoch_loss)),
                 'time': timer.interval / 60.0})
        # Wandb stuff
        wandb.log({'epoch': e, 'loss': float(np.mean(per_epoch_loss))})

    return train_acc, valid_acc, train_loss

epochs: int,
                  data: dict,
                  opt: torch.optim,
                  train_fn: Callable,
                  predict_fn: Callable,
                  device: torch.device = torch.device('cpu'),
                  data_fn: classmethod = dataiters.SimplestSampler,
                  eval_fn: Callable = default_eval
# In[56]:


args = {
    "epochs":2,
    "data":data,
    "opt": optimizer,
    "train_fn": model,
    "predict_fn": model.predict,
    "device": config['DEVICE'],
    "data_fn": partial(QuintRankingSampler, bs=config["BATCH_SIZE"]),
    "eval_fn": evaluate
}


# In[57]:


simplest_loop(**args)


# In[ ]:




