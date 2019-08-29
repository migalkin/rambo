""" Some things needed across the board"""
import torch
import pickle
import numpy as np
import numpy.random as npr
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Union, Dict
from collections import namedtuple

from mytorch.utils.goodies import Timer

Quint = namedtuple('Quint', 's p o qp qe')

KNOWN_DATASETS = ['fb15k237', 'wd15k', 'wikipeople', 'jf17k']
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


class UnknownSliceLength(Exception): pass

# Load data from disk
with open(PARSED_DATA_DIR / 'parsed_raw_data.pkl', 'rb') as f:
    raw_data = pickle.load(f)

# with open('./data/parsed_data/parsed_raw_data.pkl', 'rb') as f:
#     raw_data = pickle.load(f)

entities, predicates = [], []

for quint in raw_data:
    entities += [quint[0], quint[2]]
    if quint[4]:
        entities.append(quint[4])

    predicates.append(quint[1])
    if quint[3]:
        predicates.append(quint[3])

entities = sorted(list(set(entities)))
predicates = sorted(list(set(predicates)))

entities = ['__na__', '__pad__'] + entities
predicates = ['__na__', '__pad__'] + predicates

# uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
entoid = {pred: i for i, pred in enumerate(entities)}
prtoid = {pred: i for i, pred in enumerate(predicates)}






