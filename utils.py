""" Some things needed across the board"""
import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
import numpy.random as npr
from collections import namedtuple, defaultdict
from typing import Optional, List, Union, Dict, Callable, Tuple

from mytorch.utils.goodies import Timer, FancyDict, compute_mask

Quint = namedtuple('Quint', 's p o qp qe')

KNOWN_DATASETS = ['fb15k237', 'wd15k', 'fb15k', 'wikipeople', 'wd15k_qonly', 'wd15k_qonly_33', 'wd15k_qonly_66']  # , 'wikipeople', 'jf17k']
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


class UnknownSliceLength(Exception): pass


# Some more nice stuff
lowerall = lambda x: [itm.lower() for itm in x]


# From KrantiKariQA: https://github.com/AskNowQA/KrantikariQA/blob/50142513dcd9858377a8b044ce6a310a1d3e375e/utils/tensor_utils.py
def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def create_neighbourhood_hashes(data: Dict) -> (Dict,Dict):

    print("Creating hop1 hash.")
    hop1 = {}

    for s, p, o in data['train']: #+ data['valid'] + data['test']:
        try:
            # prun already existing triple
            hop1[o].append((p, s))
            hop1[o] = list(set(hop1[o]))
        except:
            # print(traceback.print_exc())
            hop1[o] = [(p, s)]

    print("Creating hop2 hash. This will take around 2-3 mins.")
    hop2 = {}
    for o in hop1.keys():
        _hop1 = hop1[o]
        _hop2 = []
        for p1, o1 in _hop1:
            try:
                _temp = hop1[o1]
                _temp = [tuple([p1] + list(t)) for t in _temp]
                _hop2 = _hop2 + _temp
            except:
                continue
        try:
            hop2[o].append(_hop2)
            hop2[o] = list(set(hop2[o]))
        except:
            hop2[o] = _hop2

    hop2 = {k: list(set(v)) for k, v in hop2.items()}

    return hop1, hop2