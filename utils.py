""" Some things needed across the board"""
import torch
import pickle
import numpy as np
import numpy.random as npr
import torch.nn as nn
from pathlib import Path
from collections import namedtuple
from typing import Optional, List, Union, Dict, Callable, Tuple

from mytorch.utils.goodies import Timer, FancyDict

Quint = namedtuple('Quint', 's p o qp qe')

KNOWN_DATASETS = ['fb15k237', 'wd15k', 'fb15k', 'wikipeople']  # , 'wikipeople', 'jf17k']
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


class UnknownSliceLength(Exception): pass


# Some more nice stuff
lowerall = lambda x: [itm.lower() for itm in x]
