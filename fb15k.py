from typing import Optional, Union, List, Callable, Dict
from tqdm import tqdm_notebook as tqdm
from functools import partial
from pathlib import Path
import pandas as  pd
import numpy as np
import traceback
import warnings
import logging
import random
import pickle
import wandb

# MyTorch imports
from mytorch.utils.goodies import *
from mytorch import dataiters

# Local imports
from raw_parser import Quint
from utils import *
from evaluation import *
from models import TransE

# Overwriting data dir
RAW_DATA_DIR = Path('./data/FB15K237')
DATASET = 'fb15k'

# Clamp randomness
np.random.seed(42)
random.seed(42)



