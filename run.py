"""
    The file which actually manages to run everything
"""

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
from parse_wd15k import Quint
from utils import *
from evaluation import EvaluationBench, acc, mrr, mr, hits_at, evaluate_pointwise
from models import TransE
from corruption import Corruption
from sampler import SimpleSampler
from loops import training_loop

"""
    CONFIG Things
"""
DATASET = 'wd15k'

# Clamp the randomness
np.random.seed(42)
random.seed(42)

EXPERIMENT_CONFIG = {
    'EMBEDDING_DIM': 50,
    'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    'SCORING_FUNCTION_NORM': 1,
    'MARGIN_LOSS': 1,
    'LEARNING_RATE': 0.001,
    'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    'NEGATIVE_SAMPLING_TIMES': 10,
    'BATCH_SIZE': 64,
    'EPOCHS': 1000,
    'IS_QUINTS': False,
    'EVAL_EVERY': 10,
    'WANDB': True,
    'RUN_TESTBENCH_ON_TRAIN': True
}


if __name__ == "__main__":

    # @TODO pull things from argparse

    RAW_DATA_DIR = Path('./data/raw_data/fb15k237')
    DATASET = 'fb15k237'

    training_triples = []
    valid_triples = []
    test_triples = []

    with open(RAW_DATA_DIR / "entity2id.txt", "r") as ent_file, \
            open(RAW_DATA_DIR / "relation2id.txt", "r") as rel_file, \
            open(RAW_DATA_DIR / "train2id.txt", "r") as train_file, \
            open(RAW_DATA_DIR / "valid2id.txt", "r") as valid_file, \
            open(RAW_DATA_DIR / "test2id.txt", "r") as test_file:
        num_entities = int(next(ent_file).strip("\n"))
        num_relations = int(next(rel_file).strip("\n"))
        num_trains = int(next(train_file).strip("\n"))
        for line in train_file:
            triple = line.strip("\n").split(" ")
            training_triples.append([int(triple[0]), int(triple[2]), int(triple[1])])

        num_valid = int(next(valid_file).strip("\n"))
        for line in valid_file:
            triple = line.strip("\n").split(" ")
            valid_triples.append([int(triple[0]), int(triple[2]), int(triple[1])])

        num_test = int(next(test_file).strip("\n"))
        for line in test_file:
            triple = line.strip("\n").split(" ")
            test_triples.append([int(triple[0]), int(triple[2]), int(triple[1])])

    EXPERIMENT_CONFIG['NUM_ENTITIES'] = num_entities
    EXPERIMENT_CONFIG['NUM_RELATIONS'] = num_relations

    """
        Make ze model
    """
    config = EXPERIMENT_CONFIG.copy()
    config['DEVICE'] = torch.device('cuda')
    model = TransE(config)
    model.to(config['DEVICE'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])

    if config['WANDB']:
        wandb.init(project="wikidata-embeddings")
        for k, v in config.items():
            wandb.config[k] = v

    """
        Prepare test benches
    """
    data = {'index': np.array(training_triples + test_triples), 'eval': np.array(valid_triples)}
    _data = {'index': np.array(valid_triples + test_triples), 'eval': np.array(training_triples)}

    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]
    evaluation_valid = EvaluationBench(data, model, 8000, metrics=eval_metrics, _filtered=True)
    evaluation_train = EvaluationBench(_data, model, 8000, metrics=eval_metrics, _filtered=True, trim=0.01)

    # RE-org the data
    data = {'train': data['index'], 'valid': data['valid']}

    args = {
        "epochs": config['EPOCHS'],
        "data": data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=num_entities, position=[0, 2]),  # unfiltered for train
        "device": config['DEVICE'],
        "data_fn": partial(SimpleSampler, bs=config["BATCH_SIZE"]),
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run,
        "trn_testbench": evaluation_train.run,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": config['RUN_TESTBENCH_ON_TRAIN']
    }

    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)



