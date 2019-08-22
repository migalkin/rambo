#!/usr/bin/env python
# coding: utf-8

# ## Clone OpenKE for benchmark datasets FB15K-237 and WN18
#!git clone https://github.com/thunlp/OpenKE.git
# In[ ]:


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
from corruption import Corruption
from sampler import SimpleSampler


# ## Prepare data

# In[ ]:


# Overwriting data dir
RAW_DATA_DIR = Path('./data/raw_data/fb15k237')
DATASET = 'fb15k237'

np.random.seed(42)
random.seed(42)


# In[ ]:


training_triples = []
valid_triples = []
test_triples = []

with open(RAW_DATA_DIR / "entity2id.txt", "r") as ent_file,     open(RAW_DATA_DIR / "relation2id.txt", "r") as rel_file,     open(RAW_DATA_DIR / "train2id.txt", "r") as train_file,     open(RAW_DATA_DIR / "valid2id.txt", "r") as valid_file,     open(RAW_DATA_DIR / "test2id.txt", "r") as test_file:
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


# In[ ]:


EXPERIMENT_CONFIG = {
    'NUM_ENTITIES': num_entities,
    'NUM_RELATIONS': num_relations,
    'EMBEDDING_DIM': 200,
    'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    'SCORING_FUNCTION_NORM': 1,
    'MARGIN_LOSS': 1,
    'LEARNING_RATE': 0.001,
    'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    'NEGATIVE_SAMPLING_TIMES': 10,
    'BATCH_SIZE': 8192,
    'EPOCHS': 5000,
    'IS_QUINTS': False
}


# In[ ]:


# training_triples.__len__(), valid_triples.__len__()

# def sample_negatives(triple: List) -> List:
#     if np.random.random() < 0.5:
#         # sample subject
#         return [random.choice(range(num_entities)), triple[1], triple[2]]
#     else:
#         # sample object
#         return [triple[0], triple[1], random.choice(range(num_entities))]   def generate_negatives(positive: List[List], times: int):
#     """
#         :param postive: List of the raw data
#         :param times: how many negative samples per positive sample.
#     """
#     negatives = []
#     for pos in tqdm(positive):
#         negatives_per_pos = [sample_negatives(pos) for _ in range(times)]
#         negatives.append(negatives_per_pos)
        
#     return negativestry:
#     negatives = pickle.load(open(PRETRAINING_DATA_DIR / 'fb15k_negatives.pkl', 'rb'))
# except (FileNotFoundError, IOError) as e:
#     # Generate it again
#     warnings.warn("Negative data not pre-generating. Takes three minutes.")
#     negatives = generate_negatives(training_triples + valid_triples, 
#                                    times = EXPERIMENT_CONFIG['NEGATIVE_SAMPLING_TIMES'])

#     # Dump this somewhere
#     with open(PRETRAINING_DATA_DIR / 'fb15k_negatives.pkl', 'wb+') as f:
#         pickle.dump(negatives, f)train_neg = negatives[:len(training_triples)]
# val_neg = negatives[len(training_triples):]data = {'train': {'pos': training_triples, 'neg': train_neg}, 'valid': {'pos': valid_triples, 'neg': val_neg}}
# data_pos = {'train': np.array(training_triples), 'valid': np.array(valid_triples)}np.append(np.array(data_pos['train']), data_pos['valid'], axis=0).shape, len(data_pos['train']), data_pos['valid'].shape
# # ## Model

# ## Training

# In[ ]:


config = EXPERIMENT_CONFIG.copy()
config['DEVICE'] = torch.device('cuda')
model = TransE(config)
model.to(config['DEVICE'])
optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])


wandb.init(project="wikidata-embeddings")
for k, v in config.items():
    wandb.config[k] = v


# In[ ]:


data = {'train': np.array(training_triples), 'valid': np.array(valid_triples)}
_data = {'train': np.array(valid_triples), 'valid': np.array(training_triples)}


# In[ ]:


eval_metrics = [acc, mrr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]
evaluation_valid = EvaluationBench(data, model, config["BATCH_SIZE"]*10, metrics=eval_metrics, _filtered=True)
evaluation_train = EvaluationBench(_data, model, config["BATCH_SIZE"]*10, metrics=eval_metrics, _filtered=True)


# In[ ]:


# Make a loop fn
def simplest_loop(epochs: int,
                  data: dict,
                  opt: torch.optim,
                  train_fn: Callable,
                  predict_fn: Callable,
                  neg_generator: Callable,
                  device: torch.device = torch.device('cpu'),
                  data_fn: Callable = dataiters.SimplestSampler,
                  eval_fn_trn: Callable = default_eval,
                  eval_fn_trn_like_val: Callable = default_eval,
                  eval_fn_val: Callable = default_eval,
                  eval_every: int = 1) -> (list, list, list):
    """
        A fn which can be used to train a language model.

        The model doesn't need to be an nn.Module,
            but have an eval (optional), a train and a predict function.

        Data should be a dict like so:
            {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        Train_fn must return both loss and y_pred
            
        :param @TODO
    """

    train_loss = []
    train_acc = []
    valid_acc = []
    valid_mrr = []
    valid_hits_3, valid_hits_5, valid_hits_10 = [], [], []
    train_acc = []
    train_mrr = []
    train_hits_3, train_hits_5, train_hits_10 = [], [], []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_dl = data_fn(data['train'])

            for pos in tqdm(trn_dl):
                neg = neg_generator.corrupt_batch(pos)
                opt.zero_grad()

                _pos = torch.tensor(pos, dtype=torch.long, device=device)
                _neg = torch.tensor(neg, dtype=torch.long, device=device)

                (pos_scores, neg_scores), loss = train_fn(_pos, _neg)

                per_epoch_tr_acc.append(eval_fn_trn(pos_scores=pos_scores, neg_scores=neg_scores))
                per_epoch_loss.append(loss.item())

                loss.backward()
                opt.step()

        """
            # Val
            Run through the dataset twice.
                1. same as training data (pointwise eval)
                2. One quint (pos+negs) at a time. 
        """ 
        if e % eval_every == 0 :
            with torch.no_grad():
                summary = eval_fn_val()
                per_epoch_vl_acc = summary['metrics']['acc']
                per_epoch_vl_mrr = summary['metrics']['mrr']
                per_epoch_vl_hits_3 = summary['metrics']['hits_at 3']
                per_epoch_vl_hits_5 = summary['metrics']['hits_at 5']
                per_epoch_vl_hits_10 = summary['metrics']['hits_at 10']

                
            with torch.no_grad():
                summary = eval_fn_trn_like_val()
                per_epoch_tr_acc = summary['metrics']['acc']
                per_epoch_tr_mrr = summary['metrics']['mrr']
                per_epoch_tr_hits_3 = summary['metrics']['hits_at 3']
                per_epoch_tr_hits_5 = summary['metrics']['hits_at 5']
                per_epoch_tr_hits_10 = summary['metrics']['hits_at 10']
            
            
        # Bookkeep
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        
        if e % eval_every == 0:
            valid_acc.append(per_epoch_vl_acc)
            valid_mrr.append(per_epoch_vl_mrr)
            valid_hits_3.append(per_epoch_vl_hits_3)
            valid_hits_5.append(per_epoch_vl_hits_5)
            valid_hits_10.append(per_epoch_vl_hits_10)
            
            train_acc.append(per_epoch_tr_acc)
            train_mrr.append(per_epoch_tr_mrr)
            train_hits_3.append(per_epoch_tr_hits_3)
            train_hits_5.append(per_epoch_tr_hits_5)
            train_hits_10.append(per_epoch_tr_hits_10)
        
            print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | "
                  "Vl_c: %(vlacc)0.5f | Vl_mrr: %(vlmrr)0.5f | "
                  "Vl_h3: %(vlh3)0.5f | Vl_h5: %(vlh5)0.5f | Vl_h10: %(vlh10)0.5f | "
                  "tr_c: %(tracc)0.5f | tr_mrr: %(trmrr)0.5f | "
                  "tr_h3: %(trh3)0.5f | tr_h5: %(trh5)0.5f | tr_h10: %(trh10)0.5f | "
                  "Time_Train: %(time).3f min"
                  % {'epo': e,
                     'loss': float(np.mean(per_epoch_loss)),
                     'tracc': float(np.mean(per_epoch_tr_acc)),
                     'vlacc': float(per_epoch_vl_acc),
                     'vlmrr': float(per_epoch_vl_mrr),
                     'vlh3': float(per_epoch_vl_hits_3),
                     'vlh5': float(per_epoch_vl_hits_5),
                     'vlh10': float(per_epoch_vl_hits_10),
                     'tracc': float(per_epoch_tr_acc),
                     'trmrr': float(per_epoch_tr_mrr),
                     'trh3': float(per_epoch_tr_hits_3),
                     'trh5': float(per_epoch_tr_hits_5),
                     'trh10': float(per_epoch_tr_hits_10),
                     'time': timer.interval / 60.0})

            # Wandb stuff
            wandb.log({
                'epoch': e, 
                'loss': float(np.mean(per_epoch_loss)),
                'trn_acc': float(np.mean(per_epoch_tr_acc)),
                'val_acc': float(per_epoch_vl_acc),
                'val_mrr': float(per_epoch_vl_mrr),
                'val_hits@3': float(per_epoch_vl_hits_3),
                'val_hits@5': float(per_epoch_vl_hits_5),
                'val_hits@10': float(per_epoch_vl_hits_10),
                'trn_acc': float(per_epoch_tr_acc),
                'trn_mrr': float(per_epoch_tr_mrr),
                'trn_hits@3': float(per_epoch_tr_hits_3),
                'trn_hits@5': float(per_epoch_tr_hits_5),
                'trn_hits@10': float(per_epoch_tr_hits_10),
            })
            
        else:
            print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | "
                  "Time_Train: %(time).3f min"
                  % {'epo': e,
                     'loss': float(np.mean(per_epoch_loss)),
                     'tracc': float(np.mean(per_epoch_tr_acc)),
                     'time': timer.interval / 60.0})

            # Wandb stuff
            wandb.log({
                'epoch': e, 
                'loss': float(np.mean(per_epoch_loss)),
                'trn_acc': float(np.mean(per_epoch_tr_acc))
            })
        

    return train_acc, valid_acc, valid_acc_like_trn, valid_mrr, train_loss


# In[ ]:


args = {
    "epochs":config['EPOCHS'],
    "data":data,
    "opt": optimizer,
    "train_fn": model,
    "predict_fn": model.predict,
    "device": config['DEVICE'],
    "data_fn": partial(SimpleSampler, bs=config["BATCH_SIZE"]),
    "eval_fn_trn": evaluate_pointwise,
    "eval_fn_val": evaluation_valid.run,
    "eval_fn_trn_like_val": evaluation_train.run,
    "eval_every": 100,
    "neg_generator": Corruption(n=num_entities, position=[0, 2]) # unfiltered for train
}


# In[ ]:


simplest_loop(**args)


# In[ ]:




