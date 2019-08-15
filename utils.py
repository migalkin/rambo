""" Some things needed across the board"""
from collections import namedtuple
from pathlib import Path
import pickle

Quint = namedtuple('Quint', 's p o qp qe')
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')

# Load data from disk
with open(PARSED_DATA_DIR / 'parsed_raw_data.pkl', 'rb') as f:
    raw_data = pickle.load(f)

entities, predicates = [], []

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
