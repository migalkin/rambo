from tqdm import tqdm_notebook as tqdm
from pprint import pprint
import pickle
import pandas as pd
from pathlib import Path
from collections import namedtuple
import operator
import numpy as np
from typing import Tuple, List, Dict
from parse_wd15k import Quint
import argparse

RAW_DATA_DIR = Path('./data/raw_data/wd15k')
WD15K_DATA_DIR = Path('./data/parsed_data/wd15k')
WD15K_QONLY = Path('./data/parsed_data/wd15k_qonly')
WD15K_QONLY_33 = Path('./data/parsed_data/wd15k_qonly_33')
WD15K_QONLY_66 = Path('./data/parsed_data/wd15k_qonly_66')
np.random.seed(42)

def split_statements(path, filter_qualifiers:bool, keep_prob: float) -> Tuple[List, List, List]:
    """
    Split into 70/10/20
    :param path: path to the cleaned_wd15k.pkl file
    :return: splits for train, valid, test
    """
    ds = pickle.load(open(path / "cleaned_wd15k.pkl", "rb"))
    if filter_qualifiers:
        ds = keep_only_quals(ds)
        ds_stats(ds)
        ds = remove_qualifiers(ds, keep_prob)
        ds_stats(ds)
    ind = np.arange(len(ds))
    np.random.shuffle(ind)
    train_indices, valid_indices, test_indices = ind[:int(0.7 * len(ind))], ind[int(0.7 * len(ind)):int(
        0.8 * len(ind))], ind[int(0.8 * len(ind)):]

    train = [ds[i] for i in train_indices]
    valid = [ds[i] for i in valid_indices]
    test = [ds[i] for i in test_indices]

    ds_stats(train)
    ds_stats(valid)
    ds_stats(test)

    return (train, valid, test)

def generate_triples(dataset: List[Dict]) -> List[Tuple]:
    """
    Strip off the statements from qualifiers and return only unique triples (s, p, o)
    :param dataset: List of statements
    :return: List of triples
    """
    cleaned_triples = set()
    for statement in dataset:
        triple = (statement['s'], statement['p'], statement['o'])
        cleaned_triples.add(triple)
    return list(cleaned_triples)


def generate_quints(dataset: List[Dict]) -> List[Tuple]:
    """
    Generate quints (s, p, o, qp, qe) AND if a statement contains N (qp, qe) pairs - generate N quints
    :param dataset: List of statements
    :return: List of quints
    """
    unique_quints = set()
    for statement in dataset:
        if len(statement['qualifiers']) > 0:
            for q in statement['qualifiers']:
                unique_quints.add(Quint(s=statement['s'], p=statement['p'], o=statement['o'], qp=q[0], qe=q[1]))
        else:
            unique_quints.add(Quint(s=statement['s'], p=statement['p'], o=statement['o'], qp=None, qe=None))
    return list(unique_quints)

def generate_full_statements(dataset: List[Dict]) -> List[Tuple]:
    """
    Generate tuples of arbitrary length (s, p, o, [ (qp, qe) x N ])
    :param dataset: List of statements
    :return: List of tuples
    """
    result = set()
    for statement in dataset:
        triple = [statement['s'], statement['p'], statement['o']]
        for q in statement['qualifiers']:
            triple.extend([q[0], q[1]])
        result.add(tuple(triple))
    return list(result)


def ds_stats(dataset: List[Dict]):
    count_with_qual = len([a for a in dataset if len(a['qualifiers']) > 0])
    print(f"Overall with qualifiers: {count_with_qual}")

    max_quals = list(set([len(a['qualifiers']) for a in dataset if len(a['qualifiers']) > 0]))
    for m in sorted(max_quals):
        print(f"{m}: {len([a for a in dataset if len(a['qualifiers']) == m])}")


def remove_qualifiers(ds: List[Dict], keep_prob: float = 0.33) -> List[Dict]:
    for statement in ds:
        if np.random.random() > keep_prob:
            statement['qualifiers'] = []
    return ds


def keep_only_quals(ds: List[Dict]):
    """

    :param ds: set of statements
    :return: dataset with statements that have at least one qualifier
    """
    return [i for i in ds if len(i["qualifiers"]) > 0]


if __name__ == "__main__":
    only_q = True
    keep_prob = 1.0
    assert keep_prob <= 1.0, "You can't ask to keep more than 100% of qualifiers"
    assert keep_prob >= 0.0, "You can't ask to keep less than 0% of qualifiers"
    # split the dataset into train/val/test
    train, val, test = split_statements(RAW_DATA_DIR, filter_qualifiers=only_q, keep_prob=keep_prob)
    # convert each chunk into triples/quints/full quints
    train_triples, valid_triples, test_triples = generate_triples(train), generate_triples(val), generate_triples(test)
    train_quints, valid_quints, test_quints = generate_quints(train), generate_quints(val), generate_quints(test)
    train_statements, valid_statements, test_statements = generate_full_statements(train), generate_full_statements(val), generate_full_statements(test)
    print(f"Triples: {len(train_triples)} train, {len(valid_triples)} val, {len(test_triples)} test")
    print(f"Quints: {len(train_quints)} train, {len(valid_quints)} val, {len(test_quints)} test")
    print(f"Statements: {len(train_statements)} train, {len(valid_statements)} val, {len(test_statements)} test")
    # write files
    # raise Exception
    if not only_q:
        output_dir = WD15K_DATA_DIR
    else:
        output_dir = WD15K_QONLY
    with open(output_dir / 'train_quints.pkl', 'wb+') as f:
        pickle.dump(train_quints, f)
        print(f" Writing {len(train_quints)} training quints")

    with open(output_dir / 'valid_quints.pkl', 'wb+') as f:
        pickle.dump(valid_quints, f)
        print(f" Writing {len(valid_quints)} valid quints")

    with open(output_dir / 'test_quints.pkl', 'wb+') as f:
        pickle.dump(test_quints, f)
        print(f" Writing {len(test_quints)} test quints")

    with open(output_dir / 'train_triples.pkl', 'wb+') as f:
        pickle.dump(train_triples, f)
        print(f" Writing {len(train_triples)} training triples")

    with open(output_dir / 'valid_triples.pkl', 'wb+') as f:
        pickle.dump(valid_triples, f)
        print(f" Writing {len(valid_triples)} valid triples")

    with open(output_dir / 'test_triples.pkl', 'wb+') as f:
        pickle.dump(test_triples, f)
        print(f" Writing {len(test_triples)} test triples")

    with open(output_dir / 'train_statements.pkl', 'wb+') as f:
        pickle.dump(train_statements, f)
        print(f" Writing {len(train_statements)} training statements")

    with open(output_dir / 'valid_statements.pkl', 'wb+') as f:
        pickle.dump(valid_statements, f)
        print(f" Writing {len(valid_statements)} valid statements")

    with open(output_dir / 'test_statements.pkl', 'wb+') as f:
        pickle.dump(test_statements, f)
        print(f" Writing {len(test_statements)} test statements")

