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
import load as l

Quint = namedtuple('Quint', 's p o qp qe')

KNOWN_DATASETS = ['fb15k237', 'wd15k', 'fb15k', 'wikipeople', 'wd15k_qonly', 'wd15k_qonly_33', 'wd15k_qonly_66',
                  'wd15k_33', 'wd15k_66']
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


class UnknownSliceLength(Exception): pass

class NonContinousIDSpace(Exception): pass

# Some more nice stuff
lowerall = lambda x: [itm.lower() for itm in x]


# From KrantiKariQA:
# https://github.com/AskNowQA/KrantikariQA/
# blob/50142513dcd9858377a8b044ce6a310a1d3e375e/utils/tensor_utils.py
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


def create_y_label_for_classifier(dataset: str, statement_length: int,
                                  max_qpairs=50):
    """Creates a mapping for relations and entites wether they exist in
    triples only, or both or just classifier
    The function would not be used with just triples.

    only_triples = 0
    only_qualifiers = 1
    both = 2
    """
    config = {
        'STATEMENT_LEN': statement_length,
        'DATASET': dataset,
        'MAX_QPAIRS': max_qpairs
    }
    dm = l.DataManager()
    data = dm.load(config= config)()


    in_triples_ent, in_qualifier_ent = [], []
    in_triples_rel, in_qualifier_rel = [], []

    for d in data['train']+data['test']+data['valid']:
        subject, predicate, object = d[0], d[1], d[2]
        in_triples_ent.append(subject)
        in_triples_ent.append(object)
        in_triples_rel.append(predicate)

        q_entity = list(set(d[4:][::2]))
        q_rel = list(set(d[3:][::2]))

        in_qualifier_ent = in_qualifier_ent + q_entity
        in_qualifier_rel = in_qualifier_rel + q_rel

    all_entites = list(set(in_triples_ent + in_qualifier_ent))
    all_relations = list(set(in_triples_rel + in_qualifier_rel))

    # only triples
    ent_only_triples = list(set(in_triples_ent) - set(in_qualifier_ent))
    rel_only_triples = list(set(in_triples_rel) - set(in_qualifier_rel))

    # only qualifiers
    ent_only_qualifiers = list(set(in_qualifier_ent) - set(in_triples_ent))
    rel_only_qualifiers = list(set(in_qualifier_rel) - set(in_triples_rel))

    print(f"number of ent only in triples {len(ent_only_triples)}")
    print(f"number of ent only in qualifiers {len(ent_only_qualifiers)}")
    print(f"number of rel only in triples {len(rel_only_triples)}")
    print(f"number of rel only in qualifiers {len(rel_only_qualifiers)  }")

    ent_mapping, rel_mapping = {}, {}

    for e in all_entites:
        if e in ent_only_triples:
            ent_mapping[e] = 0
        elif e in ent_only_qualifiers:
            ent_mapping[e] = 1
        else:
            ent_mapping[e] = 2

    for r in all_relations:
        if r in rel_only_triples:
            rel_mapping[r] = 0
        elif r in rel_only_qualifiers:
            rel_mapping[r] = 1
        else:
            rel_mapping[r] = 2

    # A hack for wd15k as the id space is not continious.
    if dataset == 'wd15k':
        non_cont_id = []
        for index in range(len(ent_mapping)):
            try:
                v = ent_mapping[index]
            except KeyError:
                non_cont_id.append(index)

        if len(non_cont_id) != 1:
            raise NonContinousIDSpace("More than one discontinuity. "
                                      "Code does not support this")
        '''
        @DANGER @DANGER : A random assignment because
         this key does not exist in id space
        '''
        ent_mapping[non_cont_id[0]] = 1 # random assignment.
        # n_cid = non_cont_id[0]
        # new_e_map = {}
        #
        # for key, value in ent_mapping.items():
        #     index = int(key)
        #     if index > n_cid:
        #         index = index - 1
        #     new_e_map[index] = value
        # return new_e_map, rel_mapping


    return ent_mapping, rel_mapping


def create_y_label_for_classifier(dataset: str, statement_length: int,
                                  max_qpairs=50):
    """
        Creates a mapping for relations and entites wether they exist in
            triples only, or both or just classifier
            The function would not be used with just triples.

    only_triples = 0
    only_qualifiers = 1
    both = 2
    """
    config = {
        'STATEMENT_LEN': statement_length,
        'DATASET': dataset,
        'MAX_QPAIRS': max_qpairs
    }
    dm = l.DataManager()
    data = dm.load(config= config)()


    in_triples_ent, in_qualifier_ent = [], []
    in_triples_rel, in_qualifier_rel = [], []

    for d in data['train']+data['test']+data['valid']:
        subject, predicate, object = d[0], d[1], d[2]
        in_triples_ent.append(subject)
        in_triples_ent.append(object)
        in_triples_rel.append(predicate)

        q_entity = list(set(d[4:][::2]))
        q_rel = list(set(d[3:][::2]))

        in_qualifier_ent = in_qualifier_ent + q_entity
        in_qualifier_rel = in_qualifier_rel + q_rel

    all_entites = list(set(in_triples_ent + in_qualifier_ent))
    all_relations = list(set(in_triples_rel + in_qualifier_rel))

    # only triples
    ent_only_triples = list(set(in_triples_ent) - set(in_qualifier_ent))
    rel_only_triples = list(set(in_triples_rel) - set(in_qualifier_rel))

    # only qualifiers
    ent_only_qualifiers = list(set(in_qualifier_ent) - set(in_triples_ent))
    rel_only_qualifiers = list(set(in_qualifier_rel) - set(in_triples_rel))

    print(f"number of ent only in triples {len(ent_only_triples)}")
    print(f"number of ent only in qualifiers {len(ent_only_qualifiers)}")
    print(f"number of rel only in triples {len(rel_only_triples)}")
    print(f"number of rel only in qualifiers {len(rel_only_qualifiers)  }")

    ent_mapping, rel_mapping = {}, {}

    for e in all_entites:
        if e in ent_only_triples:
            ent_mapping[e] = 0
        elif e in ent_only_qualifiers:
            ent_mapping[e] = 1
        else:
            ent_mapping[e] = 2

    for r in all_relations:
        if r in rel_only_triples:
            rel_mapping[r] = 0
        elif r in rel_only_qualifiers:
            rel_mapping[r] = 1
        else:
            rel_mapping[r] = 2

    # A hack for wd15k as the id space is not continious.
    if dataset == 'wd15k':
        non_cont_id = []
        for index in range(len(ent_mapping)):
            try:
                v = ent_mapping[index]
            except KeyError:
                non_cont_id.append(index)

        if len(non_cont_id) != 1:
            raise NonContinousIDSpace("More than one discontinuity. "
                                      "Code does not support this")
        '''
        @DANGER @DANGER : A random assignment because
         this key does not exist in id space
        '''
        ent_mapping[non_cont_id[0]] = 1 # random assignment.
        # n_cid = non_cont_id[0]
        # new_e_map = {}
        #
        # for key, value in ent_mapping.items():
        #     index = int(key)
        #     if index > n_cid:
        #         index = index - 1
        #     new_e_map[index] = value
        # return new_e_map, rel_mapping


    return ent_mapping, rel_mapping


def create_y_label_for_classifier_alternative(dataset: str, statement_length: int,
                                  max_qpairs=50):
    """Creates a mapping for relations and entites wether they exist in
    triples only, or both or just classifier
    The function would not be used with just triples.

    only_triples = 0
    only_qualifiers = 1
    both = 2
    """
    config = {
        'STATEMENT_LEN': statement_length,
        'DATASET': dataset,
        'MAX_QPAIRS': max_qpairs
    }
    dm = l.DataManager()
    data = dm.load(config= config)()

    ent_counter, rel_counter = {}, {}

    for i in range(data['n_entities']):
        ent_counter[i] = [0,0]

    for i in range(data['n_relations']):
        rel_counter[i] = [0,0]


    triples_counter, statment_counter  = 0,0
    for d in data['train']+data['test']+data['valid']:
        ent_counter[d[0]][0] = ent_counter[d[0]][0] + 1
        ent_counter[d[2]][0] = ent_counter[d[2]][0] + 1
        rel_counter[d[1]][0] = rel_counter[d[1]][0] + 1

        for i in d[4:][::2]:
            ent_counter[i][1] = ent_counter[i][1] + 1
        for i in d[3:][::2]:
            rel_counter[i][1] = rel_counter[i][1] + 1

        if d[4] != 0:
            statment_counter = statment_counter + 1
        else:
            triples_counter = triples_counter + 1


    e_map, rel_map = {}, {}

    for key, value in ent_counter.items():
        if value[0]/statment_counter > value[1]/triples_counter:
            e_map[key] = 1
        else:   # Not talking care of the equal. Very rare to encounter.
            e_map[key] = 0

    for key, value in rel_counter.items():
        if value[0]/statment_counter > value[1]/triples_counter:
            rel_map[key] = 1
        else: # Not talking care of the equal. Very rare to encounter.
            rel_map[key] = 0

    counter = 0
    for v, e in e_map.items():
        if e==0:
            counter = counter + 1

    print(f"value of counter is {counter}, remaining {len(e_map)-counter}")
    return e_map, rel_map


def create_neighbourhood_hashes(data: Dict) -> (Dict, Dict):
    # @TODO: did we test it already

    print("Creating hop1 hash.")
    hop1 = {}

    for s, p, o in data['train']:       # + data['valid'] + data['test']:
        hop1.setdefault(o, []).append((s, p))
        hop1[o] = list(set(hop1[o]))

    print("Creating hop2 hash. This will take around 2-3 mins.")
    hop2 = {}
    for o in hop1.keys():
        neighbors_of_o = hop1[o]
        _hop2 = []
        for o1, p1 in neighbors_of_o:
            _temp = hop1.get(o1, [])
            _temp = [tuple(list(t) + [p1]) for t in _temp]
            _hop2 = _hop2 + _temp

        # If no neighbors found (for some reason), don't add it
        if not _hop2:
            continue

        hop2.setdefault(o, []).extend(_hop2)
        hop2[o] = list(set(hop2[o]))

    hop2 = {k: list(set(v)) for k, v in hop2.items()}
    return hop1, hop2


def combine(*args: Union[np.ndarray, list]):
    """
        Used to semi-intelligently combine data splits

        Case A)
            args is a single element, an ndarray. Return as is.
        Case B)
            args are multiple ndarray. Numpy concat them.
        Case C)
            args is a single dict. Return as is.
        Case D)
            args is multiple dicts. Concat individual elements

    :param args: (see above)
    :return: A nd array or a dict
    """

    # Case A, C
    if len(args) == 1 and type(args[0]) is not dict:
        return np.array(args[0])

    if len(args) == 1 and type(args) is dict:
        return args

    # Case B
    if type(args) is tuple and (type(args[0]) is np.ndarray or type(args[0]) is list):
        # Expected shape will be a x n, b x n. Simple concat will do.
        return np.concatenate(args)

    # Case D
    if type(args) is tuple and type(args[0]) is dict:
        keys = args[0].keys()
        combined = {}
        for k in keys:
            combined[k] = np.concatenate([arg[k] for arg in args], dim=-1)
        return combined



#
# if __name__ == '__main__':
#     e_map, rel_map = create_y_label_for_classifier(dataset = 'wd15k',
#                                                    statement_length = -1,
#     max_qpairs = 50)