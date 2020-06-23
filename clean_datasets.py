from pathlib import Path
from typing import Dict
from collections import defaultdict
import random
import pickle
import numpy as np
import re
import json

from load import _get_uniques_, _pad_statements_, count_stats, remove_dups

def load_clean_wikipeople_statements(subtype, maxlen=17) -> Dict:
    """
        :return: train/valid/test splits for the wikipeople dataset in its quints form
    """
    DIRNAME = Path('./data/clean/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                             test_data=raw_tst,
                                                             valid_data=raw_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in raw_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in raw_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in raw_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    if subtype == "triples":
        maxlen = 3
    elif subtype == "quints":
        maxlen = 5

    train, valid, test = _pad_statements_(train, maxlen), \
                         _pad_statements_(valid, maxlen), \
                         _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}

def load_clean_jf17k_statements(subtype, maxlen=15) -> Dict:
    PARSED_DIR = Path('./data/clean/jf17k')

    training_statements = []
    test_statements = []

    with open(PARSED_DIR / 'train.txt', 'r') as train_file, \
        open(PARSED_DIR / 'test.txt', 'r') as test_file:

        for line in train_file:
            training_statements.append(line.strip("\n").split(","))

        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

    st_entities, st_predicates = _get_uniques_(training_statements, test_statements, test_statements)
    st_entities = ['__na__'] + st_entities
    st_predicates = ['__na__'] + st_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    # sample valid as 20% of train
    random.shuffle(training_statements)
    tr_st = training_statements[:int(0.8*len(training_statements))]
    val_st = training_statements[int(0.8*len(training_statements)):]

    train, valid, test = [], [], []
    for st in tr_st:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)

    for st in val_st:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)

    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)


    if subtype == "triples":
        maxlen = 3
    elif subtype == "quints":
        maxlen = 5

    train, valid, test = _pad_statements_(train, maxlen), \
                         _pad_statements_(valid, maxlen), \
                         _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_clean_wd15k(name, subtype, maxlen=43) -> Dict:
    """
        :return: train/valid/test splits for the wd15k datasets
    """
    assert name in ['wd15k', 'wd15k_qonly', 'wd15k_33', 'wd15k_66', 'wd15k_qonly_33', 'wd15k_qonly_66'], \
        "Incorrect dataset"
    assert subtype in ["triples", "quints", "statements"], "Incorrect subtype: triples/quints/statements"

    DIRNAME = Path(f'./data/clean/{name}/{subtype}')

    # Load raw shit
    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                             test_data=raw_tst,
                                                             valid_data=raw_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in raw_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in raw_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in raw_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    if subtype != "triples":
        if subtype == "quints":
            maxlen = 5
        train, valid, test = _pad_statements_(train, maxlen), \
                             _pad_statements_(valid, maxlen), \
                             _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_tkbc(name: str) -> Dict:
    print(f"Loading {name} TKBC dataset...")
    DIRNAME = Path(f'./data/clean/{name}')
    with open(DIRNAME / 'train.pickle', 'rb') as train_file, \
        open(DIRNAME / 'valid.pickle', 'rb') as val_file, \
        open(DIRNAME / 'test.pickle', 'rb') as test_file:

        train_data = pickle.load(train_file).astype(int)
        val_data = pickle.load(val_file).astype(int)
        test_data = pickle.load(test_file).astype(int)

    if name == "tkbc":
        with open(DIRNAME / 'ent_id', 'rb') as entoid, \
            open(DIRNAME / 'rel_id', 'rb') as reltoid, \
            open(DIRNAME / 'ts_id', 'rb') as tstoid:

            entities = pickle.load(entoid)
            rels = pickle.load(reltoid)
            timestamps = pickle.load(tstoid)
            rels['start_time'] = len(rels)
            rels['end_time'] = len(rels)
    else:
        with open(DIRNAME / 'ent_id', 'r') as entoid, \
            open(DIRNAME / 'rel_id', 'r') as reltoid, \
            open(DIRNAME / 'ts_id', 'r') as tstoid:
            entities = {x.strip("\n").split("\t")[0]: int(x.strip("\n").split("\t")[1])+1 for x in entoid.readlines()}
            rels = {x.strip("\n").split("\t")[0]: int(x.strip("\n").split("\t")[1])+1 for x in reltoid.readlines()}
            timestamps = {x.strip("\n").split("\t")[0]: int(x.strip("\n").split("\t")[1]) for x in tstoid.readlines()}
            entities['__na__'] = 0
            rels['__na__'] = 0
            rels['time'] = len(rels)

    total_ents = {}
    for k,v in entities.items():
        total_ents[k] = v
    for k,v in timestamps.items():
        total_ents[k] = v + len(entities)

    # count all entities and relations
    num_entities = len(entities) + len(timestamps)
    num_relations = len(rels)

    if name == "tkbc":
        max_ts = len(timestamps) - 1
        # remove if else for LARGE quals graph with 28M data points
        train = [[x[0], x[1], x[2],
                  rels['start_time'] if int(x[3]) != 0 else 0,
                  int(x[3])+len(entities) if int(x[3]) != 0 else 0,
                  rels['end_time'] if int(x[4]) != max_ts else 0,
                  int(x[4])+len(entities) if int(x[4]) != max_ts else 0] for x in train_data]
        val = [[x[0], x[1], x[2],
                  rels['start_time'] if int(x[3]) != 0 else 0,
                  int(x[3])+len(entities) if int(x[3]) != 0 else 0,
                  rels['end_time'] if int(x[4]) != max_ts else 0,
                  int(x[4])+len(entities) if int(x[4]) != max_ts else 0] for x in val_data]
        test = [[x[0], x[1], x[2],
                  rels['start_time'] if int(x[3]) != 0 else 0,
                  int(x[3])+len(entities) if int(x[3]) != 0 else 0,
                  rels['end_time'] if int(x[4]) != max_ts else 0,
                  int(x[4])+len(entities) if int(x[4]) != max_ts else 0] for x in test_data]
    elif name == "yago15k":
        max_ts = len(timestamps)
        train = [[x[0]+1, x[1]+1, x[2]+1,
                  rels['time'] if int(x[3]) != max_ts else 0,
                  int(x[3]) + len(entities) if int(x[3]) != max_ts else 0] for x in train_data]
        val = [[x[0]+1, x[1]+1, x[2]+1,
                rels['time'] if int(x[3]) != max_ts else 0,
                int(x[3]) + len(entities) if int(x[3]) != max_ts else 0] for x in val_data]
        test = [[x[0]+1, x[1]+1, x[2]+1,
                 rels['time'] if int(x[3]) != max_ts else 0,
                 int(x[3]) + len(entities) if int(x[3]) != max_ts else 0] for x in test_data]
    else:
        train = [[x[0]+1, x[1]+1, x[2]+1, rels['time'], int(x[3]) + len(entities)] for x in train_data]
        val = [[x[0]+1, x[1]+1, x[2]+1, rels['time'], int(x[3]) + len(entities)] for x in val_data]
        test = [[x[0]+1, x[1]+1, x[2]+1, rels['time'], int(x[3]) + len(entities)] for x in test_data]


    print(f"Found {len(timestamps)} timestamps")

    return {"train": train, "valid": val, "test": test, "n_entities": num_entities,
            "n_relations": num_relations, 'e2id': total_ents, 'r2id': (rels, timestamps)}


def load_yago15k_quals():
    DIRNAME = Path(f'./data/clean/yago15k_quals')
    ents = set()
    rels = set()
    timestamps = defaultdict(int)

    dataz = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    files = ['train', 'valid', 'test']
    for f in files:
        with open(DIRNAME / f, "r") as source:
            for line in source.readlines():
                statement = line.strip().split('\t')
                if len(statement) == 4:
                    continue
                elif len(statement) > 4:
                    s, p, o, qp, qe = statement
                    timestamp = int(re.search(r'\d+', qe).group())
                    timestamps[str(timestamp)] += 1
                    ents.add(s)
                    ents.add(o)
                    rels.add(p)
                    rels.add(qp)
                    dataz[f][(s,p,o)].append((qp, str(timestamp)))
                else:
                    s, p, o = statement
                    ents.add(s)
                    ents.add(o)
                    rels.add(p)
                    dataz[f][(s,p,o)].append(('__na__', '__na__'))

    ents = sorted(list(ents)) + sorted(list(timestamps.keys()))
    total_ents  = ['__na__'] + ents
    total_rels = ['__na__'] + sorted(list(rels))
    entoid = {x: i for (i, x) in enumerate(total_ents)}
    reltoid = {x: i for (i, x) in enumerate(total_rels)}
    tstoid = {str(x): (x-1) for (i,x) in enumerate(sorted([int(k) for k in list(timestamps.keys())]))}

    num_entities = len(entoid)
    num_relations = len(reltoid)
    num_timestamps = len(timestamps)

    # create final data
    train, val, test = [], [], []
    for triple, quals in dataz['train'].items():
        s = [entoid[triple[0]], reltoid[triple[1]], entoid[triple[2]]]
        for q in quals:
            s = s + [reltoid[q[0]], entoid[q[1]]]
        train.append(s)
    for triple, quals in dataz['valid'].items():
        s = [entoid[triple[0]], reltoid[triple[1]], entoid[triple[2]]]
        for q in quals:
            s = s + [reltoid[q[0]], entoid[q[1]]]
        val.append(s)
    for triple, quals in dataz['test'].items():
        s = [entoid[triple[0]], reltoid[triple[1]], entoid[triple[2]]]
        for q in quals:
            s = s + [reltoid[q[0]], entoid[q[1]]]
        test.append(s)

    train, val, test = _pad_statements_(train, 7), _pad_statements_(val, 7), _pad_statements_(test, 7)
    print(f"Found {len(timestamps)} timestamps")

    return {"train": train, "valid": val, "test": test, "n_entities": num_entities,
            "n_relations": num_relations, 'e2id':entoid, 'r2id': (total_rels, tstoid)}


def load_nodecl_dataset(name, subtype, task, maxlen=43) -> Dict:
    """

    :param name: dataset name wd15k/wd15k_33/wd15k_66/wd15k_qonly
    :param subtype: triples/statements
    :param task: so/full predict entities at sub/obj positions (for triples/statements) or all nodes incl quals
    :param maxlen: max statement length
    :return: train/valid/test splits for the wd15k datasets
    """

    assert name in ['wd15k', 'wd15k_qonly', 'wd15k_33', 'wd15k_66'], "Incorrect dataset"
    assert subtype in ["triples", "statements"], "Incorrect subtype: triples/statements"


    DIRNAME = Path(f'./data/clean/{name}/{subtype}')

    with open(DIRNAME / 'nc_edges.txt', 'r') as f:
        edges = []
        for line in f.readlines():
            edges.append(line.strip("\n").split(","))

    with open(DIRNAME / 'nc_entities.txt', 'r') as f:
        statement_entities = [l.strip("\n") for l in f.readlines()]

    with open(DIRNAME / 'nc_rels.txt', 'r') as f:
        statement_predicates = [l.strip("\n") for l in f.readlines()]

    if subtype == "triples":
        task = "so"

    with open(DIRNAME / f'nc_train_{task}_labels.json', 'r') as f:
        train_labels = json.load(f)

    with open(DIRNAME / f'nc_val_{task}_labels.json', 'r') as f:
        val_labels = json.load(f)

    with open(DIRNAME / f'nc_test_{task}_labels.json', 'r') as f:
        test_labels = json.load(f)


    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    graph, train_mask, val_mask, test_mask = [], [], [], []
    for st in edges:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        graph.append(id_st)

    if subtype != "triples":
        graph = _pad_statements_(graph, maxlen)

    # if subtype == "triples":
    #     graph = remove_dups(graph)

    train_mask = [entoid[e] for e in train_labels]
    val_mask = [entoid[e] for e in val_labels]
    test_mask = [entoid[e] for e in test_labels]

    all_labels = sorted(list(set([
        label for v in list(train_labels.values())+list(val_labels.values())+list(test_labels.values()) for label in v])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    train_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in train_labels.items()}
    val_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in val_labels.items()}
    test_y = {entoid[k]: [label2id[vi] for vi in v] for k, v in test_labels.items()}

    return {"train_mask": train_mask, "valid_mask": val_mask, "test_mask": test_mask,
            "train_y": train_y, "val_y": val_y, "test_y": test_y,
            "all_labels": all_labels, "label2id": label2id, "id2label": id2label,
            "n_entities": len(st_entities), "n_relations": len(st_predicates),
            "e2id": entoid, "r2id": prtoid, "graph": graph
            }


if __name__ == "__main__":
    #count_stats(load_clean_wd15k("wd15k","statements",43))
    load_nodecl_dataset("wd15k_qonly", "statements", "so", 15)