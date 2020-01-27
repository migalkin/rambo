"""
    File which enables easily loading any dataset we need
"""
import json
from tqdm import tqdm
from functools import partial

from utils import *


def _conv_to_our_format_(data):
    conv_data = []
    for datum in tqdm(data):
        conv_datum = []

        # Get head and tail rels
        head, tail, rel_h, rel_t = None, None, None, None
        for rel, val in datum.items():
            if rel[-2:] == '_h' and type(val) is str:
                head = val
                rel_h = rel[:-2]
            if rel[-2:] == '_t' and type(val) is str:
                tail = val
                rel_t = rel[:-2]

        assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
        assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

        # Drop this bs
        datum.pop(rel_h + '_h')
        datum.pop(rel_t + '_t')
        datum.pop('N')
        conv_datum += [head, rel_h, tail]

        # Get all qualifiers
        for k, v in datum.items():
            for _v in v:
                conv_datum += [k, _v]

        conv_data.append(tuple(conv_datum))
    return conv_data


def _conv_to_our_quint_format_(data):
    conv_data = []
    for datum in tqdm(data):
        conv_datum = []

        # Get head and tail rels
        head, tail, rel_h, rel_t = None, None, None, None
        for rel, val in datum.items():
            if rel[-2:] == '_h' and type(val) is str:
                head = val
                rel_h = rel[:-2]
            if rel[-2:] == '_t' and type(val) is str:
                tail = val
                rel_t = rel[:-2]

        assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
        assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

        # Drop this bs
        datum.pop(rel_h + '_h')
        datum.pop(rel_t + '_t')
        datum.pop('N')
        conv_datum += [head, rel_h, tail, None, None]

        if len(datum.items()) == 0:
            conv_data.append(tuple(conv_datum))
        else:
            # Get all qualifiers
            for k, v in datum.items():
                conv_datum[3] = k
                for _v in v:
                    conv_datum[4] = _v
                    conv_data.append(tuple(conv_datum))

    return conv_data


def _get_uniques_(train_data: List[tuple], valid_data: List[tuple], test_data: List[tuple]) -> (
list, list):
    """ Throw in parsed_data/wd15k/ files and we'll count the entities and predicates"""

    statement_entities, statement_predicates = [], []

    for statement in train_data + valid_data + test_data:
        statement_entities += statement[::2]
        statement_predicates += statement[1::2]

    statement_entities = sorted(list(set(statement_entities)))
    statement_predicates = sorted(list(set(statement_predicates)))

    return statement_entities, statement_predicates


def _pad_statements_(data: List[list], maxlen: int) -> List[list]:
    """ Padding index is always 0 as in the embedding layers of models. Cool? Cool. """
    result = [
        statement + [0] * (maxlen - len(statement)) if len(statement) < maxlen else statement[
                                                                                    :maxlen] for
        statement in data]
    return result


def load_wd15k_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_33_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_33'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_33_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_33'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_33_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_33'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_66_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_66'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_66_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_66'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_66_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_66'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_statements(maxlen: int) -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_quints() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]],
              entoid[q[4]]] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]],
              entoid[q[4]]] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]],
             entoid[q[4]]] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_triples() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_33_statements(maxlen: int) -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_33'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_33_quints() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_33'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_33_triples() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_33'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_66_statements(maxlen: int) -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_66'
    with open(WD15K_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(WD15K_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(WD15K_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_66_quints() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_66'
    with open(WD15K_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd15k_qonly_66_triples() -> Dict:
    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k_qonly_66'

    with open(WD15K_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WD15K_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WD15K_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_quints():
    # Load data from disk
    DIRNAME = Path('./data/raw_data/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'n-ary_train.json', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(json.loads(line))

    with open(DIRNAME / 'n-ary_test.json', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(json.loads(line))

    with open(DIRNAME / 'n-ary_valid.json', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(json.loads(line))

    # raw_trn[:-10], raw_tst[:10], raw_val[:10]
    # Conv data to our format
    conv_trn, conv_tst, conv_val = _conv_to_our_quint_format_(raw_trn), \
                                   _conv_to_our_quint_format_(raw_tst), \
                                   _conv_to_our_quint_format_(raw_val)

    # quints_entities, quints_predicates = _get_uniques_(train_data=conv_trn,
    #                                                          test_data=conv_tst,
    #                                                          valid_data=conv_val)

    # st_entities = ['__na__'] + quints_entities
    # st_predicates = ['__na__'] + quints_predicates
    # quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    # quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))
    quints_entities, quints_predicates = [], []
    for quint in conv_trn + conv_val + conv_tst:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_trn]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_val]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_tst]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_triples():
    # Load data from disk
    WP_DIR = PARSED_DATA_DIR / 'wikipeople'

    with open(WP_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WP_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WP_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_statements(maxlen=17) -> Dict:
    """
        :return: train/valid/test splits for the wikipeople dataset in its quints form
    """
    DIRNAME = Path('./data/raw_data/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'n-ary_train.json', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(json.loads(line))

    with open(DIRNAME / 'n-ary_test.json', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(json.loads(line))

    with open(DIRNAME / 'n-ary_valid.json', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(json.loads(line))

    # raw_trn[:-10], raw_tst[:10], raw_val[:10]
    # Conv data to our format
    conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn), \
                                   _conv_to_our_format_(raw_tst), \
                                   _conv_to_our_format_(raw_val)

    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=conv_trn,
                                                             test_data=conv_tst,
                                                             valid_data=conv_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in conv_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in conv_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in conv_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_fb15k237() -> Dict:
    """
        TODO: Shift all entities w 1. ZERO MUST BE PAD
        TODO: Return ID dicts?
    :return:
    """
    RAW_DATA_DIR = Path('./data/raw_data/fb15k237')

    training_triples = []
    valid_triples = []
    test_triples = []

    with open(RAW_DATA_DIR / "entity2id.txt", "r") as ent_file, \
            open(RAW_DATA_DIR / "relation2id.txt", "r") as rel_file, \
            open(RAW_DATA_DIR / "train2id.txt", "r") as train_file, \
            open(RAW_DATA_DIR / "valid2id.txt", "r") as valid_file, \
            open(RAW_DATA_DIR / "test2id.txt", "r") as test_file:
        num_entities = int(next(ent_file).strip("\n")) + 1  # One more for padding/unk
        num_relations = int(next(rel_file).strip("\n")) + 1  # One more for padding/unk

        for line in train_file:
            triple = line.strip("\n").split(" ")
            training_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

        num_valid = int(next(valid_file).strip("\n"))
        for line in valid_file:
            triple = line.strip("\n").split(" ")
            valid_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

        num_test = int(next(test_file).strip("\n"))
        for line in test_file:
            triple = line.strip("\n").split(" ")
            test_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

    return {"train": training_triples, "valid": valid_triples, "test": test_triples,
            "n_entities": num_entities, "n_relations": num_relations}


def load_fb15k() -> Dict:
    """
            TODO: Shift all entities w 1. ZERO MUST BE PAD
            TODO: Return ID dicts?
        :return:
    """
    RAW_DATA_DIR = Path('./data/raw_data/fb15k')

    training_triples = []
    valid_triples = []
    test_triples = []

    with open(RAW_DATA_DIR / "entity2id.txt", "r") as ent_file, \
            open(RAW_DATA_DIR / "relation2id.txt", "r") as rel_file, \
            open(RAW_DATA_DIR / "train2id.txt", "r") as train_file, \
            open(RAW_DATA_DIR / "valid2id.txt", "r") as valid_file, \
            open(RAW_DATA_DIR / "test2id.txt", "r") as test_file:
        num_entities = int(next(ent_file).strip("\n")) + 1
        num_relations = int(next(rel_file).strip("\n")) + 1
        num_trains = int(next(train_file).strip("\n")) + 1
        for line in train_file:
            triple = line.strip("\n").split(" ")
            training_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

        num_valid = int(next(valid_file).strip("\n"))
        for line in valid_file:
            triple = line.strip("\n").split(" ")
            valid_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

        num_test = int(next(test_file).strip("\n"))
        for line in test_file:
            triple = line.strip("\n").split(" ")
            test_triples.append([int(triple[0]) + 1, int(triple[2]) + 1, int(triple[1]) + 1])

    return {"train": training_triples, "valid": valid_triples, "test": test_triples,
            "n_entities": num_entities, "n_relations": num_relations}


def load_dummy_dataset():
    """

    :return: a dummy dataset for a model to overfit
    """
    num_rows = 1000
    num_entities = 200
    num_relations = 20
    ds = [[]]


class DataManager(object):
    """ Give me your args I'll give you a path to load the dataset with my superawesome AI """

    @staticmethod
    def load(config: Union[dict, FancyDict]) -> Callable:
        """ Depends upon 'STATEMENT_LEN' and 'DATASET' """

        # Get the necessary dataset's things.
        assert config['DATASET'] in KNOWN_DATASETS, f"Dataset {config['DATASET']} is unknown."

        if config['DATASET'] == 'wd15k':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_triples
            else:
                return partial(load_wd15k_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wikipeople':
            if config['STATEMENT_LEN'] == 5:
                return load_wikipeople_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wikipeople_triples
            else:
                print("Here mf")
                return partial(load_wikipeople_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd15k_qonly':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_qonly_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_qonly_triples
            else:
                return partial(load_wd15k_qonly_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd15k_qonly_33':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_qonly_33_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_qonly_33_triples
            else:
                return partial(load_wd15k_qonly_33_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd15k_qonly_66':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_qonly_66_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_qonly_66_triples
            else:
                return partial(load_wd15k_qonly_66_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd15k_33':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_33_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_33_triples
            else:
                return partial(load_wd15k_33_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd15k_66':
            if config['STATEMENT_LEN'] == 5:
                return load_wd15k_66_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd15k_66_triples
            else:
                return partial(load_wd15k_66_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'fb15k':
            return load_fb15k
        elif config['DATASET'] == 'fb15k237':
            return load_fb15k237

    @staticmethod
    def gather_missing_entities(data: List[list], n_ents: int, positions: List[int]) -> np.array:
        """

            Find the entities which aren't available from range(n_ents).
            Think inverse of gather_entities

        :param data: A list of triples/quints whatever
        :param n_ents: Int signifying total number of entities
        :param positions: the positions over which we intend to count these things.
        :return: np array of entities NOT appearing in range(n_ents)
        """

        appeared = np.zeros(n_ents, dtype=np.int)
        for datum in data:
            for pos in positions:
                appeared[datum[pos]] = 1

        # Return this removed from range(n_ents)
        return np.arange(n_ents)[appeared == 0]

    @staticmethod
    def get_graph_repr(train: List[List[int]], valid: List[List[int]], test: List[List[int]],
                   ne: int, nr: int, config):
        """
            Decisions:
                We are NOT making inverse of qualifier relations. Those are just repeated.
                The normal triple relations are inverted.

            Pseudocode:
                for each of train, test, valid split
                    for each triple,
                        s, o -> edge_index
                        r -> edge_type
                        r_q1,... -> list of column vectors (np.arrs)
                        e_q1,... -> list of column vectors (np.arrs)
                    endfor
                endfor

                    create reverse relations in the existing stuff.

            TODO: Check if the data has repeats (should not).  x

            :param train: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]] (already have a max length
            :param valid: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]] (already have a max length
            :param test: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]] (already have a max length
            :param ne: number of entities in the KG
            :param nr: number of relations in the KG
            :param config: the config dict
        """
        has_qualifiers: bool = True

        def _get_graph_repr_(raw) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """ Actual thing happens here """
            edge_index, edge_type = np.zeros((2, len(raw) * 2)), np.zeros((len(raw) * 2))
            qual_rel = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2))
            qual_ent = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2))

            # Add actual data
            for i, data in enumerate(raw):
                edge_index[:, i] = [data[0], data[2]]
                edge_type[i] = data[1]

                # @TODO: add qualifiers
                if has_qualifiers:
                    qual_rel[:, i] = data[3::2]
                    qual_ent[:, i] = data[4::2]

            # Add inverses
            edge_index[1, len(raw):] = edge_index[0, :len(raw)]
            edge_index[0, len(raw):] = edge_index[1, :len(raw)]
            edge_type[len(raw):] = edge_type[:len(raw)] + nr

            if has_qualifiers:
                qual_rel[:, len(raw):] = qual_rel[:, :len(raw)]
                qual_ent[:, len(raw):] = qual_ent[:, :len(raw)]

            return edge_index, edge_type, qual_rel, qual_ent

        train_edge_index, train_edge_type, train_qual_rel, train_qual_ent = _get_graph_repr_(train)
        valid_edge_index, valid_edge_type, valid_qual_rel, valid_qual_ent = _get_graph_repr_(valid)
        test_edge_index, test_edge_type, test_qual_rel, test_qual_ent = _get_graph_repr_(test)

        return train_edge_index, train_edge_type, train_qual_rel, train_qual_ent, \
               valid_edge_index, valid_edge_type, valid_qual_rel, valid_qual_ent, \
               test_edge_index, test_edge_type, test_qual_rel, test_qual_ent


if __name__ == "__main__":
    # ds = load_fb15k237()
    # ds1 = load_wd15k_quints()
    # ds2 = load_wd15k_triples()
    # ds3 = load_wd15k_qonly_quints()
    # ds4 = load_wd15k_qonly_triples()
    # print(len(ds4))

    ds = load_wd15k_66_statements(maxlen=43)
    tr = ds['train']
    vl = ds['valid']
    ts = ds['test']
    ne = ds['n_entities']
    nr = ds['n_relations']
    print("Magic Mike!")
