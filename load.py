"""
    File which enables easily loading any dataset we need
"""

from utils import *


def load_wd15k_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    WD15K_DIR = PARSED_DATA_DIR / 'wd15k'
    with open(WD15K_DIR/ 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(WD15K_DIR/ 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(WD15K_DIR/ 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints+valid_quints+test_quints:
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

    return {"train": train, "valid": valid, "test": test}


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


    triples_entities = sorted(list(set(triples_entities)))
    triples_predicates = sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test}


if __name__ == "__main__":
    quints = load_wd15k_quints()
    triples = load_wd15k_triples()