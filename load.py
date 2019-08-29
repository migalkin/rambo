"""
    File which enables easily loading any dataset we need
"""

from utils import *


def load_wd15k() -> (List, List, Dict[str, int], Dict[str, int], _):
    """

    :return:
    """

    # Load data from disk
    with open(PARSED_DATA_DIR / 'parsed_raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # with open('./data/parsed_data/parsed_raw_data.pkl', 'rb') as f:
    #     raw_data = pickle.load(f)

    entities, predicates = [], []

    for quint in raw_data:
        entities += [quint[0], quint[2]]
        if quint[4]:
            entities.append(quint[4])

        predicates.append(quint[1])
        if quint[3]:
            predicates.append(quint[3])

    entities = sorted(list(set(entities)))
    predicates = sorted(list(set(predicates)))

    entities = ['__na__', '__pad__'] + entities
    predicates = ['__na__', '__pad__'] + predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(entities)}
    prtoid = {pred: i for i, pred in enumerate(predicates)}




