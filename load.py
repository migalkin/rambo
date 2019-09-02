"""
    File which enables easily loading any dataset we need
"""
import operator
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

    return {"train": train, "valid": valid, "test": test, "num_entities": len(q_entities), "num_relations": len(q_predicates)}


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

    return {"train": train, "valid": valid, "test": test, "num_entities": len(triples_entities), "num_relations": len(triples_predicates)}

def load_fb15k237() -> Dict:
    RAW_DATA_DIR = Path('./data/raw_data/fb15k237')

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

    return {"train": training_triples, "valid": valid_triples, "test": test_triples, "num_entities": num_entities, "num_relations": num_relations}


def load_fb15k() -> Dict:
    RAW_DATA_DIR = Path('./data/raw_data/fb15k')

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

    return {"train": training_triples, "valid": valid_triples, "test": test_triples, "num_entities": num_entities, "num_relations": num_relations}


class DataManager(object):
    """ Give me your args I'll give you a path to load the dataset with my superawesome AI """

    @staticmethod
    def load(config: Union[dict, FancyDict]) -> Callable:
        """ Depends upon 'IS_QUINTS' and 'DATASET' """

        # Get the necessary dataset's things.
        assert config['DATASET'] in KNOWN_DATASETS, f"Dataset {config['DATASET']} is unknown."

        if config['DATASET'] == 'wd15k':
            if config['IS_QUINTS']:
                return load_wd15k_quints
            else:
                return load_wd15k_triples
        elif config['DATASET'] == 'fb15k':
            return load_fb15k
        elif config['DATASET'] == 'fb15k237':
            return load_fb15k237

    @staticmethod
    def gather_entities(data: List[list], n_ents: int, positions: List[int]) -> np.array:
        """
            Count the number of entities at particular positions
                As a bonus, it also excludes entity
                    which never see the light of the day
                    in our dataset.
        """
        appeared = np.zeros(n_ents, dtype=np.int)
        for datum in data:
            for pos in positions:
                appeared[datum[pos]] = 1

        # Return all the entities which are one.
        return np.arange(n_ents)[appeared.astype(np.bool)]


if __name__ == "__main__":
    ds = load_fb15k237()
    ds1 = load_wd15k_quints()
    ds2 = load_wd15k_triples()