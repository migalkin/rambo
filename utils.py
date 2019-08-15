""" Some things needed across the board"""
from collections import namedtuple
from pathlib import Path
import pickle

from tqdm import tqdm
import numpy as np

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

entities = sorted(list(set(entities)))
predicates = sorted(list(set(predicates)))

entities = ['__na__', '__pad__'] + entities
predicates = ['__na__', '__pad__'] + predicates

# uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
entoid = {pred: i for i, pred in enumerate(entities)}
prtoid = {pred: i for i, pred in enumerate(predicates)}


# Make data iterator -> Modify Simple Iterator from mytorch
class QuintRankingSampler:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__
    :return:
    """

    def __init__(self, data, bs: int = 64):

        self.bs = bs  # Batch Size (of neg items)
        self.pos = data["pos"]
        self.neg = data["neg"]

        self.initialize()

        self.n = len(self.pos)

        assert len(self.pos) == len(self.neg), "Mismatched lengths of pos and negs."

        # Shuffle everything
        self.shuffle()

    def initialize(self):
        """
            Right now, pos is a list of quints, neg is a list of list of quints.
            We want to flatten the negs; and repeat the pos
        """
        # Flatten negs
        flat_negs = []
        for negs in self.neg:
            flat_negs += negs

        self.neg_times = self.neg[0].__len__()

        # Repate pos
        repeat_pos = []
        for pos in self.pos:
            repeat_pos += [pos for _ in range(self.neg_times)]

        self.pos = np.array(repeat_pos)
        self.neg = np.array(flat_negs)

    def shuffle(self):
        """ Shuffle pos and neg together """

        shuffle_ids = np.arange(self.n)
        np.random.shuffle(shuffle_ids)
        #         print(shuffle_ids)
        self.pos = self.pos[shuffle_ids]
        self.neg = self.neg[shuffle_ids]

    def __len__(self):
        return self.n // self.bs - (1 if self.n * self.neg_times % self.bs else 0)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """
            Each time, take `bs` pos (and take one neg from each `bs` pos)
        """
        if self.i >= self.n:
            print("Should stop")
            raise StopIteration

        _pos = self.pos[self.i: min(self.i + self.bs, len(self.pos) - 1)]
        _neg = self.neg[self.i: min(self.i + self.bs, len(self.pos) - 1)]

        self.i = min(self.i + self.bs, self.n)

        return _pos, _neg

if __name__ == "__main__":
    pass
    # Test it
    # sampler = QuintRankingSampler({"pos": raw_data, "neg": ... }, bs=4000)
    # for x in tqdm(sampler):
    #     pass

