import pickle
from typing import List
import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

from utils import *


class Corruption:
    """

        Used to efficiently corrupt given data.

        # Usage
        |-> If no filtering is needed (neg may exist in dataset)
            |-> simply init the class with n_ents, and call corrupt with pos data, and num corruptions
        |-> If filtering is needed (neg must not belong in dataset)
            |-> also pass all true data while init

        # Usage Example
        **no filtering**
        gold_data = np.random.randint(0, 1000, (50, 3))
        corrupter = Corruption(n=1000, pos=[0, 2])
        corrupter.corrupt(gold_data, pos=[0, 2])  # pos overrides
        corrupter.precomute(true, neg_per_samples=1000) (although not much point of this)

        **with filtering**
        gold_data =  np.random.randint(0, 1000, (50, 3))
        corrupter = Corruption(n=1000, data=gold_data, pos=[0, 2])
        corrupter.corrupt(gold_data, pos=[0, 2]) # pos overrides

        # Features
        - Precompute (not sure if we'll do this)
        - Filtering
        - Can return in pairwise fashion
            (p,n) pairs with repeated p's if needed
    """

    def __init__(self, n, pos=None, gold_data=None):
        self.n = n
        self.pos = pos
        self.filtered = gold_data is not None
        self.hashes = self._index_(gold_data)

    def _index_(self, data):
        """ Create hashes of trues"""
        if data is None: return None

        hashes = [{} for _ in self.pos]
        for datum in data:
            for _pos, _hash in zip(self.pos, hashes):
                _remainder = datum.copy()
                _real_val = _remainder.pop(_pos)

                _hash.setdefault(_remainder, []).append(_real_val)

        return hashes

    def corrupt(self, data, pos=None):
        ...

    def corrupt_batch(self, data: np.array, pos=None):
        """
            For each positions in data, make inflections. n_infl = len(data) // len(pos)
            Returns (pos, neg) pairs
        """

        pos = self.pos if pos is None else pos
        split_data = np.array_split(data, len(pos))
        neg_data = np.zeros_like(data)

        write_index = 0
        for i, _data in enumerate(split_data):
            _pos = pos[i]
            # Split this data @ this pos

            if not self.filtered:
                """
                    If we want to skip having the entity in "data" to also be here we can do something like
                    np.hstack(
                (np.arange(0, pos_data[corruption_pos]), np.arange(pos_data[corruption_pos] + 1, self.n_entities)))
                """
                ents = np.arange(self.n)
                np.random.shuffle(ents)
                ents = ents[_data.shape[0]]

                neg_data[write_index: write_index+_data.shape[0], _pos] = ents

                write_index += _data.shape[0]

            else:
                ...





if __name__ == "__main__":
    # Just checking things
    probs = [0.3, 0.0, 0.3, 0.4]
    q_neg = sample_negatives(raw_data[0], probs)
    print(q_neg)

    l = np.random.choice(["s", "p", "o", "q"], 1000, p=probs)
    print(l[0])
    unique, counts = np.unique(l, return_counts=True)
    dict(zip(unique, counts))
    # l.count("s"), l.count("p"), l.count("o"), l.count("q")

    negative_samples = []
    for q in tqdm(raw_data):
        negative_samples.append(sample_negatives(q, probs))

    count = 0
    for n in tqdm(negative_samples):
        if n in raw_data:
            print(n)
            count += 1

    print(f"{count} / {len(raw_data)} are not unique negatives")