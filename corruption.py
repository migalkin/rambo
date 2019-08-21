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

    def __init__(self, n, pos: list = None, gold_data: np.array = None, debug: bool = False):
        self.n = n
        self.pos, self.debug = pos, debug
        self.filtering = gold_data is not None
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

    def _get_entities_(self, n: int, excluding: Union[int, np.array] = None, keys: np.array = None, data_hash: dict = None) -> np.array:
        """
            Step 1: Create random entities (n times)
            Step 2: If not filtering and excluding, a while loop to ensure all are replaced
            Step 3: If filtering, then we verify if the ent has not appeared in the dataset

        :param n: number of things to inflect
        :param excluding:
            - int - don't have this entity
            - np.array - don't have these entities AT THESE POSITIONs
        :param keys: complete data used for filtering
        :param data_hash: the data hash we're using to do the filtering
        :return: (n,) entities
        """

        # Step 1
        entities = np.random.permutation(np.arange(self.n))[:n]

        # Step 2
        if excluding and not self.filtering:

            # If excluding is single
            if type(excluding) in [int, float]:
                excluding = np.repeat(excluding, n)

            if self.debug:
                repeats = 0

            while True:
                eq = entities == excluding
                if not eq.any():
                    # If they're completely dissimilar
                    break
                new_entities = np.random.choice(np.arange(n), int(np.sum(eq)))
                entities[eq] = new_entities

                if self.debug:
                    repeats += 1

        if self.debug:
            print(f"Corruption: The excluding loop went for {repeats} times.")

        # Step 3
        if self.filtering:
            raise NotImplementedError

        return entities

    def corrupt(self, data: np.array , pos=None) -> np.array:
        """
            For corrupting one true data point, every possible manner
        :param data: np.array of that which needs all forms of corruption
        :param pos: optional param which specifies the positions to corrupt
        :return: np.array of (n, _) where n is num of corrupted things
        """
        pos = self.pos if pos is None else pos
        write_index = 0

        # Get a n_ent * len(pos) array
        corrupted = np.zeros((len(pos) * (self.n-1), len(data)))

        # @TODO: complete this. too sleepy

        return None


    def corrupt_batch(self, data: np.array, pos=None):
        """
            For each positions in data, make inflections. n_infl = len(data) // len(pos)
            Returns (pos, neg) pairs
        """
        pos = self.pos if pos is None else pos

        split_data = np.array_split(data, len(pos))
        neg_data = np.zeros_like(data)

        write_index = 0
        for i, (_pos, _data) in enumerate(zip(pos, split_data)):

            entities = self._get_entities_(_data.shape[0], excluding=_data[_pos, :])
            neg_data[write_index: write_index+_data.shape[0], _pos] = entities
            write_index += _data.shape[0]


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