""" Some things needed across the board"""
import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Union
from collections import namedtuple

from mytorch.utils.goodies import Timer

Quint = namedtuple('Quint', 's p o qp qe')
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')
PRETRAINING_DATA_DIR = Path('./data/pre_training_data')


class UnknownSliceLength(Exception): pass


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


class SingleSampler:
    """
        Another sampler which gives correct + all corrupted things for one triple
        [NOTE]: Depriciated
    """

    def __init__(self, data: dict, bs: int, n_items: int = 5):
        """
            Returns a pair of `bs` [pos, neg, neg, neg, neg ... ] per iteration.

        :param data: a dict with {'pos': __, 'neg': __} format
        :param bs: int of batch size (returned on one iter)
        :param n_items: 5 if we're sampling quints, 3 if triples.
        """
        self.data = {'pos': np.array(data['pos'], dtype=np.int), 'neg': np.array(data['neg'], dtype=np.int)}
        self.bs = bs
        self.n_items = n_items

        assert len(self.data['pos']) == len(self.data['neg']), "Mismatched lengths between pos and neg data!"
        self.shuffle()

    def shuffle(self):
        shuffle_ids = np.arange(len(self.data['pos']))
        np.random.shuffle(shuffle_ids)

        self.data = {'pos': self.data['pos'][shuffle_ids], 'neg': self.data['neg'][shuffle_ids]}

    def __len__(self):
        return len(self.data['pos']) // self.bs

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """ Concat pos quint with n neg quints (corresponding)"""
        if self.i >= self.data['pos'].__len__():
            print("Should stop")
            raise StopIteration

        res = np.zeros((self.bs, self.data['neg'][self.i].__len__() + 1, self.n_items), dtype=np.int)
        pos = self.data['pos'][self.i: min(self.i + self.bs, len(self.data['pos']))]
        neg = self.data['neg'][self.i: min(self.i + self.bs, len(self.data['neg']))]
        for i, (_p, _n) in enumerate(zip(pos, neg)):
            res[i, 0] = _p
            res[i, 1:] = _n

        self.i += self.bs
        return res


class EvaluationBench:
    """
        Sampler which generates every possible negative corruption of a positive triple.
        So given 15k entities and triples (not quints), we get 30k-2 negative triples.

        # Corruption Scheme
        1. Case: Triples
            -> given <s,p,o>
            -> make <s`, p, o> where s` \in E, and s` != s
                -> if <s`, p, o> \in dataset: remove if "filtered" flag
            -> make <s, p, o`> where o` \in E and o` != o
                -> if <s, p, o`> \in dataset: remove if "filtered" flag
        2. Case: Quints
            -> given <s, p, o, qp, qe>
            -> make <s`, p, o, qp, qe> where s` \in E, and s` != s
                -> if <s`, p, o, qp, qe> \in dataset: remove if "filtered" flag
            -> make <s, p, o`, qp, qe> where o` \in E and o` != o
                -> if <s, p, o`, qp, qe> \in dataset: remove if "filtered" flag
            -> make <s, p, o, qp, qe`> where qe` \in E, and qe` != qe
                -> if <s, p, o, qp, qe`> \in dataset: remove if "filtered" flag

        Once done, it uses the model given to it, feeding it data and storing the np outputs.
        Returns crisp, concise metrics.

        # Data Storing Scheme
        1. Keep hashes of data along with actual data (which is 1D array of 3 or 5 items)
        2. Generate all negatives if not already found in disk. If found, load. (DO NOT RUN THIS ON LOW RAM PCs)
    """

    def __init__(self, data: Union[List[int], np.array], model: nn.Module,
                 bs: int, filtered: bool = False, quints: bool = True):
        """

        :param data: list/iter of positive triples. Np array are appreciated
        :param bs: anything under 256 is shooting yourself in the foot.
        :param filtered: if you want corrupted triples checked.
        :param quints: if the data has 5 elements
        """
        self.bs, self.filtered, self.quints = bs, filtered, quints
        self.model = model

        self.data, self.hashes = self.store_pos(data)
        self.n_entities = self.model.num_entities


    def store_pos(self, pos_data: Union[List[int], np.array]) \
            -> (np.array, Union[None, List[dict]]):
        """
            if we're filtering, we need to come back to this data very often.
            In that case, we compute <ent> based hashes for quick lookup of corrupted element.

            @TODO: test!!

        :param pos_data: iter
        :return: (pos_data, and hashes if needed)
        """

        if not self.filtered:
            return pos_data, None

        # We ARE doing filtering if here.
        if self.quints:
            hashes = {}, {}, {}
            """
            Dear reader,
            
            Make of this what you will. 
            Peace!
            [ 
                : ---- :D
                : ---- :D
                : ---- :D  
            ]
            """
            for quint in pos_data:
                s, p, o, qp, qe = quint
                hashes[0].setdefault((p,o, qp, qe), []).append(s)
                hashes[1].setdefault((s, p, qp, qe), []).append(o)
                hashes[2].setdefault((s, p, o, qp), []).append(qe)

        else:
            hashes = {}, {}
            for triple in pos_data:
                s, p, o, qp, qe = triple
                hashes[0].setdefault((p,o), []).append(s)
                hashes[1].setdefault([s, p], []).append(o)

        return pos_data, hashes

    def _get_corruptable_entities_(self, corruption_pos:int, pos_data: np.array) -> Union[list, np.array]:
        """
        Get all possible corruptable entities given this triple/quint and for this position {1,3,5}.
        Eg. if self.filtered:
                corruption_position = 0
                ps = [11,22,33,44,55]
                also exists in dataset = [99,22,33,44,55]
                then - n_ent - 2 diff things
            else:
                all but ps[corruption_position]

            :param corruption_pos: int signifying whether we're fucking up s, o, or qe
            :param pos_data: true triple/quint
            :return list of ints/np.array of ints
        """
        assert corruption_pos in [0, 2, 4], "Excuse me waht the fukk. Invalid position for corruption!"

        if not self.filtered:
            return np.hstack((np.arange(0, pos_data[corruption_pos]), np.arange(pos_data[corruption_pos] + 1, self.n_entities)))
        else:
            _hash_pos = int(corruption_pos / 2)
            hashes = self.hashes[_hash_pos]

            key = tuple(np.hstack((pos_data[:corruption_pos], pos_data[corruption_pos + 1:])))
            existing = hashes[key]
            raw_corruptables = np.arange(self.n_entities)
            return np.delete(raw_corruptables, existing)

    def get_negatives(self, ps: np.array) -> np.array:
        """
            Generate all needed negatives for given triple/quint

        :return: None
        """
        n = 3 if self.quints else 2
        # neg_datum = np.zeros((self.n_entities*n - n, len(ps)))

        # Corrupting s
        wrong_s = self._get_corruptable_entities_(corruption_pos=0, pos_data=ps)
        neg_datum_s = np.zeros((wrong_s.shape[0], len(ps)))
        neg_datum_s[:, 0] =  wrong_s
        neg_datum_s[:, 1:] = ps[1:]

        # Corrupting o
        wrong_o = self._get_corruptable_entities_(corruption_pos=2, pos_data=ps)
        neg_datum_o = np.zeros((wrong_o.shape[0], len(ps)))
        neg_datum_o[:, 2] = wrong_o
        neg_datum_o[:, :2] = ps[:2]

        # If we have quints, also copy over the qualifiers
        if self.quints:
            neg_datum_o[:, 3:] = ps[3:]

            # Corrupting qe
            wrong_qe = self._get_corruptable_entities_(corruption_pos=4, pos_data=ps)
            neg_datum_qe = np.zeros((wrong_qe.shape[0], len(ps)))
            neg_datum_qe[:, -1] = wrong_qe
            neg_datum_qe[:, :-1] = ps[:-1]

            neg_datum = np.vstack((neg_datum_s, neg_datum_o, neg_datum_qe))
        else:
            neg_datum = np.vstack((neg_datum_s, neg_datum_o))

        return neg_datum


if __name__ == "__main__":
    class DummyModel(nn.Module):
        num_entities = 15000

    pos_data = np.random.randint(0, 15000, (400000, 5))
    bs = 2
    filtered = True
    quint = True


    eb = EvaluationBench(data=pos_data, model=DummyModel(), bs=bs, filtered=filtered, quints=quint)
    ps = pos_data[0]
    print(ps)
    with Timer() as timer:
        ng = eb.get_negatives(ps)
    print(ng.shape, timer.interval, timer.interval*pos_data.shape[0])
    print(ng)



