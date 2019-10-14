from utils import *
from corruption import Corruption

class SimpleSampler:
    """
        Simply iterate over X
    """

    def __init__(self, data: Union[np.array, list], bs: int = 64):
        self.bs = bs
        self.data = np.array(data)  # pos data only motherfucker

        self.shuffle()

    def shuffle(self):
        npr.shuffle(self.data)

    def __len__(self):
        return self.data.shape[0] // self.bs

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """
            Each time, take `bs` pos
        """
        if self.i >= self.data.shape[0]:
            print("Should stop")
            raise StopIteration

        _pos = self.data[self.i: min(self.i + self.bs, len(self.data) - 1)]
        self.i = min(self.i + self.bs, self.data.shape[0])
        return _pos


# Make data iterator -> Modify Simple Iterator from mytorch
class QuintRankingSampler:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__

         [NOTE]: Depriciated
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

        # Repeat pos
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


class NeighbourhoodSampler(SimpleSampler):
    def __init__(self, data: Union[np.array, list], corruptor: Corruption, bs: int = 64, hashes=None):
        super().__init__(data, bs)
        self.corruptor = corruptor
        self.hop1, self.hop2 = hashes if hashes is not None else [{}, {}]

    def __next__(self):
        """
            Each time, take `bs` pos, get neg, get hops for both pos and neg...
        """

        _pos = super().__next__()
        _neg = self.corruptor.corrupt_batch(_pos)

        _pos_objs = _pos[:, 2]  # all Pos (bs)
        _neg_objs = _neg[:, 2]  # all Neg (bs)

        _pos_hop1, _pos_hop2 = self._get_neighborhoods_(_pos_objs)
        _neg_hop1, _neg_hop2 = self._get_neighborhoods_(_neg_objs)

        return _pos, _pos_hop1, _pos_hop2, _neg, _neg_hop1, _neg_hop2

    def _get_neighborhoods_(self, objs: list) -> (np.ndarray, np.ndarray):
        """
            Pull hop1, hop2 from self.hop*, and then pad and return

        :param objs: list of objects
        :return: (nparr hop1, nparr hop2)
        """

        # First and second neighbourhood
        hop1, hop2 = [], []
        for e in objs:
            hop1.append(self.hop1.get(e, [0, 0]))       # Hop1 is list of list of tuples
            hop2.append(self.hop2.get(e, [0, 0, 0]))       # Hop2 is list of list of tuples

        if [] in hop1 or [] in hop2:
            print('shit')

        # Pad Stuff
        _pad = (0, 0)
        max1, max2 = max(len(neighbors) for neighbors in hop1), max(len(neighbors) for neighbors in hop2)
        _hop1, _hop2 = np.zeros((self.bs, max1, 2)), np.zeros((self.bs, max2, 3))
        for i, datum in enumerate(hop1):
            _hop1[i, :len(datum)] = datum
        for i, datum in enumerate(hop2):
            _hop2[i, :len(datum)] = datum

        return _hop1, _hop2


class SingleSampler:
    """
        Another sampler which gives correct + all corrupted things for one triple
        [NOTE]: Depreciated
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