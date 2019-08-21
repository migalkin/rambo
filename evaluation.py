from functools import partial
from tqdm import tqdm
import types

# Local
from utils import *

class EvaluationBench:
    """
        Sampler which generates every possible negative corruption of a positive triple.
        So given 15k entities and triples (not quints), we get 30k-2 negative triples.

        # Corruption Scheme
        1. Case: Triples
            -> given <s,p,o>
            -> make <s`, p, o> where s` in E, and s` != s
                -> if <s`, p, o> in dataset: remove if "filtered" flag
            -> make <s, p, o`> where o` in E and o` != o
                -> if <s, p, o`> in dataset: remove if "filtered" flag
        2. Case: Quints
            -> given <s, p, o, qp, qe>
            -> make <s`, p, o, qp, qe> where s` in E, and s` != s
                -> if <s`, p, o, qp, qe> in dataset: remove if "filtered" flag
            -> make <s, p, o`, qp, qe> where o` in E and o` != o
                -> if <s, p, o`, qp, qe> in dataset: remove if "filtered" flag
            -> make <s, p, o, qp, qe`> where qe` in E, and qe` != qe
                -> if <s, p, o, qp, qe`> in dataset: remove if "filtered" flag

        Once done, it uses the model given to it, feeding it data and storing the np outputs.
        Returns crisp, concise metrics.

        # Data Storing Scheme
        1. Keep hashes of data along with actual data (which is 1D array of 3 or 5 items)
        2. Generate all negatives if not already found in disk. If found, load. (DO NOT RUN THIS ON LOW RAM PCs)
    """

    def __init__(
            self,
            data: Dict[str, Union[List[int], np.array]],
            model: nn.Module,
            bs: int,
            metrics: list,
            _filtered: bool = False,
            _quints: bool = True):
        """

        :param data: {'train': list/iter of positive triples, 'valid': list/iter of positive triples}.
            Np array are appreciated
        :param bs: anything under 256 is shooting yourself in the foot.
        :param _filtered: if you want corrupted triples checked.
        :param _quints: if the data has 5 elements
        """
        self.bs, self.filtered, self.quints = bs, _filtered, _quints
        self.model = model

        self.data_valid = data['valid']
        self.hashes = self._index_pos_(np.append(data['train'], data['valid']))
        self.n_entities = self.model.num_entities
        self.metrics = metrics

    def reset(self):
        """ Call when you wanna run again but not change hashes etc """
        raise NotImplementedError

    def _index_pos_(self, pos_data: Union[List[int], np.array]) \
            -> (np.array, Union[None, List[dict]]):
        """
            if we're filtering, we need to come back to this data very often.
            In that case, we compute <ent> based hashes for quick lookup of corrupted element.

            @TODO: test!!

        :param pos_data: iter
        :return: (pos_data, and hashes if needed)
        """

        if not self.filtered:
            return None

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
                hashes[0].setdefault((p, o, qp, qe), []).append(s)
                hashes[1].setdefault((s, p, qp, qe), []).append(o)
                hashes[2].setdefault((s, p, o, qp), []).append(qe)

        else:
            hashes = {}, {}
            for triple in pos_data:
                s, p, o, qp, qe = triple
                hashes[0].setdefault((p, o), []).append(s)
                hashes[1].setdefault([s, p], []).append(o)

        return hashes

    def _get_corruptable_entities_(self, corruption_pos: int, pos_data: np.array) -> Union[list, np.array]:
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
        assert corruption_pos in [0, 2, 4], "Invalid position for corruption!"

        if not self.filtered:
            return np.hstack(
                (np.arange(0, pos_data[corruption_pos]), np.arange(pos_data[corruption_pos] + 1, self.n_entities)))
        else:
            _hash_pos = int(corruption_pos / 2)
            hashes = self.hashes[_hash_pos]

            key = tuple(np.hstack((pos_data[:corruption_pos], pos_data[corruption_pos + 1:])))
            existing = hashes[key]
            raw_corruptables = np.arange(self.n_entities)
            return np.delete(raw_corruptables, existing)

    def _get_negatives_(self, ps: np.array) -> np.array:
        """
            Generate all needed negatives for given triple/quint

        :return: None
        """
        n = 3 if self.quints else 2
        # neg_datum = np.zeros((self.n_entities*n - n, len(ps)))

        # Corrupting s
        wrong_s = self._get_corruptable_entities_(corruption_pos=0, pos_data=ps)
        neg_datum_s = np.zeros((wrong_s.shape[0], len(ps)))
        neg_datum_s[:, 0] = wrong_s
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

    def _compute_metric_(self, scores: np.array) -> List[Union[float, np.float]]:
        """ See what metrics are to be computed, and compute them."""
        return [_metric(scores) for _metric in self.metrics ]

    def _summarize_metrics_(self, accumulated_metrics: np.array) -> np.array:
        """
            Aggregate metrics across time. Accepts np array of (len(self.data_valid), len(self.metrics))
        """
        mean = np.mean(accumulated_metrics, axis=0)
        summary = {}
        for i, _metric in enumerate(self.metrics):
            if _metric.__class__ is partial:
                summary[_metric.func.__name__ + ' ' + str(_metric.keywords['k'])] = mean[i]
            else:
                summary[_metric.__name__] = mean[i]

        return summary

    @staticmethod
    def summarize_run(summary: dict):
        """ Nicely print what just went down """
        print(f"This run over {summary['data_length']} {'quints' if summary['quints'] else 'triples'} took "
              f"%(time).3f min" % {'time': summary['time_taken']/60.0})
        print("---------\n")
        for k,v in summary['metrics'].items():
            print(k, ':', "%(v).4f" % {'v': v})

    def run(self, *args, **kwargs):
        """
            Calling this iterates through different data points, generates negatives, passes them to the model,
                collects the scores, computes the metrics, and reports them.
        """
        metrics = []

        with Timer() as timer:
            for pos in tqdm(self.data_valid):
                neg = self._get_negatives_(pos)

                if len(neg) + 1 < self.bs:  # Can do it in one iteration
                    with torch.no_grad():
                        x = torch.tensor(np.vstack((pos.transpose(), neg)), dtype=torch.long,
                                         device=self.model.config['DEVICE'])
                        scores = self.model.predict(x)

                else:
                    scores = torch.tensor([], dtype=torch.float, device=self.model.config['DEVICE'])
                    for i in range(neg.shape[0])[::self.bs]:  # Break it down into batches and then do dis
                        _neg = neg[i: i + self.bs]
                        if i == 0:
                            x = torch.tensor(np.vstack((pos.transpose(), _neg)), dtype=torch.long,
                                             device=self.model.config['DEVICE'])
                        else:
                            x = torch.tensor(_neg, dtype=torch.long, device=self.model.config['DEVICE'])
                        _scores = self.model.predict(x)
                        # print(f"Org scores: {scores.shape}, {scores.device} | New scores: {_scores.shape}, {_scores.device}")
                        scores = torch.cat((scores, _scores))

                _metrics = self._compute_metric_(scores)
                metrics.append(_metrics)

        # Spruce up the summary with more information
        time_taken = timer.interval
        metrics = self._summarize_metrics_(metrics)
        summary = {'metrics': metrics, 'time_taken': time_taken, 'data_length': len(self.data_valid),
                   'quints': self.quints, 'filtered': self.filtered}

        self.summarize_run(summary)

        return summary


def acc(scores: torch.Tensor) -> np.float:
    """ Accepts a (n, ) tensor """
    return (torch.argmin(scores, dim=0) == 0).float().detach().cpu().numpy()


def mrr(scores: torch.Tensor) -> np.float:
    """ Tested | Accepts one (n,) tensor """
    ranks = (torch.argsort(scores, dim=0) == 0).nonzero()[0]
    recirank = 1.0/(ranks+1).float()
    return recirank.detach().cpu().numpy()


def hits_at(scores: torch.Tensor, k: int=5) -> float:
    """ Tested | Accepts one (n,) tensor """
    rank = (torch.argsort(scores, dim=0) == 0).nonzero()[0] + 1
    # print(rank)
    if rank <= k:
        return 1.0
    else:
        return 0.0

def evaluate_pointwise(pos_scores: torch.Tensor, neg_scores:torch.Tensor)->torch.Tensor:
    """
        Given a pos and neg quint, how many times did the score for positive be more than score for negative
    
        :param pos_scores: scores corresponding to pos quints (bs, )
        :param neg_scores: scores corresponding to neg quints (bs, )
        :return accuracy (0d tensor)
    """
    return torch.mean((pos_scores<neg_scores).float()).item()
    
def evaluate_dataset(scores:torch.Tensor):
    """
        Compute score for `bs` set of [pos, neg, neg .....] quints.
        Assume pos is at the first position.
        
        
        :param scores: torch tensor of scores (bs,neg_samples+1)
        :returns (acc, mrr) both 1d tensors.
    """
    accuracy = (torch.argmin(scores, dim=1)==0).float()
    ranks = (torch.argsort(scores, dim=1) == 0).nonzero()[:,1]
    print(ranks)
    recirank = 1.0/(ranks+1).float()
    
    return accuracy.detach().cpu().numpy(), recirank.detach().cpu().numpy()
   

if __name__ == "__main__":
    class DummyModel(nn.Module):
        num_entities = 15000


    pos_data = np.random.randint(0, 15000, (400000, 5))
    bs = 2
    filtered = True
    quint = True

    eb = EvaluationBench(data=pos_data, model=DummyModel(), bs=bs, _filtered=filtered, _quints=quint)

    ps = pos_data[0]
    print(ps)
    with Timer() as timer:
        ng = eb._get_negatives_(ps)
    print(ng.shape, timer.interval, timer.interval * pos_data.shape[0])
    print(ng)
