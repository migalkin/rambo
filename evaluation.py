from functools import partial
from tqdm import tqdm
import types

# Local
from utils import *
from corruption import Corruption


class EvaluationBench:
    """
    Sampler which for each true triple,
        |-> compares it with **all** possible negative triples, and reports metrics
    """

    def __init__(self, data: Dict[str, Union[List[int], np.array]], model: nn.Module,
                 bs: int, metrics: list, _filtered: bool = False, trim: float = None):
        """
            :param data: {'train': list/iter of positive triples, 'valid': list/iter of positive triples}.
            Np array are appreciated
            :param model: the nn module we're testing
            :param bs: anything under 256 is shooting yourself in the foot.
            :param _filtered: if you want corrupted triples checked.
            """
        self.bs, self.filtered = bs, _filtered
        self.model = model
        self.data_valid = data['valid']
        self.metrics = metrics

        # Find the kind of data we're dealing with
        self.max_len_data = max(data['train'].shape[1], data['valid'].shape[1])
        self.corruption_positions = list(range(0, self.max_len_data, 2))

        # Create a corruption object
        self.corrupter = Corruption(n=self.model.num_entities, position=self.corruption_positions, debug=False,
                                    gold_data=np.vstack((data['train'], data['valid'])) if self.filtered else None)

        if trim is not None:
            assert trim <= 1.0, "Trim ratio can not be more than 1.0"
            self.data_valid = np.random.permutation(self.data_valid)[:int(trim*len(self.data_valid))]

    def reset(self):
        """ Call when you wanna run again but not change hashes etc """
        raise NotImplementedError

    def _compute_metric_(self, scores: np.array) -> List[Union[float, np.float]]:
        """ See what metrics are to be computed, and compute them."""
        return [_metric(scores) for _metric in self.metrics]

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
        print(f"This run over {summary['data_length']} datapoints took "
              f"%(time).3f min" % {'time': summary['time_taken'] / 60.0})
        print("---------\n")
        for k, v in summary['metrics'].items():
            print(k, ':', "%(v).4f" % {'v': v})

    def run(self, *args, **kwargs):
        """
            Calling this iterates through different data points, generates negatives, passes them to the model,
                collects the scores, computes the metrics, and reports them.

            Update: run functino now actually goes over only
        """
        metrics = []

        with Timer() as timer:
            with torch.no_grad():
                for positive_data in tqdm(self.data_valid):

                    metric_across_positions = []

                    # Same pos data will be looped over for each position, its result stored separately
                    for position in self.corruption_positions:

                        neg = self.corrupter.corrupt_one_position(positive_data, position)

                        if len(neg) + 1 < self.bs:  # Can do it in one iteration

                            x = torch.tensor(np.vstack((positive_data.transpose(), neg)), dtype=torch.long,
                                             device=self.model.config['DEVICE'])
                            scores = self.model.predict(x)

                        else:
                            scores = torch.tensor([], dtype=torch.float, device=self.model.config['DEVICE'])

                            for i in range(neg.shape[0])[::self.bs]:  # Break it down into batches and then do dis
                                _neg = neg[i: i + self.bs]
                                if i == 0:
                                    x = torch.tensor(np.vstack((positive_data.transpose(), _neg)), dtype=torch.long,
                                                     device=self.model.config['DEVICE'])
                                else:
                                    x = torch.tensor(_neg, dtype=torch.long, device=self.model.config['DEVICE'])
                                _scores = self.model.predict(x)

                                scores = torch.cat((scores, _scores))

                        _metrics = self._compute_metric_(scores)
                        metric_across_positions.append(_metrics)

                    metrics.append(np.mean(metric_across_positions, axis=0))
        # Spruce up the summary with more information
        time_taken = timer.interval
        metrics = self._summarize_metrics_(metrics)
        summary = {'metrics': metrics, 'time_taken': time_taken, 'data_length': len(self.data_valid),
                   'max_len_data': self.max_len_data, 'filtered': self.filtered}

        self.summarize_run(summary)

        return summary

    def run_faster(self, *args, **kwargs):
        """
            Similar to run but tries to put multiple pos in one batch if permitted by bs
            TODO
        :param args:
        :param kwargs:
        :return:
        """
        ...


def acc(scores: torch.Tensor) -> np.float:
    """ Accepts a (n, ) tensor """
    return (torch.argmin(scores, dim=0) == 0).float().detach().cpu().numpy().item()


def mrr(scores: torch.Tensor) -> np.float:
    """ Tested | Accepts one (n,) tensor """
    ranks = (torch.argsort(scores, dim=0) == 0).nonzero()[0]
    recirank = 1.0 / (ranks + 1).float()
    return recirank.detach().cpu().numpy().item()


def mr(scores: torch.Tensor) -> np.float:
    """ Tested | Accepts one (n,) tensor """
    ranks = (torch.argsort(scores, dim=0) == 0).nonzero()[0]
    return ranks.detach().cpu().numpy().item()


def hits_at(scores: torch.Tensor, k: int = 5) -> float:
    """ Tested | Accepts one (n,) tensor """
    rank = (torch.argsort(scores, dim=0) == 0).nonzero()[0] + 1
    if rank <= k:
        return 1.0
    else:
        return 0.0


def evaluate_pointwise(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> Union[int, float, bool]:
    """
        Given a pos and neg quint, how many times did the score for positive be more than score for negative

        :param pos_scores: scores corresponding to pos quints (bs, )
        :param neg_scores: scores corresponding to neg quints (bs, )
        :return accuracy (0d tensor)
    """
    return torch.mean((pos_scores < neg_scores).float()).item()


def evaluate_dataset(scores: torch.Tensor):
    """
        Compute score for `bs` set of [pos, neg, neg .....] quints.
        Assume pos is at the first position.


        :param scores: torch tensor of scores (bs,neg_samples+1)
        :returns (acc, mrr) both 1d tensors.
    """
    accuracy = (torch.argmin(scores, dim=1) == 0).float()
    ranks = (torch.argsort(scores, dim=1) == 0).nonzero()[:, 1]
    recirank = 1.0 / (ranks + 1).float()

    return accuracy.detach().cpu().numpy(), recirank.detach().cpu().numpy()


if __name__ == "__main__":
    class DummyModel(nn.Module):
        num_entities = 15000

    pos_data = np.random.randint(0, 15000, (400000, 5))
    pos_data_vl = np.random.randint(0, 15000, (400000, 5))
    data = {'train': pos_data, 'valid': pos_data_vl}
    bs = 2
    filtered = True
    quint = True

    eb = EvaluationBench(data=data, model=DummyModel(), bs=bs, _filtered=filtered, metrics=[acc, mrr])

    ps = pos_data[0]
    print(ps)
    with Timer() as timer:
        ng = eb.corrupter.corrupt_one(ps)
    print(ng.shape, timer.interval, timer.interval * pos_data.shape[0])
    print(ng)
