"""
    The file which actually manages to run everything
"""

from functools import partial
from pprint import pprint
import random
import wandb
import sys

# MyTorch imports
from mytorch.utils.goodies import *

# Local imports
from parse_wd15k import Quint
from load import IntelligentLoader
from utils import *
from evaluation import EvaluationBench, acc, mrr, mr, hits_at, evaluate_pointwise
from models import TransE
from corruption import Corruption
from sampler import SimpleSampler
from loops import training_loop

"""
    CONFIG Things
"""

# Clamp the randomness
np.random.seed(42)
random.seed(42)

DEFAULT_CONFIG = {
    'EMBEDDING_DIM': 50,
    'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    'SCORING_FUNCTION_NORM': 1,
    'MARGIN_LOSS': 1,
    'LEARNING_RATE': 0.001,
    'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    'NEGATIVE_SAMPLING_TIMES': 10,
    'BATCH_SIZE': 64,
    'EPOCHS': 1000,
    'IS_QUINTS': True,
    'EVAL_EVERY': 10,
    'WANDB': True,
    'RUN_TESTBENCH_ON_TRAIN': True,
    'DATASET': 'wd15k',
    'POSITIONS': [0, 2],
    'DEVICE': 'cuda'
}

if __name__ == "__main__":

    # Get parsed arguments
    parsed_args = parse_args(sys.argv[1:])

    print(parsed_args)

    # Superimpose this on default config
    # TODO- Needs to go to mytorch.
    for k, v in parsed_args.items():
        if k not in DEFAULT_CONFIG.keys():
            DEFAULT_CONFIG[k.upper()] = v
        else:
            needed_type = type(DEFAULT_CONFIG[k.upper()])
            DEFAULT_CONFIG[k.upper()] = needed_type(v)

    data = IntelligentLoader.get_dataset(config=DEFAULT_CONFIG)()
    try:
        training_triples, valid_triples, test_triples, num_entities, num_relations = data.values()
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {DEFAULT_CONFIG['DATASET']}")

    # Custom Sanity Checks
    if DEFAULT_CONFIG['DATASET'] == 'wd15k':
        assert 'is_quints' in lowerall(parsed_args), "You use WD15k dataset and don't specify whether to treat them " \
                                                     "as quints or not. Nicht cool'"

    DEFAULT_CONFIG['NUM_ENTITIES'] = num_entities
    DEFAULT_CONFIG['NUM_RELATIONS'] = num_relations

    pprint(DEFAULT_CONFIG)

    """
        Make ze model
    """
    config = DEFAULT_CONFIG.copy()
    config['DEVICE'] = torch.device(config['DEVICE'])
    model = TransE(config)
    model.to(config['DEVICE'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])

    if config['WANDB']:
        wandb.init(project="wikidata-embeddings",
                   notes=config.get('NOTES', ''))
        for k, v in config.items():
            wandb.config[k] = v

    """
        Prepare test benches
    """
    data = {'index': np.array(training_triples + test_triples), 'eval': np.array(valid_triples)}
    _data = {'index': np.array(valid_triples + test_triples), 'eval': np.array(training_triples)}

    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]
    evaluation_valid = EvaluationBench(data, model, 8000,
                                       metrics=eval_metrics, _filtered=True,
                                       positions=config.get('POSITIONS', None))
    evaluation_train = EvaluationBench(_data, model, 8000,
                                       metrics=eval_metrics, _filtered=True,
                                       positions=config.get('POSITIONS', None), trim=0.01)

    # RE-org the data
    data = {'train': data['index'], 'valid': data['eval']}

    args = {
        "epochs": config['EPOCHS'],
        "data": data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=num_entities,
                                    position=config.get('POSITIONS', [0, 2, 4] if config['IS_QUINTS'] else [0, 2])),
        "device": config['DEVICE'],
        "data_fn": partial(SimpleSampler, bs=config["BATCH_SIZE"]),
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run,
        "trn_testbench": evaluation_train.run,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": config['RUN_TESTBENCH_ON_TRAIN']
    }

    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)
