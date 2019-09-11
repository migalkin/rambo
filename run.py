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
from load import DataManager
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

"""
    Explanation:
        *ENT_POS_FILTERED* 
            a flag which if False, implies that while making negatives, 
                we should exclude entities that appear ONLY in non-corrupting positions.
            Do not turn it off if the experiment is about predicting qualifiers, of course.

        *POSITIONS*
            the positions on which we should inflect the negatives.
"""
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
    'IS_QUINTS': None,
    'EVAL_EVERY': 10,
    'WANDB': True,
    'RUN_TESTBENCH_ON_TRAIN': True,
    'DATASET': 'wd15k',
    'CORRUPTION_POSITIONS': [0, 2],
    'DEVICE': 'cuda',
    'ENT_POS_FILTERED': True,
    'USE_TEST': False
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
            default_val = DEFAULT_CONFIG[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                DEFAULT_CONFIG[k.upper()] = needed_type(v)
            else:
                DEFAULT_CONFIG[k.upper()] = v

    # Custom Sanity Checks
    if DEFAULT_CONFIG['DATASET'] == 'wd15k':
        assert DEFAULT_CONFIG['IS_QUINTS'] is not None, "You use WD15k dataset and don't specify whether to treat them " \
                                                     "as quints or not. Nicht cool'"
    if max(DEFAULT_CONFIG['CORRUPTION_POSITIONS']) > 2:     # If we're corrupting something apart from S and O
        assert DEFAULT_CONFIG['ENT_POS_FILTERED'] is False, f"Since we're corrupting objects at pos. " \
                                                            f"{DEFAULT_CONFIG['CORRUPTION_POSITIONS']}," \
                                                            f"You must allow including entities which appear" \
                                                            f"exclusively in qualifiers, too!"

    """
        Load data based on the args/config
    """
    data = DataManager.load(config=DEFAULT_CONFIG)()
    try:
        training_triples, valid_triples, test_triples, num_entities, num_relations = data.values()
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {DEFAULT_CONFIG['DATASET']}")

    if DEFAULT_CONFIG['ENT_POS_FILTERED']:
        ent_excluded_from_corr = DataManager.gather_missing_entities(data=training_triples + valid_triples + test_triples,
                                                                     positions=DEFAULT_CONFIG['CORRUPTION_POSITIONS'],
                                                                     n_ents=num_entities)
        DEFAULT_CONFIG['NUM_ENTITIES_FILTERED'] = len(ent_excluded_from_corr)
    else:
        ent_excluded_from_corr = []
        DEFAULT_CONFIG['NUM_ENTITIES_FILTERED'] = len(ent_excluded_from_corr)

    print(num_entities-DEFAULT_CONFIG['NUM_ENTITIES_FILTERED'])
    DEFAULT_CONFIG['NUM_ENTITIES'] = num_entities
    DEFAULT_CONFIG['NUM_RELATIONS'] = num_relations


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
    if config['USE_TEST']:
        data = {'index': np.array(training_triples + valid_triples), 'eval': np.array(test_triples)}
        _data = {'index': np.array(valid_triples + test_triples), 'eval': np.array(training_triples)}
        tr_data = {'train': np.array(training_triples + valid_triples), 'valid': data['eval']}
    else:
        data = {'index': np.array(training_triples + test_triples), 'eval': np.array(valid_triples)}
        _data = {'index': np.array(valid_triples + test_triples), 'eval': np.array(training_triples)}
        tr_data = {'train': np.array(training_triples), 'valid': data['eval']}

    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]
    evaluation_valid = EvaluationBench(data, model, bs=8000,
                                       metrics=eval_metrics, filtered=True,
                                       n_ents=num_entities,
                                       excluding_entities=ent_excluded_from_corr,
                                       positions=config.get('CORRUPTION_POSITIONS', None))
    evaluation_train = EvaluationBench(_data, model, bs=8000,
                                       metrics=eval_metrics, filtered=True,
                                       n_ents=num_entities,
                                       excluding_entities=ent_excluded_from_corr,
                                       positions=config.get('CORRUPTION_POSITIONS', None), trim=0.01)

    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=num_entities, excluding=ent_excluded_from_corr,
                                    position=config.get('CORRUPTION_POSITIONS', [0, 2, 4] if config['IS_QUINTS'] else [0, 2])),
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
