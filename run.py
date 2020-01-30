"""
    The file which actually manages to run everything

    TODO: How do we init a model with another model?

    # Data.
    Typically, sending the config dict, and executing the returned function gives us data,
        in the form of
            -> train_data (list of list of 43 / 5 or 3 elements)
            -> valid_data
            -> test_data
            -> n_entities (an integer)
            -> n_relations (an integer)
            -> ent2id (dictionary to interpret the data above, if needed)
            -> rel2id

    However, when we want to run a GCN based model, we work with
        COO representations of triples, and
        adjacency representations of qualifiers.

        In this case, for each split: [train, valid, test], we return
        -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
        -> edge_type (n) array with [relation] corresponding to sub, obj above
        -> qual_rel (20 x n) matrix with [r_q1, r_q2, r_q3, ..., r_q20] for each relation above
        -> qual_ent (20 x n) matrix with [e_q1, e_q2, e_q3, ..., e_q20] for each relation above

        So here, train_data will be a dict containing these four ndarrays.

"""
import os
os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import wandb
import sys

# MyTorch imports
from mytorch.utils.goodies import *

# Local imports
from parse_wd15k import Quint
from load import DataManager
from utils import *
from evaluation import EvaluationBench, EvaluationBenchArity, evaluate_pointwise
from evaluation import  acc, mrr, mr, hits_at
from models import TransE, ConvKB, KBGat
from corruption import Corruption
from sampler import SimpleSampler, NeighbourhoodSampler
from loops import training_loop, training_loop_neighborhood

"""
    CONFIG Things
"""

# Clamp the randomness
np.random.seed(42)
random.seed(42)

"""
    TODO: Add detailed explanations for these.
    TODO: Shall we also make recipes here?
    
    Explanation:
        *ENT_POS_FILTERED* 
            a flag which if False, implies that while making negatives, 
                we should exclude entities that appear ONLY in non-corrupting positions.
            Do not turn it off if the experiment is about predicting qualifiers, of course.

        *POSITIONS*
            the positions on which we should inflect the negatives.
        
        *SELF_ATTENTION*: int
            If 1 -> 1D self attention is used
            If 2 -> 2D self attention is used
            Anything else, no self attention is used.
"""
DEFAULT_CONFIG = {
    'BATCH_SIZE': 512,
    'CORRUPTION_POSITIONS': [0, 2],
    'DATASET': 'wd15k',
    'DEVICE': 'cpu',
    'EMBEDDING_DIM': 50,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 1000,
    'EVAL_EVERY': 20,
    'LEARNING_RATE': 0.001,
    'MARGIN_LOSS': 5,
    'MAX_QPAIRS': 43,
    'MODEL_NAME': 'ConvKB',
    'NARY_EVAL': False,
    'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    'NEGATIVE_SAMPLING_TIMES': 10,
    'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    'NUM_FILTER': 5,
    'PROJECT_QUALIFIERS': False,
    'PRETRAINED_DIRNUM': '',
    'RUN_TESTBENCH_ON_TRAIN': True,
    'SAVE': False,
    'SELF_ATTENTION': 0,
    'SCORING_FUNCTION_NORM': 1,
    'STATEMENT_LEN': -1,
    'USE_TEST': False,
    'WANDB': False
}

KBGATARGS = {
    'OUT': 25,
    'HEAD': 3,
    'ALPHA': 0.5
}

DEFAULT_CONFIG['KBGATARGS'] = KBGATARGS

if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    parsed_args = parse_args(sys.argv[1:])
    print(parsed_args)

    # Superimpose this on default config
    # TODO- Needs to go to mytorch.
    for k, v in parsed_args.items():
        if k not in config.keys():
            config[k.upper()] = v
        else:
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v

    """
        Custom Sanity Checks
    """
    # If we're corrupting something apart from S and O
    if max(config['CORRUPTION_POSITIONS']) > 2:
        assert config['ENT_POS_FILTERED'] is False, \
            f"Since we're corrupting objects at pos. {config['CORRUPTION_POSITIONS']}, " \
            f"You must allow including entities which appear exclusively in qualifiers, too!"

    """
        Loading and preparing data
    """
    data = DataManager.load(config=config)()

    # Break down the data
    try:
        train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {config['DATASET']}")

    config['NUM_ENTITIES'] = n_entities
    config['NUM_RELATIONS'] = n_relations

    # KBGat Specific hashes (to compute neighborhood)
    if config['MODEL_NAME'].lower() == 'kbgat':
        assert config['DATASET'] == 'fb15k237'
        hashes = create_neighbourhood_hashes(data)
    else:
        hashes = None

    # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
    if DEFAULT_CONFIG['ENT_POS_FILTERED']:
        ent_excluded_from_corr = DataManager.gather_missing_entities(
            data=train_data + valid_data + test_data,
            positions=config['CORRUPTION_POSITIONS'],
            n_ents=n_entities)
    else:
        ent_excluded_from_corr = [0]

    if config['MODEL_NAME'].lower() == 'compgcn':
        # Replace the data with their graph repr formats
        train_data = DataManager.get_graph_repr(train_data, config)
        valid_data = DataManager.get_graph_repr(valid_data, config)
        test_data = DataManager.get_graph_repr(test_data, config)

    print(f"Training on {n_entities} entities")
    print(f"Evaluating on {n_entities - len(ent_excluded_from_corr)} entities")

    """
        Make ze model
    """
    config['DEVICE'] = torch.device(config['DEVICE'])

    if config['MODEL_NAME'].lower() == 'transe':
        model = TransE(config)
    elif config['MODEL_NAME'].lower() == 'convkb':
        model = ConvKB(config)
    elif config['MODEL_NAME'].lower() == 'kbgat':
        if config['PRETRAINED_DIRNUM'] != '':   # @TODO: how do we pull the models
            pretrained_models = ...
            raise NotImplementedError
        else:
            pretrained_models = None
        model = KBGat(config, pretrained_models)
    elif config['MODEL_NAME'].lower() == 'compgcn':
        ...
    else:
        raise AssertionError('Unknown Model Name')

    model.to(config['DEVICE'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])

    if config['WANDB']:
        wandb.init(project="wikidata-embeddings")
        for k, v in config.items():
            wandb.config[k] = v

    """
        Prepare test benches.
        
            When computing train accuracy (`ev_tr_data`), we wish to use all the other data 
                to avoid generating true triples during corruption. 
            Similarly, when computing test accuracy, we index train and valid splits 
                to avoid generating negative triples.
    """
    if config['USE_TEST']:
        ev_vl_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data, valid_data), 'valid': ev_vl_data['eval']}
    else:
        ev_vl_data = {'index': combine(train_data, test_data), 'eval': combine(valid_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data), 'valid': ev_vl_data['eval']}

    # if config['MODEL_NAME'].lower() == 'compgcn':
    #   TODO: DO something.

    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3),
                    partial(hits_at, k=5), partial(hits_at, k=10)]

    if not config['NARY_EVAL']:
        evaluation_valid = EvaluationBench(ev_vl_data, model, bs=8000, metrics=eval_metrics,
                                           filtered=True, n_ents=n_entities,
                                           excluding_entities=ent_excluded_from_corr,
                                           positions=config.get('CORRUPTION_POSITIONS', None))
        evaluation_train = EvaluationBench(ev_tr_data, model, bs=8000, metrics=eval_metrics,
                                           filtered=True, n_ents=n_entities,
                                           excluding_entities=ent_excluded_from_corr,
                                           positions=config.get('CORRUPTION_POSITIONS', None),
                                           trim=0.01)
    else:
        evaluation_valid = EvaluationBenchArity(ev_vl_data, model, bs=8000, metrics=eval_metrics,
                                                filtered=True, n_ents=n_entities,
                                                excluding_entities=ent_excluded_from_corr)
        evaluation_train = EvaluationBenchArity(ev_tr_data, model, bs=8000, metrics=eval_metrics,
                                                filtered=True, n_ents=n_entities,
                                                excluding_entities=ent_excluded_from_corr,
                                                trim=0.01)

    # Saving stuff
    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=n_entities, excluding=[0],
                                    position=list(range(0, config['MAX_QPAIRS'], 2))),
        "device": config['DEVICE'],
        "data_fn": partial(SimpleSampler, bs=config["BATCH_SIZE"]),
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run,
        "trn_testbench": evaluation_train.run,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": config['RUN_TESTBENCH_ON_TRAIN'],
        "savedir": savedir,
        "save_content": save_content
    }

    if config['MODEL_NAME'] == 'kbgat':
        # Change the data fn
        neg_generator = args.pop('neg_generator')
        args['data_fn'] = partial(NeighbourhoodSampler, bs=config["BATCH_SIZE"],
                                  corruptor=neg_generator, hashes=hashes)

        training_loop = training_loop_neighborhood

    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)
