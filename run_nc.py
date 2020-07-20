import os

os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import wandb
import sys
import collections
# from torchsummary import summary


# from apex import amp

# MyTorch imports
# from mytorch.utils.goodies import *

# Local imports
# from parse_wd15k import Quint
# from load import DataManager
from data_manager import DataManager
from utils import *
from utils_mytorch import FancyDict, parse_args, BadParameters, mt_save_dir
from evaluation import EvaluationBench, EvaluationBenchArity, \
    EvaluationBenchGNNMultiClass, evaluate_pointwise, eval_classification
from evaluation import acc, mrr, mr, hits_at
from models import TransE, ConvKB, KBGat, CompGCNConvE, CompGCNDistMult, CompGCNTransE, \
    CompGCNTransEStatements, CompGCNDistMultStatement, CompGCNConvEStatement, CompGCN_ConvKB, \
    CompGCN_ConvKB_Statement, CompGCN_ConvKB_Hinge_Statement, CompGCN_Transformer_Triples, ConvE_Triple_Baseline, \
    Transformer_Baseline
from models_statements import CompGCN_Transformer, CompGCN_ConvPar, CompGCN_ObjectMask_Transformer, \
    CompGCN_Transformer_TripleBaseline, Transformer_Statements, RGAT_Transformer
from models_nc import StarE_NC
from corruption import Corruption
from sampler import SimpleSampler, NeighbourhoodSampler, MultiClassSampler, NodeClSampler
from loops import training_loop, training_loop_neighborhood, training_loop_gcn, training_loop_node_classification
from clean_datasets import load_nodecl_dataset

"""
    CONFIG Things
"""

# Clamp the randomness
np.random.seed(42)
random.seed(42)
torch.manual_seed(132)


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
    'MODEL_NAME': 'stare',
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
    'WANDB': False,
    'LABEL_SMOOTHING': 0.0,
    'SAMPLER_W_QUALIFIERS': False,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True,

    'CL_TASK': 'so',  # so or full
    'SWAP': False
}

KBGATARGS = {
    'OUT': 25,
    'HEAD': 3,
    'ALPHA': 0.5
}

COMPGCNARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 80,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'rotate',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    'QUAL_OPN': 'rotate',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,

    # For TransE
    'GAMMA': 40.0,

    # For ConvE Only
    'HID_DROP2': 0.1,
    'FEAT_DROP': 0.3,
    'N_FILTERS': 200,
    'KERNEL_SZ': 7,
    'K_W': 5,  # def 10
    'K_H': 10,  # def 20,

    # For Hinge Only
    'MULTI_CONVS': False,

    # For Transformer
    'T_LAYERS': 2,
    'T_N_HEADS': 4,
    'T_HIDDEN': 512,
    'POSITIONAL': True,
    'POS_OPTION': 'default',
    'TIME': False,
    'POOLING': 'concat'

}

DEFAULT_CONFIG['KBGATARGS'] = KBGATARGS
DEFAULT_CONFIG['COMPGCNARGS'] = COMPGCNARGS

if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    kbgatconfig = KBGATARGS.copy()
    gcnconfig = COMPGCNARGS.copy()
    parsed_args = parse_args(sys.argv[1:])
    print(parsed_args)

    # Superimpose this on default config
    for k, v in parsed_args.items():
        # If its a generic arg
        if k in config.keys():
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v
        # If its a compgcnarg
        elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
            default_val = gcnconfig[k[4:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                gcnconfig[k[4:].upper()] = needed_type(v)
            else:
                gcnconfig[k[4:].upper()] = v
        # If its a kbgatarg
        elif k.lower().startswith('kbgat_') and k[6:] in kbgatconfig:
            default_val = kbgatconfig[k[6:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                kbgatconfig[k[6:].upper()] = needed_type(v)
            else:
                kbgatconfig[k[6:].upper()] = v

        else:
            config[k.upper()] = v

    config['KBGATARGS'] = kbgatconfig
    config['COMPGCNARGS'] = gcnconfig

    data = load_nodecl_dataset(name=config["DATASET"],
                               subtype="triples" if config["STATEMENT_LEN"] == 3 else "statements",
                               task=config["CL_TASK"],
                               maxlen=config["MAX_QPAIRS"])

    config['NUM_ENTITIES'] = data["n_entities"]
    config['NUM_RELATIONS'] = data["n_relations"]
    train_mask, val_mask, test_mask = data["train_mask"], data["valid_mask"], data["test_mask"]
    train_y, val_y, test_y = data["train_y"], data["val_y"], data["test_y"]
    all_labels, label2id = data["all_labels"], data["label2id"]
    graph = data["graph"]
    config['NUM_CLASSES'] = len(all_labels)

    if config['USE_TEST']:
        input_data = {"train": train_y, "eval": test_y}
    else:
        if config['SWAP']:
            input_data = {"train": val_y, "eval": train_y}
        else:
            input_data = {"train": train_y, "eval": val_y}

    if config['MODEL_NAME'].lower().startswith('stare'):
        # Replace the data with their graph repr formats
        if config['COMPGCNARGS']['QUAL_REPR'] == 'sparse':
            train_data_gcn = DataManager.get_alternative_graph_repr(graph, config)
        else:
            print("Supported QUAL_REPR is `sparse`")
            raise NotImplementedError
        # add reciprocals to the train data
        # reci = DataManager.add_reciprocals(graph, config)
        # train_data_gcn.extend(reci)
    else:
        train_data_gcn, valid_data_gcn, test_data_gcn = None, None, None

    print(f"Training on {len(input_data['train'])} entities")

    config['DEVICE'] = torch.device(config['DEVICE'])

    if config['MODEL_NAME'].lower() == 'stare':
        model = StarE_NC(train_data_gcn, config)
    # if config['MODEL_NAME'].lower() == 'transe':
    #     model = TransE(config)
    # elif config['MODEL_NAME'].lower() == 'convkb':
    #     model = ConvKB(config)
    # elif config['MODEL_NAME'].lower().startswith('compgcn'):
    #     if config['MODEL_NAME'].lower().endswith('transe'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             model = CompGCNTransEStatements(train_data_gcn, config)
    #         else:
    #             model = CompGCNTransE(train_data_gcn, config)
    #     elif config['MODEL_NAME'].lower().endswith('conve'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             model = CompGCNConvEStatement(train_data_gcn, config)
    #         else:
    #             model = CompGCNConvE(train_data_gcn, config)
    #     elif config['MODEL_NAME'].lower().endswith('conve_baseline'):
    #         model = ConvE_Triple_Baseline(config)
    #     elif config['MODEL_NAME'].lower().endswith('trans_baseline'):
    #         model = Transformer_Baseline(config)
    #     elif config['MODEL_NAME'].lower().endswith('triple_baseline'):
    #         assert config['SAMPLER_W_QUALIFIERS'] is True, "only works for qual-aware encoder"
    #         model = CompGCN_Transformer_TripleBaseline(train_data_gcn, config)
    #     elif config['MODEL_NAME'].lower().endswith('stats_baseline'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             if config['COMPGCNARGS']['TIME']:
    #                 e2id = data['e2id']
    #                 id2e = {v: k for k, v in e2id.items()}
    #                 tstoid = data['r2id'][1]
    #                 model = Transformer_Statements(config, (id2e, tstoid))
    #             else:
    #                 model = Transformer_Statements(config)
    #         else:
    #             raise NotImplementedError
    #     elif config['MODEL_NAME'].lower().endswith('distmult'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             model = CompGCNDistMultStatement(train_data_gcn, config)
    #         else:
    #             model = CompGCNDistMult(train_data_gcn, config)
    #     elif config['MODEL_NAME'].lower().endswith('convkb'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             print(
    #                 f"ConvKB will use {(config['MAX_QPAIRS'] - 1, config['COMPGCNARGS']['KERNEL_SZ'])} kernel. Otherwize change KERNEL_SZ param. Standard is 1")
    #             model = CompGCN_ConvKB_Statement(train_data_gcn, config)
    #         else:
    #             print(
    #                 f"ConvKB will use {(2, config['COMPGCNARGS']['KERNEL_SZ'])} kernel. Otherwize change KERNEL_SZ param. Standard is 1")
    #             model = CompGCN_ConvKB(train_data_gcn, config)
    #     elif config['MODEL_NAME'].lower().endswith('transformer'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             if config['COMPGCNARGS']['TIME']:
    #                 e2id = data['e2id']
    #                 id2e = {v: k for k, v in e2id.items()}
    #                 tstoid = data['r2id'][1]
    #                 if 'objectmask' in config['MODEL_NAME']:
    #                     model = CompGCN_ObjectMask_Transformer(train_data_gcn, config, (id2e, tstoid))
    #                 else:
    #                     model = CompGCN_Transformer(train_data_gcn, config, (id2e, tstoid))
    #             else:
    #                 if 'objectmask' in config['MODEL_NAME']:
    #                     model = CompGCN_ObjectMask_Transformer(train_data_gcn, config)
    #                 else:
    #                     model = CompGCN_Transformer(train_data_gcn, config)
    #         else:
    #             model = CompGCN_Transformer_Triples(train_data_gcn, config)
    #             # print("Transformer decoder is for qual decoder only (so far)")
    #             # raise NotImplementedError
    #     elif config['MODEL_NAME'].lower().endswith('rgat'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             model = RGAT_Transformer(train_data_gcn, config)
    #         else:
    #             print("ConvPar decoder is for qual decoder only (so far)")
    #             raise NotImplementedError
    #     elif config['MODEL_NAME'].lower().endswith('convpar'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             model = CompGCN_ConvPar(train_data_gcn, config)
    #         else:
    #             print("ConvPar decoder is for qual decoder only (so far)")
    #             raise NotImplementedError
    #     elif config['MODEL_NAME'].lower().endswith('hinge'):
    #         if config['SAMPLER_W_QUALIFIERS']:
    #             print(
    #                 f"ConvKB will use {(config['MAX_QPAIRS'] - 1, config['COMPGCNARGS']['KERNEL_SZ'])} kernel. Otherwize change KERNEL_SZ param. Standard is 1")
    #             model = CompGCN_ConvKB_Hinge_Statement(train_data_gcn, config)
    #         else:
    #             raise NotImplementedError("Have to implement CompGCN-ConvKB-Hinge model for non statements.")
    #             print(
    #                 f"ConvKB will use {(2, config['COMPGCNARGS']['KERNEL_SZ'])} kernel. Otherwize change KERNEL_SZ param. Standard is 1")
    #             model = CompGCN_ConvKB_Hinge_Statement(train_data_gcn, config)
    #     else:
    #         raise BadParameters(f"Unknown Model Name {config['MODEL_NAME']}")
    else:
        raise BadParameters(f"Unknown Model Name {config['MODEL_NAME']}")

    # adding multi-gpu training support
    # if torch.cuda.device_count() > 1:
    #     print("Using ", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(config['DEVICE'])
    print("Model params ", sum([param.nelement() for param in model.parameters()]))

    if config['OPTIMIZER'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    else:
        print("Unexpected optimizer, we support `sgd` or `adam` at the moment")
        raise NotImplementedError


    #model, optimizer = amp.initialize(model, optimizer,opt_level='O1')
    if config['WANDB']:
        wandb.init(project="wikidata-embeddings")
        for k, v in config.items():
            wandb.config[k] = v

    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    args = {
        "epochs": config['EPOCHS'],
        "opt": optimizer,
        "train_fn": model,
        "device": config['DEVICE'],
        "eval_fn": eval_classification,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": config['RUN_TESTBENCH_ON_TRAIN'],
        "savedir": savedir,
        "save_content": save_content,
        "qualifier_aware": config['SAMPLER_W_QUALIFIERS'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "scheduler": None
    }

    if config['MODEL_NAME'].lower().startswith('stare'):
        training_loop = training_loop_node_classification
        sampler = NodeClSampler(data=input_data,
                                num_labels=len(all_labels),
                                label2id=label2id,
                                lbl_smooth=config['LABEL_SMOOTHING'])

        args['data_fn'] = sampler.get_data
        args['criterion'] = torch.nn.BCEWithLogitsLoss(pos_weight=sampler.pos_weights.to(config['DEVICE']))

        if config['LR_SCHEDULER']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
            args['scheduler'] = scheduler



    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)

