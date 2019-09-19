from run import *

DEFAULT_CONFIG['DATASET'] = 'wd15k'
DEFAULT_CONFIG['STATEMENT_LEN'] = -1
DEFAULT_CONFIG['ENT_POS_FILTERED'] = True
DEFAULT_CONFIG['WANDB'] = False
DEFAULT_CONFIG['SELF_ATTENTION'] = 1


data = DataManager.load(config=DEFAULT_CONFIG)()
training_triples, valid_triples, test_triples, num_entities, num_relations = data.values()

ent_excluded_from_corr = DataManager.gather_missing_entities(
            data=training_triples + valid_triples + test_triples,
            positions=DEFAULT_CONFIG['CORRUPTION_POSITIONS'],
            n_ents=num_entities)
DEFAULT_CONFIG['NUM_ENTITIES_FILTERED'] = len(ent_excluded_from_corr)

print(num_entities - DEFAULT_CONFIG['NUM_ENTITIES_FILTERED'])
DEFAULT_CONFIG['NUM_ENTITIES'] = num_entities
DEFAULT_CONFIG['NUM_RELATIONS'] = num_relations

config = DEFAULT_CONFIG.copy()
config['DEVICE'] = torch.device(config['DEVICE'])
model = TransE(config)
model.to(config['DEVICE'])
optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])

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
    "neg_generator": Corruption(n=num_entities, excluding=[0],
                                position=list(range(0, config['MAX_QPAIRS'], 2))),
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