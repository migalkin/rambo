import torch
from data_manager import DataManager
from models_statements import CompGCN_Transformer
import json

# that dict was not saved, so redefining the saved config here
COMPGCNARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 200,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.1,
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
    'HID_DROP2': 0.3,
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
    'POOLING': 'avg'

}

config = json.load(open("models/stare_s/config.json","r"))
config['COMPGCNARGS'] = COMPGCNARGS
data = DataManager.load(config=config)()
try:
    train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()
except ValueError:
    raise ValueError(f"Honey I broke the loader for {config['DATASET']}")


train_data_gcn = DataManager.get_alternative_graph_repr(train_data + valid_data, config)
reci = DataManager.add_reciprocals(train_data, config)
train_data.extend(reci)
reci_valid = DataManager.add_reciprocals(valid_data, config)
valid_data.extend(reci_valid)
reci_test = DataManager.add_reciprocals(test_data, config)
test_data.extend(reci_test)

config['DEVICE'] = torch.device("cpu")

model = CompGCN_Transformer(train_data_gcn, config)
model.load_state_dict(torch.load("models/stare_s/model.torch", map_location=torch.device("cpu")))
model.eval()

data_p = torch.tensor(test_data[5:7], dtype=torch.long)
sub, rel, quals = data_p[:, 0], data_p[:, 1], data_p[:, 3:].contiguous()
preds = model(sub, rel, quals)
res = torch.argsort(preds, dim=1, descending=True)
for i in range(res.shape[0]):
    print(f"Correct: {data_p[i][2]}. Top 10 predictions: ",res[i][:10])

