import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from gnn_encoder import CompQGCNEncoder, CompGCNBase, CompQGCNEncoder_NC
from rgat_layer import RGATEncoder
from utils_gcn import get_param

class StarE_NC(CompQGCNEncoder_NC):
    model_name = 'StarE for node classification'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_NC'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.node_emb_dim = config['EMBEDDING_DIM']
        self.num_classes = config['NUM_CLASSES']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        self.to_classes = nn.Linear(self.node_emb_dim, self.num_classes)


    def forward(self, train_mask):
        '''
        :param train_mask: nodes for classification
        :return:
        '''
        all_ent, rels = self.forward_base(self.hidden_drop, self.feature_drop)
        nodes = torch.index_select(all_ent, 0, train_mask)

        probs = self.to_classes(nodes)
        probs = torch.sigmoid(probs)
        return probs
