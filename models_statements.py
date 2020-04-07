import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from models import CompQGCNEncoder

class CompGCN_Transformer(CompQGCNEncoder):
    model_name = 'CompGCN_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.num_transformer_layers = config['COMPGCNARGS']['T_LAYERS']
        self.num_heads = config['COMPGCNARGS']['T_N_HEADS']
        self.num_hidden = config['COMPGCNARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['COMPGCNARGS']['FEAT_DROP'])
        self.encoder = TransformerEncoder(encoder_layers, config['COMPGCNARGS']['T_LAYERS'])

        self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        # self._initialize()

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        x = self.fc(x)
        # x = self.hidden_drop2(x)
        # x = self.bn2(x)
        # x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score
