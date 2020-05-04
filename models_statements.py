import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from gnn_encoder import CompQGCNEncoder


class TimeEncode(torch.nn.Module):
    """
     The implementation is similar to https://openreview.net/pdf?id=rJeW1yHYwH
    """
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        # reduced linspace from (0,9) to (0,3) as in our temporal datasets min-max variance is of 3rd order of magnitude
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 3, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)

        return harmonic

class CompGCN_Transformer(CompQGCNEncoder):
    model_name = 'CompGCN_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.num_transformer_layers = config['COMPGCNARGS']['T_LAYERS']
        self.num_heads = config['COMPGCNARGS']['T_N_HEADS']
        self.num_hidden = config['COMPGCNARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['COMPGCNARGS']['POSITIONAL']
        self.time = config['COMPGCNARGS']['TIME']  # treat qual values as numbers and pass them through the t_enc
        self.pooling = config['COMPGCNARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['COMPGCNARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['COMPGCNARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        if self.time:
            self.time_encoder = TimeEncode(self.d_model)
            self.id2e = id2e[0]
            self.tstoid = id2e[1]
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)
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
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            """
                op 2 - we want positional encodings to reflect key-value pairs along with the main triple
                s p qp qe qp qe 0 0 0 0
                1 1 2  2  3  3  0 0 0 0 
            """
            # positions[:, 1::2] = positions[:, 0::2]  # turning 0 1 2 3 4 5 6 7 into 0 0 2 2 4 4 6 6
            # positions = (positions // 2) + 1  # turning into 1 2 3 4
            # positions = positions * (1 - mask.int())  # turning into 1 2 3 4 0 0 0 0 for masked positions
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            if self.time:
                # TODO: Time Magic here, mike tested for yago and icews, but debug if you want
                time_embeddings = torch.zeros((stk_inp.shape[0], stk_inp.shape[1], stk_inp.shape[2]), dtype=torch.float, device=self.device)
                # get qual values which are not padding indices
                qual_non_padded = quals * (1 - mask[:, 2:].int())
                quals_indices = qual_non_padded[:, 1::2].reshape(1, -1).squeeze(0)  # get only values and flatten
                # get their numerical values from the dictionary
                quals_values = [float(self.tstoid[self.id2e[int(x)]]) + 1 if int(x) != 0 else 0 for x in quals_indices]
                vals = torch.tensor(quals_values, dtype=torch.float, device=self.device).view(stk_inp.shape[1], quals.shape[1] // 2).transpose(1,0)
                timed_vals = self.time_encoder(vals)  # shape: quals/2, bs, emb_dim
                time_embeddings[3::2, :, :] = timed_vals
                # zeroify padding indices
                time_embeddings = time_embeddings * (1 - mask.int()).transpose(1, 0).unsqueeze(2)
                stk_inp = stk_inp + pos_embeddings + time_embeddings
            else:
                stk_inp = stk_inp + pos_embeddings

        # stk_inp = self.layer_norm(stk_inp)
        # stk_inp = self.hidden_drop2(stk_inp)
        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)
        # x = self.hidden_drop2(x)
        # x = self.bn2(x)
        # x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        # x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ObjectMask_Transformer(CompQGCNEncoder):
    model_name = 'CompGCN_ObjectMask_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.num_transformer_layers = config['COMPGCNARGS']['T_LAYERS']
        self.num_heads = config['COMPGCNARGS']['T_N_HEADS']
        self.num_hidden = config['COMPGCNARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['COMPGCNARGS']['POSITIONAL']
        self.time = config['COMPGCNARGS']['TIME']  # treat qual values as numbers and pass them through the t_enc


        self.object_mask_emb = torch.nn.Parameter(torch.randn(1, self.emb_dim,dtype=torch.float32),True)
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['COMPGCNARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['COMPGCNARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'], self.d_model)
        if self.time:
            self.time_encoder = TimeEncode(self.d_model)
            self.id2e = id2e[0]
            self.tstoid = id2e[1]
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)
        # self._initialize()

    def concat(self, e1_embed, rel_embed, obj_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        obj_embed = obj_embed.view(-1,1, self.emb_dim)
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
        stack_inp = torch.cat([e1_embed, rel_embed, obj_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp        # 14, 128, 200

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)


        # bs*emb_dim , ......, bs*6*emb_dim
        object_mask = self.object_mask_emb.repeat(sub.shape[0], 1)
        ins = torch.zeros((sub.shape), dtype=torch.bool, device=self.device)
        mask = torch.cat((mask[:, :2], ins.unsqueeze(1), mask[:, 2:]), axis=1)

        stk_inp = self.concat(sub_emb, rel_emb, object_mask, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            """
                op 2 - we want positional encodings to reflect key-value pairs along with the main triple
                s p qp qe qp qe 0 0 0 0
                1 1 2  2  3  3  0 0 0 0 
            """
            # positions[:, 1::2] = positions[:, 0::2]  # turning 0 1 2 3 4 5 6 7 into 0 0 2 2 4 4 6 6
            # positions = (positions // 2) + 1  # turning into 1 2 3 4
            # positions = positions * (1 - mask.int())  # turning into 1 2 3 4 0 0 0 0 for masked positions
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            if self.time:
                # TODO: Time Magic here, mike tested for yago and icews, but debug if you want
                time_embeddings = torch.zeros((stk_inp.shape[0], stk_inp.shape[1], stk_inp.shape[2]), dtype=torch.float, device=self.device)
                # get qual values which are not padding indices
                qual_non_padded = quals * (1 - mask[:, 2:].int())
                quals_indices = qual_non_padded[:, 1::2].reshape(1, -1).squeeze(0)  # get only values and flatten
                # get their numerical values from the dictionary
                quals_values = [float(self.tstoid[self.id2e[int(x)]]) + 1 if int(x) != 0 else 0 for x in quals_indices]
                vals = torch.tensor(quals_values, dtype=torch.float, device=self.device).view(stk_inp.shape[1], quals.shape[1] // 2).transpose(1, 0)
                timed_vals = self.time_encoder(vals)  # shape: quals/2, bs, emb_dim
                time_embeddings[3::2, :, :] = timed_vals
                # zeroify padding indices
                time_embeddings = time_embeddings * (1 - mask.int()).transpose(1, 0).unsqueeze(2)
                stk_inp = stk_inp + pos_embeddings + time_embeddings
            else:
                stk_inp = stk_inp + pos_embeddings

        # stk_inp = self.layer_norm(stk_inp)
        # stk_inp = self.hidden_drop2(stk_inp)
        x = self.encoder(stk_inp, src_key_padding_mask=mask)[2] # to get the object position

        # x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        x = self.fc(x)
        # x = self.hidden_drop2(x)
        # x = self.bn2(x)
        # x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        # x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ConvPar(CompQGCNEncoder):
    model_name = 'CompGCN_ConvPar_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_ConvKB_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.n_filters = config['COMPGCNARGS']['N_FILTERS']
        self.kernel_sz = config['COMPGCNARGS']['KERNEL_SZ']
        # self.bias = config['COMPGCNARGS']['BIAS']

        self.entity_embedding_norm_type: int = 2
        self.l_p_norm_entities = config['NORM_FOR_NORMALIZATION_OF_ENTITIES']
        self.scoring_fct_norm = config['SCORING_FUNCTION_NORM']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        self.conv_layers = nn.ModuleList()
        self.flat_sizes = dict()
        for i in range(2, config['MAX_QPAIRS'], 2):
            self.conv_layers.append(
                torch.nn.Conv2d(1, out_channels=self.n_filters,
                                kernel_size=(i, self.kernel_sz),
                                stride=(config['MAX_QPAIRS'] - 1, 1),  # to limit to only one vertical pass along H
                                padding=0,
                                bias=config['COMPGCNARGS']['BIAS'])
            )
            current_width = self.emb_dim - self.kernel_sz + 1
            current_height = 1  # config['MAX_QPAIRS'] - i
            current_flat_size = current_width * current_height
            self.flat_sizes[i // 2] = current_flat_size
        self.temp_flat_sz = self.emb_dim - self.kernel_sz + 1
        self.flat_sz = self.n_filters * (self.emb_dim - self.kernel_sz + 1)
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
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).unsqueeze(1)  # [bs, 1, 2 + num_qual_pairs, emb_dim]
        #stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)
        x = self.bn0(stk_inp)

        temp = torch.zeros((stk_inp.shape[2] // 2, stk_inp.shape[0], self.n_filters, self.temp_flat_sz),
                           device=self.device)
        for i, layer in enumerate(self.conv_layers):
            output = layer(x)
            output = F.relu(output)
            output = self.bn2(output)
            output = output.view(-1, self.n_filters, self.flat_sizes[i+1])
            temp[i, :, :, :] = output

        x, _ = torch.min(temp, 0)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

class CompGCN_Transformer_TripleBaseline(CompQGCNEncoder):
    model_name = 'CompGCN_Transformer_Triple_Baseline'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['COMPGCNARGS']['HID_DROP2']
        self.feat_drop = config['COMPGCNARGS']['FEAT_DROP']
        self.num_transformer_layers = config['COMPGCNARGS']['T_LAYERS']
        self.num_heads = config['COMPGCNARGS']['T_N_HEADS']
        self.num_hidden = config['COMPGCNARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['COMPGCNARGS']['POSITIONAL']
        self.time = config['COMPGCNARGS']['TIME']  # treat qual values as numbers and pass them through the t_enc
        self.pooling = config['COMPGCNARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['COMPGCNARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['COMPGCNARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        if self.time:
            self.time_encoder = TimeEncode(self.d_model)
            self.id2e = id2e[0]
            self.tstoid = id2e[1]
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)
        # self._initialize()

    def concat(self, e1_embed, rel_embed):
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
        #quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)
        mask = mask[:, :2]

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            """
                op 2 - we want positional encodings to reflect key-value pairs along with the main triple
                s p qp qe qp qe 0 0 0 0
                1 1 2  2  3  3  0 0 0 0 
            """
            # positions[:, 1::2] = positions[:, 0::2]  # turning 0 1 2 3 4 5 6 7 into 0 0 2 2 4 4 6 6
            # positions = (positions // 2) + 1  # turning into 1 2 3 4
            # positions = positions * (1 - mask.int())  # turning into 1 2 3 4 0 0 0 0 for masked positions
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            if self.time:
                # TODO: Time Magic here, mike tested for yago and icews, but debug if you want
                time_embeddings = torch.zeros((stk_inp.shape[0], stk_inp.shape[1], stk_inp.shape[2]), dtype=torch.float, device=self.device)
                # get qual values which are not padding indices
                qual_non_padded = quals * (1 - mask[:, 2:].int())
                quals_indices = qual_non_padded[:, 1::2].reshape(1, -1).squeeze(0)  # get only values and flatten
                # get their numerical values from the dictionary
                quals_values = [float(self.tstoid[self.id2e[int(x)]]) + 1 if int(x) != 0 else 0 for x in quals_indices]
                vals = torch.tensor(quals_values, dtype=torch.float, device=self.device).view(stk_inp.shape[1], quals.shape[1] // 2).transpose(1,0)
                timed_vals = self.time_encoder(vals)  # shape: quals/2, bs, emb_dim
                time_embeddings[3::2, :, :] = timed_vals
                # zeroify padding indices
                time_embeddings = time_embeddings * (1 - mask.int()).transpose(1, 0).unsqueeze(2)
                stk_inp = stk_inp + pos_embeddings + time_embeddings
            else:
                stk_inp = stk_inp + pos_embeddings

        # stk_inp = self.layer_norm(stk_inp)
        # stk_inp = self.hidden_drop2(stk_inp)
        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)
        # x = self.hidden_drop2(x)
        # x = self.bn2(x)
        # x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        # x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score