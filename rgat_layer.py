import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

from utils_gcn import get_param, MessagePassing, ccorr, rotate, softmax, xavier_normal_
from utils_mytorch import compute_mask
from utils import masked_softmax
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops
from gnn_encoder import CompGCNBase
from typing import Dict

class EdgeGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_rel, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, config=None, **kwargs):
        #super(EdgeGATConv, self).__init__(aggr='add', **kwargs)
        super(self.__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_rel = num_rel
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.p = config

        self.weight = get_param((in_channels, heads * out_channels))
        self.att = get_param((1, heads, 4 * out_channels))
        #self.w_q = get_param((in_channels, in_channels))
        self.w_q = get_param((heads * out_channels, heads * out_channels))
        self.w_r = get_param((in_channels, heads * out_channels))

        if self.concat:
            self.edge_update = get_param((heads * out_channels * 3, heads * out_channels))
            self.rel_proj = get_param((heads * out_channels, heads * out_channels))
        else:
            self.edge_update = get_param((out_channels * 3, out_channels))
            self.rel_proj = get_param((heads * out_channels, out_channels))

        self.device = None

        if bias and concat:
            self.bias = get_param((1, heads * out_channels))
        elif bias and not concat:
            self.bias = get_param((1, out_channels))
        else:
            self.register_parameter('bias', None)

        self.loop_rel = get_param((1, in_channels))  # (1,100)
        self.loop_ent = get_param((1, in_channels))  # new

        # self.reset_parameters()

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     glorot(self.att)
    #     zeros(self.bias)

    # def forward(self, x, edge_index, edge_type, rel_embed,
    #                 qualifier_ent=None, qualifier_rel=None, quals=None)
    # def forward(self, x, edge_index, size=None):
    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None):
        """"""
        if self.device is None:
            self.device = edge_index.device


        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if self.p['STATEMENT_LEN'] != 3:
            num_quals = quals.size(1) // 2
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is



        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_type = torch.cat([edge_type, self.loop_type], dim=0)

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
            rel_embed = torch.matmul(rel_embed, self.w_r)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        x = self.propagate('add', edge_index=edge_index, size=x.size(0), x=x, edge_type=edge_type,
                              rel_embed=rel_embed, ent_embed=x, qualifier_ent=quals[1],
                              qualifier_rel=quals[0], qual_index=quals[2])
        # if not self.concat:
        #     rel_embed = rel_embed.view(-1, self.heads, self.out_channels).mean(dim=1)
        return x, torch.mm(rel_embed[:-1], self.rel_proj)


    def message(self, edge_index, x_i, x_j, size, edge_type, rel_embed, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None):
        # Compute attention coefficients.
        edge_index_i = edge_index[0]
        x_j = x_j.view(-1, self.heads, self.out_channels)

        rels = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                     qualifier_rel, edge_type, qual_index)  # EDGES X [2*DIM]

        rels = rels.view(rels.shape[0], self.heads, -1)
        # if mode != 'loop':
        #
        # else:
        #     rel_emb = torch.index_select(rel_embed, 0, edge_type)
        # rels = rel_embed[edge_type].unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, rels], dim=-1)  # EDGES X [3*DIM]

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels*3)
        else:
            aggr_out = aggr_out.mean(dim=1)

        aggr_out = torch.mm(aggr_out, self.edge_update)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """
        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """
        #output = torch.zeros((num_edges, qual_embeddings.shape[1]), dtype=torch.float).to(self.device)
        # unq, unq_inv = torch.unique(qual_index, return_inverse=True)
        # np.add.at(out[:2, :], unq_inv, quals[:2, :])
        #ind = torch.LongTensor(qual_index)
        #output.index_add_(dim=0, index=qual_index, source=qual_embeddings)  # TODO check this magic carefully
        if self.p['COMPGCNARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges, fill_value=fill)
        elif self.p['COMPGCNARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges, fill_value=fill)
        # output = np.zeros((num_edges, qual_embeddings.shape[1]))
        # ind = qual_index.detach().cpu().numpy()
        # np.add.at(output, ind, qual_embeddings.detach().cpu().numpy())
        # output = torch.tensor(output, dtype=torch.float).to(self.device)
        return output

    def qual_transform(self, qualifier_ent, qualifier_rel):
        """

        :return:
        """
        if self.p['COMPGCNARGS']['QUAL_OPN'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.p['COMPGCNARGS']['QUAL_OPN'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.p['COMPGCNARGS']['QUAL_OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['COMPGCNARGS']['QUAL_OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed


    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):

        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        rel_part_emb = rel_embed[edge_type]

        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)
        qualifier_emb = torch.einsum('ij,jk -> ik',
                                     self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                     self.w_q)

        return torch.cat([rel_part_emb, qualifier_emb], dim=-1)




class RGATEncoder(CompGCNBase):
    def __init__(self, graph_repr: Dict[str, np.ndarray], config: dict, timestamps: dict = None):
        super().__init__(config)

        self.device = config['DEVICE']

        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)

        if not self.triple_mode:
            self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)

        self.gcn_dim = self.emb_dim if self.n_layer == 1 else self.gcn_dim

        self.init_embed = get_param((self.num_ent, self.emb_dim))
        self.init_embed.data[0] = 0



        # What about bases?
        if self.n_bases > 0:
            self.init_rel = get_param((self.n_bases, self.emb_dim))
            raise NotImplementedError
        else:
            # if self.model_nm.endswith('transe'):
            #     self.init_rel = get_param((self.num_rel, self.emb_dim))
            #     # self.init_rel = nn.Embedding(self.num_rel, self.emb_dim, padding_idx=0)
            # elif config['COMPGCNARGS']['OPN'] == 'rotate' or config['COMPGCNARGS']['QUAL_OPN'] == 'rotate':
            #     phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
            #     self.init_rel = nn.Parameter(torch.cat([
            #         torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
            #         torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            #     ], dim=0))
            # else:
            self.init_rel = get_param((self.num_rel * 2, self.emb_dim))
                # self.init_rel = nn.Embedding(self.num_rel * 2, self.emb_dim, padding_idx=0)
            # xavier_normal_(self.init_rel.weight)
            # self.init_rel.weight.data[0] = 0
            self.init_rel.data[0] = 0


        self.conv1 = EdgeGATConv(self.emb_dim, self.gcn_dim, self.num_rel, heads=4,
                                       config=config)
        self.conv2 = EdgeGATConv(self.gcn_dim * 4, self.emb_dim, self.num_rel, heads=4,
                                       config=config, concat=False)

        if self.conv1: self.conv1.to(self.device)
        if self.conv2: self.conv2.to(self.device)

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2,
                     quals=None, embed_qualifiers: bool = False, return_mask: bool = False):
        """
        :return:
        """
        r = self.init_rel

        if not self.triple_mode:
            # x, edge_index, edge_type, rel_embed, qual_ent, qual_rel
            x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r,
                              qualifier_ent=None,
                              qualifier_rel=None,
                              quals=self.quals)

            x = drop1(x)
            x, r = self.conv2(x=x, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r,
                              qualifier_ent=None,
                              qualifier_rel=None,
                              quals=self.quals)

        else:
            x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r)

            x = drop1(x)
            x, r = self.conv2(x=x, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r)

        x = drop2(x)

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        if embed_qualifiers:
            assert quals is not None, "Expected a tensor as quals."
            # flatten quals
            quals_ents = quals[:, 1::2].view(1,-1).squeeze(0)
            quals_rels = quals[:, 0::2].view(1,-1).squeeze(0)
            qual_obj_emb = torch.index_select(x, 0, quals_ents)
            # qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2])
            qual_rel_emb = torch.index_select(r, 0, quals_rels)
            qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
            qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
            if not return_mask:
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x
            else:
                # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
                # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
                # so we first initialize with False
                mask = torch.zeros((sub.shape[0], quals.shape[1] + 2)).bool().to(self.device)
                # and put True where qual entities and relations are actually padding index 0
                mask[:, 2:] = quals == 0
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x, mask

        return sub_emb, rel_emb, x
