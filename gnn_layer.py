import torch

from utils_gcn import get_param, MessagePassing, ccorr, rotate
from utils_mytorch import compute_mask
from utils import masked_softmax
from torch_scatter import scatter_add

class CompQGCNConvLayer(MessagePassing):
    """ The important stuff. """

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 config=None):
        super(self.__class__, self).__init__()

        self.p = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))  # (100,200)
        self.w_in = get_param((in_channels, out_channels))  # (100,200)
        self.w_out = get_param((in_channels, out_channels))  # (100,200)
        self.w_rel = get_param((in_channels, out_channels))  # (100,200)

        if self.p['STATEMENT_LEN'] != 3:
            if self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'sum' or self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'attn':
                self.w_q = get_param((in_channels, in_channels))  # new for quals setup
            elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'concat':
                self.w_q = get_param((2 * in_channels, in_channels))  # need 2x size due to the concat operation

        self.loop_rel = get_param((1, in_channels))  # (1,100)
        self.loop_ent = get_param((1, in_channels))  # new

        self.drop = torch.nn.Dropout(self.p['COMPGCNARGS']['GCN_DROP'])
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p['COMPGCNARGS']['BIAS']: self.register_parameter('bias', Parameter(
            torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if self.p['STATEMENT_LEN'] != 3:
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                self.in_index_qual_ent, self.out_index_qual_ent = qualifier_ent[:, :num_edges], \
                                                                  qualifier_ent[:, num_edges:]

                self.in_index_qual_rel, self.out_index_qual_rel = qualifier_rel[:, :num_edges], \
                                                                  qualifier_rel[:, num_edges:]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                num_quals = quals.size(1) // 2
                self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
                self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
                self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        # @TODO: compute the norms for qual_rel and pass it along

        if self.p['STATEMENT_LEN'] != 3:

            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":

                if self.p['COMPGCNARGS']['SUBBATCH'] == 0:

                    in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type,
                                            rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                            ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                            qualifier_rel=self.in_index_qual_rel, qual_index=None)

                    loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                              rel_embed=rel_embed, edge_norm=None, mode='loop',
                                              ent_embed=None, qualifier_ent=None, qualifier_rel=None, qual_index=None)

                    out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type,
                                             rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                             ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                             qualifier_rel=self.out_index_qual_rel, qual_index=None)
                else:
                    loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                              rel_embed=rel_embed, edge_norm=None, mode='loop',
                                              ent_embed=None, qualifier_ent=None, qualifier_rel=None, qual_index=None)
                    in_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                    out_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                    num_batches = (num_edges // self.p['COMPGCNARGS']['SUBBATCH']) + 1
                    for i in range(num_edges)[::self.p['COMPGCNARGS']['SUBBATCH']]:
                        subbatch_in_index = self.in_index[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_in_type = self.in_type[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_in_norm = self.in_norm[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_in_qual_ent = self.in_index_qual_ent[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_in_qual_rel = self.in_index_qual_rel[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_out_index = self.out_index[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_out_type = self.out_type[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_out_norm = self.out_norm[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_out_qual_ent = self.out_index_qual_ent[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                        subbatch_out_qual_rel = self.out_index_qual_rel[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]

                        in_res += self.propagate('add', subbatch_in_index, x=x, edge_type=subbatch_in_type,
                                                 rel_embed=rel_embed, edge_norm=subbatch_in_norm, mode='in',
                                                 ent_embed=x, qualifier_ent=subbatch_in_qual_ent,
                                                 qualifier_rel=subbatch_in_qual_rel,
                                                 qual_index=None)
                        out_res += self.propagate('add', subbatch_out_index, x=x, edge_type=subbatch_out_type,
                                                  rel_embed=rel_embed, edge_norm=subbatch_out_norm, mode='out',
                                                  ent_embed=x, qualifier_ent=subbatch_out_qual_ent,
                                                  qualifier_rel=subbatch_out_qual_rel,
                                                  qual_index=None)
                    in_res = torch.div(in_res, float(num_batches))
                    out_res = torch.div(out_res, float(num_batches))

            # or the mode is sparse and we're doing a lot of new stuff
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                if self.p['COMPGCNARGS']['SUBBATCH'] == 0:
                    in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type,
                                            rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                            ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                            qualifier_rel=self.in_index_qual_rel,
                                            qual_index=self.quals_index_in)

                    loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                              rel_embed=rel_embed, edge_norm=None, mode='loop',
                                              ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                              qual_index=None)

                    out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type,
                                             rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                             ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                             qualifier_rel=self.out_index_qual_rel,
                                             qual_index=self.quals_index_out)
                else:
                    """
                    TODO: Slice the new quals matrix by batches but keep an eye on qual_index 
                        -> all from the main batch have to be there
                    """
                    print("Subbatching in the sparse mode is still in TODO")
                    raise NotImplementedError

        else:
            if self.p['COMPGCNARGS']['SUBBATCH'] == 0:
                in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type,
                                        rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                        ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                        qual_index=None)

                loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                          rel_embed=rel_embed, edge_norm=None, mode='loop',
                                          ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                          qual_index=None)

                out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type,
                                         rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                         ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                         qual_index=None)
            else:
                loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                          rel_embed=rel_embed, edge_norm=None, mode='loop',
                                          ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                          qual_index=None)
                in_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                out_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                num_batches = (num_edges // self.p['COMPGCNARGS']['SUBBATCH']) + 1
                for i in range(num_edges)[::self.p['COMPGCNARGS']['SUBBATCH']]:
                    subbatch_in_index = self.in_index[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                    subbatch_in_type = self.in_type[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                    subbatch_in_norm = self.in_norm[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                    subbatch_out_index = self.out_index[:, i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                    subbatch_out_type = self.out_type[i: i + self.p['COMPGCNARGS']['SUBBATCH']]
                    subbatch_out_norm = self.out_norm[i: i + self.p['COMPGCNARGS']['SUBBATCH']]

                    in_res += self.propagate('add', subbatch_in_index, x=x, edge_type=subbatch_in_type,
                                        rel_embed=rel_embed, edge_norm=subbatch_in_norm, mode='in',
                                        ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                             qual_index=None)
                    out_res += self.propagate('add', subbatch_out_index, x=x, edge_type=subbatch_out_type,
                                         rel_embed=rel_embed, edge_norm=subbatch_out_norm, mode='out',
                                         ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                              qual_index=None)
                in_res = torch.div(in_res, float(num_batches))
                out_res = torch.div(out_res, float(num_batches))



        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p['COMPGCNARGS']['BIAS']:
            out = out + self.bias
        out = self.bn(out)

        # Ignoring the self loop inserted, return.
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        if self.p['COMPGCNARGS']['OPN'] == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p['COMPGCNARGS']['OPN'] == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p['COMPGCNARGS']['OPN'] == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.p['COMPGCNARGS']['OPN'] == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return trans_embed

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

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):
        """
            Aggregates the qualifier matrix (3, edge_index, emb_dim)
        :param qualifier_emb:
        :param rel_part_emb:
        :param type:
        :param alpha
        :return:

        @TODO: Check for activation over qualifier_emb
        """
        # qualifier_emb = torch.mm(qualifier_emb.sum(axis=0), self.w_q)

        if self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'sum':
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                qualifier_emb = torch.mm(qualifier_emb.sum(axis=0), self.w_q)  # [N_EDGES / 2 x EMB_DIM]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                qualifier_emb = torch.einsum('ij,jk -> ik',
                                             self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                             self.w_q)

            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb      # [N_EDGES / 2 x EMB_DIM]
        elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'concat':
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                qualifier_emb = qualifier_emb.sum(axis=0)                  # [N_EDGES / 2 x EMB_DIM]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])

            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_q)                         # [N_EDGES / 2 x EMB_DIM]

        elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'attn':
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                qualifier_emb = torch.mm(qualifier_emb.sum(axis=0), self.w_q)  # [N_EDGES / 2 x EMB_DIM]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]), self.w_q)

            aggregated = self._self_attention_2d(rel_part_emb, qualifier_emb)
            return aggregated
        else:
            raise NotImplementedError

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):
        """
        :param rel_embed:
        :param qualifier_ent:
        :param qualifier_rel:
        :return:

        index select from embedding
        phi operation between qual_ent, qual_rel
        """

        # Step1 - embed them
        # qualifier_emb_rel = rel_embed[qualifier_rel.reshape(1, -1)]. \
        #     reshape(qualifier_rel.shape[0], qualifier_rel.shape[1], -1)
        #
        # qualifier_emb_ent = ent_embed[qualifier_ent.reshape(1, -1)]. \
        #     reshape(qualifier_ent.shape[0], qualifier_ent.shape[1], -1)

        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        rel_part_emb = rel_embed[edge_type]

        # TODO: check if rel_embed is dim is same
        # as ent_embed dim else use a linear layer

        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)

        # @TODO: pass it through a parameter layer
        # Pass it through a aggregate layer

        return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.p['COMPGCNARGS']['TRIPLE_QUAL_WEIGHT'],
                                        qual_index=qual_index)

    # return qualifier_emb
    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None):
        weight = getattr(self, 'w_{}'.format(mode))

        if self.p['STATEMENT_LEN'] != 3:
            # add code here
            if mode != 'loop':
                if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                    rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                             qualifier_rel, edge_type)  #
                elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                    rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                 qualifier_rel, edge_type, qual_index)
            else:
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.einsum('ij,jk->ik', xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[
            col]  # Norm parameter D^{-0.5} *

        return norm

    def _self_attention_2d(self, main, qual):
        """ Simple self attention """
        # @TODO: Add scaling factor
        scaling = float(main.shape[-1]) ** -0.5

        ct = torch.cat((main.unsqueeze(2), qual.unsqueeze(2)), dim=1)
        score = torch.bmm(ct, ct.transpose(2, 1)) * scaling
        mask = compute_mask(score, padding_idx=0)
        score = masked_softmax(score, mask)
        return torch.sum(torch.mm(score, ct), dim=1)

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges):
        """
        # TODO
        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :return: [1, N_EDGES]
        """
        #output = torch.zeros((num_edges, qual_embeddings.shape[1]), dtype=torch.float).to(self.device)
        # unq, unq_inv = torch.unique(qual_index, return_inverse=True)
        # np.add.at(out[:2, :], unq_inv, quals[:2, :])
        #ind = torch.LongTensor(qual_index)
        #output.index_add_(dim=0, index=qual_index, source=qual_embeddings)  # TODO check this magic carefully
        output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        # output = np.zeros((num_edges, qual_embeddings.shape[1]))
        # ind = qual_index.detach().cpu().numpy()
        # np.add.at(output, ind, qual_embeddings.detach().cpu().numpy())
        # output = torch.tensor(output, dtype=torch.float).to(self.device)
        return output

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)
