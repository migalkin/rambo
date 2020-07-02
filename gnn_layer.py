import torch
import numpy as np
import torch.nn.functional as F

from utils_gcn import get_param, MessagePassing, ccorr, rotate, softmax
from utils_mytorch import compute_mask
from utils import masked_softmax
from torch_scatter import scatter_add, scatter_mean


try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

class CompQGCNConvLayer(MessagePassing):
    """ The important stuff.
        The layer derives from message passing base class which automates the process
        of sending messages and updating the representation. One needs to write two function namely:
        message and update. Rest is taken care by torch-geometric-like framework
     """

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 config=None):
        super(self.__class__, self).__init__()

        self.p = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))  # (100,200) self loop parameters
        self.w_in = get_param((in_channels, out_channels))  # (100,200) regular relation parameters
        self.w_out = get_param((in_channels, out_channels))  # (100,200) inverse relation parameters
        self.w_rel = get_param((in_channels, out_channels))  # (100,200)

        if self.p['STATEMENT_LEN'] != 3: # which means there are qualifiers are present
            if self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'sum' or self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'mul' or \
                    self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'attn':
                self.w_q = get_param((in_channels, in_channels))  # new for quals setup
            elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'concat':
                self.w_q = get_param((2 * in_channels, in_channels))  # need 2x size due to the concat operation

        self.loop_rel = get_param((1, in_channels))  # (1,100) parameters for self-loop
        self.loop_ent = get_param((1, in_channels))  #

        self.drop = torch.nn.Dropout(self.p['COMPGCNARGS']['GCN_DROP'])
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p['COMPGCNARGS']['ATTENTION']:
            assert self.p['COMPGCNARGS']['GCN_DIM'] == self.p['EMBEDDING_DIM'], "Current attn implementation requires those tto be identical"
            assert self.p['EMBEDDING_DIM'] % self.p['COMPGCNARGS']['ATTENTION_HEADS'] == 0, "should be divisible"
            self.heads = self.p['COMPGCNARGS']['ATTENTION_HEADS']
            self.attn_dim = self.out_channels // self.heads
            self.negative_slope = self.p['COMPGCNARGS']['ATTENTION_SLOPE']
            self.attn_drop = self.p['COMPGCNARGS']['ATTENTION_DROP']
            self.att = get_param((1, self.heads, 2 * self.attn_dim))

        if self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'attn':
            assert self.p['COMPGCNARGS']['GCN_DIM'] == self.p[
                'EMBEDDING_DIM'], "Current attn implementation requires those tto be identical"
            assert self.p['EMBEDDING_DIM'] % self.p['COMPGCNARGS']['ATTENTION_HEADS'] == 0, "should be divisible"
            self.att_qual = get_param((1, self.heads, 2 * self.attn_dim))
            if not self.p['COMPGCNARGS']['ATTENTION']:
                self.heads = self.p['COMPGCNARGS']['ATTENTION_HEADS']
                self.attn_dim = self.out_channels // self.heads
                self.negative_slope = self.p['COMPGCNARGS']['ATTENTION_SLOPE']
                self.attn_drop = self.p['COMPGCNARGS']['ATTENTION_DROP']

        if self.p['COMPGCNARGS']['BIAS']: self.register_parameter('bias', Parameter(
            torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None):
        """

        See end of doc string for explaining.

        :param x: all entities*dim_of_entities (for jf17k -> 28646*200)
        :param edge_index: COO matrix (2 list each having nodes with index
        [1,2,3,4,5]
        [3,4,2,5,4]

        Here node 1 and node 3 are connected with edge.
        And the type of edge can be found using edge_type.

        Note that there are twice the number of edges as each edge is also reversed.
        )
        :param edge_type: The type of edge connecting the COO matrix
        :param rel_embed: 2 Times Total relation * emb_dim (200 in our case and 2 Times because of inverse relations)
        :param qualifier_ent:
        :param qualifier_rel:
        :param quals: Another sparse matrix

        where
            quals[0] --> qualifier relations type
            quals[1] --> qualifier entity
            quals[2] --> index of the original COO matrix that states for which edge this qualifier exists ()


        For argument sake if a knowledge graph has following statements

        [e1,p1,e4,qr1,qe1,qr2,qe2]
        [e1,p1,e2,qr1,qe1,qr2,qe3]
        [e1,p2,e3,qr3,qe3,qr2,qe2]
        [e1,p2,e5,qr1,qe1]
        [e2,p1,e4]
        [e4,p3,e3,qr4,qe1,qr2,qe4]
        [e1,p1,e5]
                                                 (incoming)         (outgoing)
                                            <----(regular)------><---(inverse)------->
        Edge index would be             :   [e1,e1,e1,e1,e2,e4,e1,e4,e2,e3,e5,e4,e3,e5]
                                            [e4,e2,e3,e5,e4,e3,e5,e1,e1,e1,e1,e2,e4,e1]

        Edge Type would be              :   [p1,p1,p2,p2,p1,p3,p1,p1_inv,p1_inv,p2_inv,p2_inv,p1_inv,p3_inv,p1_inv]

                                            <-------on incoming-----------------><---------on outgoing-------------->
        quals would be                  :   [qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2,qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2]
                                            [qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4,qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4]
                                            [0,0,1,1,2,2,3,5,5,0,0,1,1,2,2,3,5,5]
                                            <--on incoming---><--outgoing------->

        Note that qr1,qr2... and qe1, qe2, ... all belong to the same space
        :return:
        """
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0) # adding self loop params
        num_edges = edge_index.size(1) // 2 # as other half is repeated
        num_ent = x.size(0) # entity embedding matrix



        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] # as first is regular index, second half is inverse rel
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if self.p['STATEMENT_LEN'] != 3: # qualifier information is present in the graph
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full": # not used any more.
                self.in_index_qual_ent, self.out_index_qual_ent = qualifier_ent[:, :num_edges], \
                                                                  qualifier_ent[:, num_edges:]

                self.in_index_qual_rel, self.out_index_qual_rel = qualifier_rel[:, :num_edges], \
                                                                  qualifier_rel[:, num_edges:]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                num_quals = quals.size(1) // 2
                self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:] # first half if regular and second is repeated
                self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:] # first half if regular amd second is the repeated
                self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]


        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations
            
            
        '''
        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device) # add self loop edges
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here.

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
                                            qual_index=self.quals_index_in,
                                            source_index=self.in_index[0])

                    loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                              rel_embed=rel_embed, edge_norm=None, mode='loop',
                                              ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                              qual_index=None, source_index=None)

                    out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type,
                                             rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                             ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                             qualifier_rel=self.out_index_qual_rel,
                                             qual_index=self.quals_index_out,
                                             source_index=self.out_index[0])
                else:
                    """
                    TODO: Slice the new quals matrix by batches but keep an eye on qual_index 
                        -> all from the main batch have to be there
                    """
                    # print("Subbatching in the sparse mode is still in TODO")
                    # raise NotImplementedError
                    loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                              rel_embed=rel_embed, edge_norm=None, mode='loop',
                                              ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                              qual_index=None)
                    # i = 0
                    in_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                    out_res = torch.zeros((x.shape[0], self.out_channels)).to(self.device)
                    num_batches = (num_edges // self.p['COMPGCNARGS']['SUBBATCH']) + 1
                    all_edges = np.random.permutation(np.arange(num_edges))
                    for i in range(num_edges)[::self.p['COMPGCNARGS']['SUBBATCH']]:
                        # sample edges
                        edges = torch.tensor(all_edges[i: i+self.p['COMPGCNARGS']['SUBBATCH']], device=self.device)
                        # edges = torch.tensor(np.sort(np.random.choice(num_edges, self.p['COMPGCNARGS']['SUBBATCH'], replace=False)), device=self.device)
                        subbatch_in_index = self.in_index[:, edges]
                        subbatch_in_type = self.in_type[edges]
                        subbatch_in_norm = self.in_norm[edges]
                        # find if there are any quals
                        # the following magic has been ripped off from SO https://stackoverflow.com/questions/56176439/pytorch-argsort-ordered-with-duplicate-elements-in-the-tensor
                        # subbatch_in_quals = torch.nonzero(self.quals_index_in[..., None] == edges)[:, 0]
                        # subbatch_in_q_index = torch.nonzero(self.quals_index_in[..., None] == edges)[:, 1]
                        # the above yields CUDA OOM, worse solution below
                        subbatch_q_index = torch.tensor([k for k, x in enumerate(edges) if x in self.quals_index_in for _ in range(self.quals_index_in.eq(x).sum().item())], device=self.device)
                        #subbatch_in_quals = torch.tensor([k for k, x in enumerate(self.quals_index_in) if x in edges], device=self.device)
                        # subbatch_in_quals = torch.tensor([l.item() for k in edges if k in self.quals_index_in for l in torch.nonzero(self.quals_index_in == k)])
                        subbatch_in_quals = torch.cat([torch.nonzero(self.quals_index_in == k) for k in edges]).squeeze(1)
                        subbatch_in_q_ents = self.in_index_qual_ent[subbatch_in_quals]
                        subbatch_in_q_rels = self.in_index_qual_rel[subbatch_in_quals]
                        # temp1 = in_edges_with_quals[..., None] == edges
                        # subbatch_in_q_index = torch.nonzero(temp1.t())[:, 0]

                        subbatch_out_index = self.out_index[:, edges]
                        subbatch_out_type = self.out_type[edges]
                        subbatch_out_norm = self.out_norm[edges]
                        # find if there are any quals
                        # out_edges_with_quals = torch.tensor(np.sort(np.intersect1d(self.quals_index_out, edges)), device=self.device)
                        # subbatch_out_quals = torch.nonzero(self.quals_index_out[..., None] == edges)[:, 0]
                        # subbatch_out_q_index = torch.nonzero(self.quals_index_out[..., None] == edges)[:, 1]
                        subbatch_out_q_ents = self.out_index_qual_ent[subbatch_in_quals]
                        subbatch_out_q_rels = self.out_index_qual_rel[subbatch_in_quals]
                        # temp2 = out_edges_with_quals[:, None] == edges
                        # subbatch_out_q_index = torch.nonzero(temp2.t())[:, 0]
                        # propagate
                        in_res += self.propagate('add', subbatch_in_index, x=x, edge_type=subbatch_in_type,
                                                rel_embed=rel_embed, edge_norm=subbatch_in_norm, mode='in',
                                                ent_embed=x, qualifier_ent=subbatch_in_q_ents,
                                                qualifier_rel=subbatch_in_q_rels,
                                                qual_index=subbatch_q_index)
                        out_res += self.propagate('add', subbatch_out_index, x=x, edge_type=subbatch_out_type,
                                                 rel_embed=rel_embed, edge_norm=subbatch_out_norm, mode='out',
                                                 ent_embed=x, qualifier_ent=subbatch_out_q_ents,
                                                 qualifier_rel=subbatch_out_q_rels,
                                                 qual_index=subbatch_q_index)
                        # iterate
                        # i += self.p['COMPGCNARGS']['SUBBATCH']
                    # avg in and out
                    in_res = torch.div(in_res, float(num_batches))
                    out_res = torch.div(out_res, float(num_batches))

        else:
            if self.p['COMPGCNARGS']['SUBBATCH'] == 0:
                in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type,
                                        rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                        ent_embed=x, qualifier_ent=None, qualifier_rel=None,
                                        qual_index=None, source_index=self.in_index[0])

                loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                          rel_embed=rel_embed, edge_norm=None, mode='loop',
                                          ent_embed=x, qualifier_ent=None, qualifier_rel=None,
                                          qual_index=None, source_index=None)

                out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type,
                                         rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                         ent_embed=x, qualifier_ent=None, qualifier_rel=None,
                                         qual_index=None, source_index=self.out_index[0])
            else:
                loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                          rel_embed=rel_embed, edge_norm=None, mode='loop',
                                          ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                          qual_index=None, source_index=None)
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
                                             qual_index=None, source_index=None)
                    out_res += self.propagate('add', subbatch_out_index, x=x, edge_type=subbatch_out_type,
                                         rel_embed=rel_embed, edge_norm=subbatch_out_norm, mode='out',
                                         ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                              qual_index=None, source_index=None)
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
            In qualifier_aggregate method following steps are performed

            qualifier_emb looks like -
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            rel_part_emb       :   [qq,ww,ee,rr,tt, .....]                      (here qq, ww, ee .. are of 200 dim)

            Note that rel_part_emb for jf17k would be around 61k*200

            Step1 : Pass the qualifier_emb to self.coalesce_quals and multiply the returned output with a weight.
            qualifier_emb   : [aa,bb,cc,dd,ee, ...... ]                 (here aa, bb, cc are of 200 dim each)
            Note that now qualifier_emb has the same shape as rel_part_emb around 61k*200

            Step2 : Combine the updated qualifier_emb (see Step1) with rel_part_emb based on defined aggregation strategy.



            Aggregates the qualifier matrix (3, edge_index, emb_dim)
        :param qualifier_emb:
        :param rel_part_emb:
        :param type:
        :param alpha
        :return:

        self.coalesce_quals    returns   :  [q+a+b+d,w+c+e+g,e'+f,......]        (here each element in the list is of 200 dim)


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

        elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'mul':
            if self.p['COMPGCNARGS']['QUAL_REPR'] == "full":
                qualifier_emb = torch.mm(qualifier_emb.sum(axis=0), self.w_q)  # [N_EDGES / 2 x EMB_DIM]
            elif self.p['COMPGCNARGS']['QUAL_REPR'] == "sparse":
                qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1), self.w_q)

            return rel_part_emb * qualifier_emb

        elif self.p['COMPGCNARGS']['QUAL_AGGREGATE'] == 'attn':
            # only for sparse mode
            expanded_rels = torch.index_select(rel_part_emb, 0, qual_index)  # Nquals x D
            expanded_rels = expanded_rels.view(-1, self.heads, self.attn_dim)  # Nquals x heads x h_dim
            qualifier_emb = torch.mm(qualifier_emb, self.w_q).view(-1, self.heads, self.attn_dim)  # Nquals x heads x h_dim

            alpha_r = torch.einsum('bij,kij -> bi', [torch.cat([expanded_rels, qualifier_emb], dim=-1), self.att_qual])
            alpha_r = F.leaky_relu(alpha_r, self.negative_slope)  # Nquals x heads
            alpha_r = softmax(alpha_r, qual_index, rel_part_emb.size(0))  # Nquals x heads
            alpha_r = F.dropout(alpha_r, p=self.attn_drop)  # Nquals x heads
            expanded_rels = (expanded_rels * alpha_r.view(-1, self.heads, 1)).view(-1, self.heads * self.attn_dim)  # Nquals x D
            single_rels = scatter_add(expanded_rels, qual_index, dim=0, dim_size=rel_part_emb.size(0), fill_value=0)  # Nedges x D
            copy_mask = single_rels.sum(dim=1) != 0.0
            rel_part_emb[copy_mask] = single_rels[copy_mask]  # Nedges x D
            return rel_part_emb
        else:
            raise NotImplementedError

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):
        """
        The update_rel_emb_with_qualifier method performs following functions:

        Input is the secondary COO matrix (QE (qualifier entity), QR (qualifier relation), edge index (Connection to the primary COO))

        Step1 : Embed all the input
            Step1a : Embed the qualifier entity via ent_embed (So QE shape is 33k,1 -> 33k,200)
            Step1b : Embed the qualifier relation via rel_embed (So QR shape is 33k,1 -> 33k,200)
            Step1c : Embed the main statement edge_type via rel_embed (So edge_type shape is 61k,1 -> 61k,200)

        Step2 : Combine qualifier entity emb and qualifier relation emb to create qualifier emb (See self.qual_transform).
            This is generally just summing up. But can be more any pair-wise function that returns one vector for a (qe,qr) vector

        Step3 : Update the edge_type embedding with qualifier information. This uses scatter_add/scatter_mean.


        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)

        After:
            edge_type          :   [q+(a+b+d),w+(c+e+g),e'+f,......]        (here each element in the list is of 200 dim)


        :param ent_embed: essentially x (28k*200 in case of Jf17k)
        :param rel_embed: essentially relation embedding matrix

        For secondary COO matrix (QE, QR, edge index)
        :param qualifier_ent:  QE
        :param qualifier_rel: QR
        edge_type:
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

        # Step 1: embed everything
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
    def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, source_index=None):
        """

        The message method performs following functions

        Step1 : get updated relation representation (rel_embed) [edge_type] by aggregating qualifier information (self.update_rel_emb_with_qualifier).
        Step2 : Obtain edge message by transforming the node embedding with updated relation embedding (self.rel_transform).
        Step3 : Multiply edge embeddings (transform) by weight
        Step4 : Return the messages. They will be sent to subjects (1st line in the edge index COO)
        Over here the node embedding [the first list in COO matrix] is representing the message which will be sent on each edge


        More information about updating relation representation please refer to self.update_rel_emb_with_qualifier

        :param x_j: objects of the statements (2nd line in the COO)
        :param x_i: subjects of the statements (1st line in the COO)
        :param edge_type: relation types
        :param rel_embed: embedding matrix of all relations
        :param edge_norm:
        :param mode: in (direct) / out (inverse) / loop
        :param ent_embed: embedding matrix of all entities
        :param qualifier_ent:
        :param qualifier_rel:
        :param qual_index:
        :param source_index:
        :return:
        """
        weight = getattr(self, 'w_{}'.format(mode))

        if self.p['STATEMENT_LEN'] != 3:
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

        if self.p['COMPGCNARGS']['ATTENTION'] and mode != 'loop':
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)
            # if LazyTensor is not None:
            #     ns, att = LazyTensor(torch.cat([x_i, out], dim=-1).unsqueeze(-3)), LazyTensor(self.att.unsqueeze(-3))
            #     alpha = (ns * att).sum(dim=-1, backend='auto').sum(dim=1).squeeze(-1)
            # else:
            #     alpha = (torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att]))
            alpha = torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att])
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            return out * alpha.view(-1, self.heads, 1)
        else:
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, mode):
        if self.p['COMPGCNARGS']['ATTENTION'] and mode != 'loop':
            aggr_out = aggr_out.view(-1, self.heads * self.attn_dim)

        return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.

        Yet another torch scatter functionality. See coalesce_quals for a rough idea.

        row         :      [1,1,2,3,3,4,4,4,4, .....]        (about 61k for Jf17k)
        edge_weight :      [1,1,1,1,1,1,1,1,1,  ....] (same as row. So about 61k for Jf17k)
        deg         :      [2,1,2,4,.....]            (same as num_ent about 28k in case of Jf17k)

        :param edge_index:
        :param num_ent:
        :return:
        """
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

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

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

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)
