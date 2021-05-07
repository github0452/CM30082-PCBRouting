import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GraphEmbedding(nn.Module):
    def __init__(self, dim_embedding, usePosEncoding=False):
        super().__init__() #initialise nn.Modules
        self.usePosEncoding = usePosEncoding
        self.dim_model = dim_embedding
        self.L_embedding = nn.Linear(4, dim_embedding, bias=False)
        self.setParam(dim_embedding)

    def setParam(self, dim_embedding):
        param = self.L_embedding.weight
        param.data.uniform_(-(1. / math.sqrt(param.size(0))), 1. / math.sqrt(param.size(0)))

    def getPosEnco(self, inputs):
        n_batch, prob_s = inputs.size()
        ####### need to change for depot
        ''' Init the sinusoid position encoding table '''
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / self.dim_model) for j in range(self.dim_model)]
            for pos in range(1,prob_s+1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor).to(inputs.device)
        position_enc = position_enc.expand(n_batch, prob_s, self.dim_model) #add dim=0 and repeat to batch size
        # get index according to the solutions
        index = [torch.nonzero(inputs.long() == i)[:,1][:,None].expand(n_batch, self.dim_model)
                 for i in inputs[0].sort()[0]]
        index = torch.stack(index, 1)
        # return
        pos_enc = torch.gather(position_enc, 1, index).clone()
        return pos_enc

    # inputs: inputs [n_batch, n_node, 4]
    # outputs: embeddedGraph [n_batch, n_node, dim_embedding]
    def forward(self, inputs, positions=None):
        posEncoding = 0
        if self.usePosEncoding:
            posEncoding = self.getPosEnco(positions)
        embedding = self.L_embedding(inputs)
        return embedding + posEncoding#[:embedding.size(0)]

class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__() #initialise nn.Modules
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q_input, dim_k, dim_v, n_head, dim_k_input=None, dim_v_input=None):
        super().__init__() #initialise nn.Modules
        self.dim_q_input = dim_q_input
        self.dim_k_input = dim_k_input if dim_k_input else dim_q_input
        self.dim_v_input = dim_v_input if dim_v_input else dim_q_input
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_head = n_head
        #layers
        self.L_proj_q = nn.Linear(self.dim_q_input, n_head*dim_k, bias=False)
        self.L_proj_k = nn.Linear(self.dim_k_input, n_head*dim_k, bias=False)
        self.L_proj_v = nn.Linear(self.dim_v_input, n_head*dim_v, bias=False)
        self.fc = nn.Linear(n_head*dim_v, self.dim_v_input, bias=False)

    # inputs: [n_batch, n_node, dim_embedding]
    def forward(self, q, k=None, v=None, mask=None):
        k = q if k is None else k
        v = k if v is None else v
        n_batch, len_q, _ = q.size()
        # pass through linear layers and seperate different heads: n_batch x len_x x n_head x dim_x
        qs = torch.stack(torch.chunk(self.L_proj_q(q), self.n_head, dim=-1), dim=1)  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(torch.chunk(self.L_proj_k(k), self.n_head, dim=-1), dim=1)  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(torch.chunk(self.L_proj_v(v), self.n_head, dim=-1), dim=1)  # [batch_size, n_head, n_node, v_dim]
        # calculate compatability/attention
        norm_factor = 1/(self.dim_k**0.5)
        attn = norm_factor * torch.matmul(qs, ks.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))
        # calculate final multi-head query
        q = torch.matmul(F.softmax(attn, dim=-1), vs)
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(n_batch, len_q, -1)
        q = self.fc(q)
        return q

class FeedForward(nn.Sequential):
    def __init__(self, dim_model, dim_hidden, dim_out=None):
        dim_out = dim_model if dim_out is None else dim_out
        super().__init__(
            nn.Linear(dim_model, dim_hidden),
	        nn.ReLU(inplace = False),
            nn.Linear(dim_hidden, dim_out)
        ) #initialise nn.Modules
        # inputs: [n_batch, n_node, dim_model]
        # outputs: [n_batch, n_node, dim_model]

class TransformerEncoderL(nn.Sequential):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, batch_norm=True, momentum=0.1):
        super().__init__() #initialise nn.Modules
        self.MMA = SkipConnection(MultiHeadAttention(dim_model, dim_k, dim_v, n_head))
        self.FF = SkipConnection(FeedForward(dim_model, dim_hidden))
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_1 = Normalization(dim_model, momentum)
            self.norm_2 = Normalization(dim_model, momentum)

    # inputs: [batch_size, n_node, embedding_size]
    # outputs: [batch_size, n_node, embedding_size]
    def forward(self, inputs):
        enc_output = self.MMA(inputs) #how is the multihead attention pulled together? mean? addition?
        if self.batch_norm:
            enc_output = self.norm_1(enc_output)
        enc_output = self.FF(enc_output)
        if self.batch_norm:
            enc_output = self.norm_2(enc_output)
        return enc_output

class Normalization(nn.Module):
    def __init__(self, embed_dim, momentum=0.1, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, momentum=momentum, affine=True)
        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input
