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

    # inputs: inputs [n_batch, seq_len, 4]
    # outputs: embeddedGraph [n_batch, seq_len, dim_embedding]
    def forward(self, inputs, positions=None):
        posEncoding = 0
        if self.usePosEncoding:
            posEncoding = self.getPosEnco(positions)
        embedding = self.L_embedding(inputs)
        return embedding + posEncoding

class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__() #initialise nn.Modules
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_model, dim_k, dim_v):
        super().__init__() #initialise nn.Modules
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        #layers
        self.w_qs = nn.Linear(dim_model, n_head*dim_k, bias=False)
        self.w_ks = nn.Linear(dim_model, n_head*dim_k, bias=False)
        self.w_vs = nn.Linear(dim_model, n_head*dim_v, bias=False)
        self.fc = nn.Linear(n_head*dim_v, dim_model, bias=False)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    # inputs: [n_batch, seq_len, dim_embedding]
    def forward(self, q, k, v):
        n_batch, len_q, _ = q.size()
        len_k, len_v = k.size(1), v.size(1)
        residual = q
        q = self.layer_norm(q)
        # pass through linear layers
        q = self.w_qs(q) #[n_batch, seq_len, dim_embedding] => [n_batch, seq_len, n_head*dim_k]
        k = self.w_ks(k)
        v = self.w_vs(v)
        # Separate different heads: n_batch x seq_len x n_head x dim_k
        q = q.view(n_batch, len_q, self.n_head, self.dim_k)
        k = k.view(n_batch, len_k, self.n_head, self.dim_k)
        v = v.view(n_batch, len_v, self.n_head, self.dim_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q and k are multiplied
        norm_factor = 1/(self.dim_k**0.5)
        attn = norm_factor * torch.matmul(q, k.transpose(2, 3))
        # softmax and dropout applied
        attn = F.softmax(attn, dim=-1)
        # attn and v multiplied
        q = torch.matmul(attn, v)
        # dropout applied
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(n_batch, len_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q

class SelfMultiHeadAttention(MultiHeadAttention):
    def forward(self, input):
        return MultiHeadAttention.forward(self, input, input, input)

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.L_ff = nn.Sequential(
                    nn.Linear(dim_model, dim_hidden),
			        nn.ReLU(inplace = False),
                    nn.Linear(dim_hidden, dim_model))
        self.L_norm = nn.LayerNorm(dim_model, eps=1e-6)

    # inputs: [n_batch, seq_len, dim_embedding]
    def forward(self, inputs):
        residual = inputs
        x = self.L_ff(inputs) # [n_batch, seq_len, dim_model]
        x += residual
        x = self.L_norm(x)
        return x
