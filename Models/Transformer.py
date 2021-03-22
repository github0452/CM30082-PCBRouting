import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, SelfMultiHeadAttention

class EncoderL(nn.Module):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.MMA = SkipConnection(SelfMultiHeadAttention(n_head, dim_model, dim_k, dim_v))
        self.norm_1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.FF = SkipConnection(FeedForward(dim_model, dim_hidden))
        self.norm_2 = nn.LayerNorm(dim_model, eps=1e-6)

    # inputs: [batch_size, n_node, embedding_size]
    # outputs: [batch_size, n_node, embedding_size]
    def forward(self, inputs):
        enc_output = self.MMA(inputs) #how is the multihead attention pulled together? mean? addition?
        enc_output = self.norm_1(enc_output)
        enc_output = self.FF(enc_output)
        enc_output = self.norm_2(enc_output)
        return enc_output

class Attention(nn.Module):
    """
    Multi-head attention
    Input:
      q: [batch_size, n_node, hidden_dim]
      k, v: q if None
    Output:
      att: [n_node, hidden_dim]
    """
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        if v is None:
            v = q

        bsz, n_node, hidden_dim = q.size()

        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, v_dim]

        normalizer = self.k_dim**0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))
        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node,
                                          self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att

class Transformer(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        dim_model = int(model_config['dim_model'])
        dim_hidden = int(model_config['dim_hidden'])
        dim_k = int(model_config['dim_k'])
        dim_v = int(model_config['dim_v'])
        self.L_embedder = GraphEmbedding(dim_model, usePosEncoding=False)
        self.L_encoder = nn.Sequential(*(EncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_graph_context = nn.Linear(dim_model, dim_model, bias=False)
        self.L_node_context = nn.Linear(dim_model, dim_model, bias=False)
        self.L_decoder_attention = Attention(dim_model * 3, dim_k, dim_v, n_head, k_hidden_dim=dim_model, v_hidden_dim=dim_model)
        # self.L_decoder_multi_head_attention = MultiHeadAttention(n_head, dim_model, dim_k, dim_v)
        self.W_v_f = nn.Parameter(torch.FloatTensor(size=[1, 1, dim_model]).uniform_())
        self.W_v_l = nn.Parameter(torch.FloatTensor(size=[1, 1, dim_model]).uniform_())
        # self.L1 = nn.Linear(dim_model, dim_model, bias=False)
        # self.L2 = nn.Linear(dim_model, dim_model, bias=False)
        # self.L3 = nn.Linear(dim_model*3, dim_model, bias=False)

    # problem points ->
    def forward(self, problems, sampling=True):
        n_batch, n_node, _ = problems.size()
        #ENCODE PROBLEM - invariant so must be problem
        embedded_problems = self.L_embedder(problems, None)
        node_context = self.L_encoder(embedded_problems) # [n_batch, n_node, dim_model]
        graph_context = self.L_graph_context(node_context.mean(dim=1, keepdim=True))
        #sequence information
        first = self.W_v_f.repeat(n_batch, 1, 1)
        last = self.W_v_l.repeat(n_batch, 1, 1)
        mask = torch.zeros([n_batch, n_node], device=problems.device).bool()
        # embd_graph & enc_states, [n_batch, n_node, dim_model]
        action_list = None #action_list: ([step, n_batch])
        action_probs_list = [] # probability of each action taken, action_probs_list: (step x [n_batch])
        while (action_list is None or action_list.size(0) != problems.size(1)):
            #DECODER STUFF
            context_query = torch.cat([graph_context, last, first], dim=-1)
            # if action_list is None:
            #     context_query = torch.cat((graph_context, \
            #         self.W_v_f.repeat(n_batch, 1, 1), \
            #         self.W_v_l.repeat(n_batch, 1, 1)), \
            #         dim=1)
            # else:
            #     context_query = torch.cat((graph_context, node_context[:,1,:], node_context[:,len(action_list),:]), dim=1)
            # context_query = context_query.unsqueeze(dim=1).repeat(1, n_node, 1)
            q = self.L_decoder_attention(context_query, node_context, node_context, mask=mask)
            # key, value = self.L1(node_context), self.L2(node_context)
            # query = self.L3(context_query)
            # #bmm query and key
            # compatability = torch.matmul(query, key.transpose(1, 2)/math.sqrt(query.size(-1)))
            # #softmax result. multiple value and compatability
            # embedded_logits = torch.matmul(torch.softmax(compatability, dim=-1), value)
            #another decoder layer - SINGLE ATTENTION HEAD
            k = self.L_node_context(node_context)
            logits = torch.matmul(q, k.transpose(-2, -1)/math.sqrt(q.size(-1)))
            logits = logits.squeeze(dim=1)
            logits = logits.tanh()*10 #[n_batch, n_node]
            logits = logits.masked_fill(mask, float('-inf'))
            #mask stuff
            if action_list is not None: # mask the previous actions
                logits[[[i for i in range(n_batch)], action_list]] = -np.inf
            # pick an action
            if sampling:
                probs = F.softmax(logits, dim=1) #soft max the probabilities
                actions = probs.multinomial(1).squeeze(1) # sample an index for each problem in the batch, actions: torch.Size([100])
            else:
                actions = logits.argmax(dim = 1)
                probs = Variable(torch.zeros(logits.size()), requires_grad = True).to(logits.device)
                probs[:, actions] = 1
            # add it to various lists
            if action_list is None:
                action_list = actions.unsqueeze(dim=0)
            else:
                action_list = torch.cat((action_list, actions.unsqueeze(dim=0)), dim=0)
            action_probs_list.append(probs[[x for x in range(len(probs))], actions.data])
            # update for next loop
            mask = mask.scatter(1, actions.unsqueeze(dim=-1), True)
            visit_idx = actions.unsqueeze(-1).repeat(1, 1, node_context.size(-1))
            last = torch.gather(node_context, 1, visit_idx).transpose(0, 1)
            if len(action_probs_list) == 1:
                first = last
        return action_probs_list, action_list

class TransformerWrapped:
        # creates the everything around the networks
        def __init__(self, env, trainer, model_config, optimizer_config): # hyperparameteres
            self.env = env
            self.trainer = trainer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.actor = Transformer(model_config).to(self.device)
            self.trainer.passIntoParameters(self.actor.parameters(), optimizer_config)

        # given a problem size and batch size will train the model
        def train(self, n_batch, p_size, path=None):
            problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
            # run through the model
            self.actor.train()
            action_probs_list, action_list = self.actor(problems, sampling=True) #action_probs_list (n_node x [n_batch])
            # calculate reward and probability
            reward = self.env.evaluate(problems.detach(), action_list.transpose(0, 1)).to(self.device)
            probs = action_probs_list[0]
            for i in range(1, len(action_probs_list)):
                probs = probs * action_probs_list[i]
            # use this to train
            R, loss = self.trainer.train(problems, reward, probs, self.actor, self.env)
            return R, loss

        # given a batch size and problem size will test the model
        def test(self, n_batch, p_size, path=None):
            problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
            # run through model
            self.actor.eval()
            problems = problems.to(self.device)
            action_probs_list, action_list = self.actor(problems, sampling=True)
            R = self.env.evaluate(problems, action_list.transpose(0, 1)).to(self.device)
            return R

        def save(self):
            model_dict = {}
            model_dict['actor_model_state_dict'] = self.actor.actor.state_dict()
            model_dict.update(self.trainer.save())
            return model_dict

        def load(self, checkpoint):
            self.actor.actor.load_state_dict(checkpoint['actor_model_state_dict'])
            self.trainer.load(checkpoint)

# generate problems
# problems = torch.FloatTensor(self.env.genProblems(batch_size, problem_size))
if __name__ == "__main__":
    #__init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout):
    test = Transformer(4, 2, 128, 256, 64, 64)
    testArray = torch.zeros((100, 5, 4))
    testArray2 = torch.zeros((100, 3, 4))
    print(testArray.size())
    testArray = test(testArray, testArray2)
    print(testArray.size())
