import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, SelfMultiHeadAttention

class EncoderL(nn.Module):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v):
        super().__init__() #initialise nn.Modules
        self.MMA = SelfMultiHeadAttention(n_head, dim_model, dim_k, dim_v)
        self.norm_1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.FF = SkipConnection(FeedForward(dim_model, dim_hidden))
        self.norm_2 = nn.LayerNorm(dim_model, eps=1e-6)

    # inputs: [batch_size, seq_len, embedding_size]
    # outputs: [batch_size, seq_len, embedding_size]
    def forward(self, inputs):
        # residual = inputs
        enc_output = self.MMA(inputs) #how is the multihead attention pulled together? mean? addition?
        # enc_output = enc_output + residual
        # enc_output = self.norm_1(enc_output)
        enc_output = self.FF(enc_output)
        # enc_output = self.norm_2(enc_output)
        return enc_output

class Compatability(nn.Module):
    def __init__(self, dim_model, dim_key):
        super().__init__() #initialise nn.Modules
        self.n_heads = 1
        self.W_query = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        self.W_key = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        # initialise parameters

    def forward(self, query, exchange, solution_indexes):
        ref = query.clone()
        batch_size, seq_len, input_dim = ref.size()
        n_query = query.size(1)
        refFlat = ref.contiguous().view(-1, input_dim)
        qflat = query.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, seq_len, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(refFlat, self.W_key).view(shp)
        compatibility_raw = torch.matmul(Q, K.transpose(2, 3))
        compatibility = torch.tanh(compatibility_raw[0]) * 10
        #mask infeasible solutions?
        #max pointless options
        pointless = torch.eye(seq_len).repeat(batch_size, 1, 1).to(compatibility.device)
        pointless = pointless * -1e8
        compatibility = compatibility + pointless
        # mask previous choice exchange
        # if exchange is not None:
        #     compatibility[torch.arange(batch_size), exchange[:,0], exchange[:,1]] = -np.inf
        #     compatibility[torch.arange(batch_size), exchange[:,1], exchange[:,0]] = -np.inf
        # print(compatibility[0:1])
        c = compatibility.view(batch_size, -1)
        logits = F.softmax(c, dim=-1)
        return logits

class TSP_improve(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        dim_model = int(model_config['dim_model'])
        dim_hidden = int(model_config['dim_hidden'])
        dim_k = int(model_config['dim_k'])
        dim_v = int(model_config['dim_v'])
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        self.L_embedder = GraphEmbedding(dim_model, usePosEncoding=True)
        self.L_encoder = nn.Sequential(*(EncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        # self.L_encoder = nn.ModuleList([EncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)])
        self.L_project_graph = nn.Linear(dim_model, dim_model, bias=False)
        self.L_project_node = nn.Linear(dim_model, dim_model, bias=False)
        self.L_decoder = Compatability(dim_model, dim_model)

    def forward(self, problems, solution_indexes, exchange, do_sample=True):
        seq_len = problems.size(1)
        x_embed = self.L_embedder(problems, solution_indexes)
        x_encode = self.L_encoder(x_embed) # torch.Size([2, 50, 128])
        # embd graph and node features
        max_pooling = x_encode.max(1)[0] # max Pooling
        graph_feature = self.L_project_graph(max_pooling)[:, None, :]
        node_feature = self.L_project_node(x_encode)
        # pass it through decoder
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([batch_size, seq_len, 128])
        logits = self.L_decoder(fusion, exchange, solution_indexes) # # att torch.Size([batch_size, seq_len^2])
        # select or sample
        pair_index = logits.multinomial(1) if do_sample else logits.max(-1)[1].view(-1,1)
        selected_likelihood = logits.gather(1, pair_index)
        col_selected = pair_index % seq_len
        row_selected = pair_index // seq_len
        pair = torch.cat((row_selected,col_selected),-1)
        return selected_likelihood, pair

class TSP_improveWrapped:
    def __init__(self, device, env, model_config):
        # set variables
        self.device = device
        self.env = env
        self.T = float(config['T'])
        self.actor = TSP_improve(model_config).to(self.device)

    # given a problem size and batch size will train the model
    def train(self, problems):
        problems = problems.to(self.device)
        batch_size, problem_size, _ = problems
        solutions = self.env.getInitialSolution(batch_size, problem_size).to(self.device)
        #tracking information about the sequence
        best_so_far = self.env.evaluate(problems, solutions) #[batch_size]
        initial_reward = best_so_far.clone() #INFO
        action_history = [] #INFO
        reward_history = [] #INFO
        exchange = None
        # run through the model
        t = 0
        self.M_actor.train()
        while t < self.T:
            likelihood, exchange = self.actor(problems, solutions, exchange) #
            solutions = self.env.nextState(solutions, exchange)
            #calc reward_history
            cost = self.env.evaluate(problems, solutions)
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            reward_history.append(cost)
            action_history.append(exchange)
            reward = best_for_now - best_so_far
            # reward = reward - best_so_far  #low is good
            best_so_far = best_for_now
            t = t + 1
            log_lh = torch.log(likelihood).to(reward.device)
            # log_lh[log_lh < -1000] = 0.
            print("reward", reward[0:10])
            #USE policybased thingy
        #return reward, probability

    # given a batch size and problem size will test the model
    def test(self, batch_size, problem_size, data_loc=None):
        self.M_actor.eval()
        # generate problems and initial solutions
        problems = problems.to(self.device)
        batch_size, problem_size, _ = problems
        solutions = self.env.getInitialSolution(batch_size, problem_size).to(self.device)
        #tracking information about the sequence
        R = self.env.evaluate(problems, solutions) #[batch_size]
        # run through the model
        exchange = None
        t = 0
        self.M_actor.eval()
        while t < self.T:
            likelihood, exchange = self.M_actor(problems, solutions, exchange)
            solutions = self.env.nextState(solutions, exchange)
            # solutions = self.env.nextState(solutions, exchange.detach())
            reward = self.env.evaluate(problems, solutions)
            R = torch.cat((R[None, :], reward[None, :]), 0).min(0)[0]
            t = t + 1
        return R
