import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, SelfMultiHeadAttention

class EncoderL(nn.Module):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v):
        super().__init__() #initialise nn.Modules
        self.MMA = SkipConnection(SelfMultiHeadAttention(n_head, dim_model, dim_k, dim_v))
        self.norm_1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.FF = SkipConnection(FeedForward(dim_model, dim_hidden))
        self.norm_2 = nn.LayerNorm(dim_model, eps=1e-6)

    # inputs: [batch_size, seq_len, embedding_size]
    # outputs: [batch_size, seq_len, embedding_size]
    def forward(self, inputs):
        enc_output = self.MMA(inputs) #how is the multihead attention pulled together? mean? addition?
        enc_output = self.norm_1(enc_output)
        enc_output = self.FF(enc_output)
        enc_output = self.norm_2(enc_output)
        return enc_output

class Compatability(nn.Module):
    def __init__(self, dim_model, dim_key):
        super().__init__() #initialise nn.Modules
        self.n_heads = 1
        self.W_query = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        self.W_key = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        self.init_params()
        # initialise parameters

    def init_params(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, query, exchange, solution_indexes):
        ref = query.clone()
        batch_size, seq_len, input_dim = ref.size()
        n_query = query.size(1)
        refFlat = ref.contiguous().view(-1, input_dim)
        qflat = query.contiguous().view(-1, input_dim)
        # print(refFlat[0])
        # print(qflat[0])
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, seq_len, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(refFlat, self.W_key).view(shp)
        # print(Q[0])
        # print(K[0])
        compatability_raw = torch.matmul(Q, K.transpose(2, 3)).squeeze(0)
        # print(compatability_raw.size())
        # print("compatability_raw")
        # print(compatability_raw[0])
        # compatability = torch.tanh(compatability_raw) * 10.0
        compatability = compatability_raw
        # print("compatability")
        # print(compatability[0])
        #max pointless options
        pointless = torch.eye(seq_len).repeat(batch_size, 1, 1).to(compatability_raw.device)
        compatability[pointless.bool()] = -np.inf
        # print("compatability", compatability[0])
        # mask previous choice exchange
        if exchange is not None:
            compatability[torch.arange(batch_size), exchange[:,0], exchange[:,1]] = -np.inf
            compatability[torch.arange(batch_size), exchange[:,1], exchange[:,0]] = -np.inf
        # print(compatability[0:1])
        c = compatability.view(batch_size, -1)
        logits = F.softmax(c, dim=-1)
        # print(logits[0])
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
    def __init__(self, env, trainer, model_config, optimizer_config):
        self.env = env
        self.trainer = trainer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = TSP_improve(model_config).to(self.device)
        self.trainer.passIntoParameters(self.actor.parameters(), optimizer_config)
        self.T = int(optimizer_config['T'])

    # given a problem size and batch size will train the model
    def train(self, n_batch, p_size, path=None):
        problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
        # setup inital parameters
        state = self.env.getStartingState(n_batch, p_size).to(self.device)
        best_so_far = self.env.evaluate(problems, state).to(self.device)
        exchange = None
        # tracking stuff for information
        action_history = [] #INFO
        state_history = [state]
        reward_history = [] #INFO
        total_loss = 0
        # run through the model
        t = 0
        self.actor.train()
        while t < self.T:
            probability, exchange = self.actor(problems, state, exchange)
            next_state = self.env.step(state, exchange)
            # calculate reward
            cost = self.env.evaluate(problems, next_state).to(self.device)
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            reward = best_for_now - best_so_far #unflipped reward
            # reward = best_so_far - best_for_now #
            #save things
            reward_history.append(cost)
            action_history.append(exchange)
            state_history.append(next_state)
            # train
            # print("Reward: {0}, prob: {1}, action:{2}".format(reward[0], probability[0].item(), exchange[0]))
            R, loss_dict = self.trainer.train(problems, reward, probability)
            total_loss += loss_dict['actor_loss']
            #update for next iteration
            state = next_state
            best_so_far = best_for_now
            t += 1
        return best_so_far, {'actor_loss': total_loss}

    # given a batch size and problem size will test the model
    def test(self, n_batch, p_size, path=None):
        problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
        # setup inital parameters
        state = self.env.getStartingState(n_batch, p_size).to(self.device)
        best_so_far = self.env.evaluate(problems, state) #[batch_size]
        exchange = None
        t = 0
        # pass through model
        # run through the model
        self.actor.eval()
        while t < self.T:
            #pass through model
            _, exchange = self.actor(problems, state, exchange)
            state = self.env.step(state, exchange)
            cost = self.env.evaluate(problems, state)
            best_so_far = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            #setup for next round
            t += 1
        return best_so_far

    def save(self):
        model_dict = {}
        model_dict['actor_model_state_dict'] = self.actor.actor.state_dict()
        model_dict.update(self.trainer.save())
        return model_dict

    def load(self, checkpoint):
        self.actor.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.trainer.load(checkpoint)
