import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, TransformerEncoderL

class Compatability(nn.Module):
    def __init__(self, dim_model, dim_key):
        super().__init__() #initialise nn.Modules
        self.n_heads = 1
        self.W_query = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        self.W_key = nn.Parameter(torch.Tensor(dim_model, self.n_heads*dim_key))
        self.init_params() # initialise parameters

    def init_params(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, query, prev_exchange, solution_indexes):
        ref = query.clone()
        batch_size, n_node, input_dim = ref.size()
        n_query = query.size(1)
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, n_node, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(query.contiguous().view(-1, input_dim), self.W_query).view(shp_q)
        K = torch.matmul(ref.contiguous().view(-1, input_dim), self.W_key).view(shp)
        compatability = torch.matmul(Q, K.transpose(2, 3)).squeeze(0)
        # compatability = torch.tanh(compatability) * 10.0
        # mask pointless options and previous exchange
        compatability[torch.eye(n_node, device=compatability.device).repeat(batch_size, 1, 1).bool()] = -np.inf
        if prev_exchange is not None:
            compatability[torch.arange(batch_size), prev_exchange[:,0], prev_exchange[:,1]] = -np.inf
            compatability[torch.arange(batch_size), prev_exchange[:,1], prev_exchange[:,0]] = -np.inf
        compatability = compatability.view(batch_size, -1)
        logits = F.softmax(compatability, dim=-1)
        return logits

class TSP_improve(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        dim_model = int(model_config['dim_model'])
        dim_hidden = int(model_config['dim_hidden'])
        dim_k = int(model_config['dim_k'])
        dim_v = dim_k
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        self.L_embedder = GraphEmbedding(dim_model, usePosEncoding=True)
        self.L_encoder = nn.Sequential(*(TransformerEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_project_graph = nn.Linear(dim_model, dim_model, bias=False)
        self.L_project_node = nn.Linear(dim_model, dim_model, bias=False)
        self.L_decoder = Compatability(dim_model, dim_model)

    def forward(self, problems, solution_indexes, prev_exchange, do_sample=True):
        n_node = problems.size(1)
        x_embed = self.L_embedder(problems, solution_indexes)
        x_encode = self.L_encoder(x_embed) # torch.Size([2, 50, 128])
        # embd graph and node features
        max_pooling = x_encode.max(1)[0] # max Pooling
        graph_feature = self.L_project_graph(max_pooling)[:, None, :]
        node_feature = self.L_project_node(x_encode)
        # pass it through decoder
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([batch_size, n_node, 128])
        logits = self.L_decoder(fusion, prev_exchange, solution_indexes) # # att torch.Size([batch_size, n_node^2])
        # select or sample
        pair_index = logits.multinomial(1) if do_sample else logits.max(-1)[1].view(-1,1)
        selected_likelihood = logits.gather(1, pair_index).squeeze(dim=-1)
        col_selected = pair_index % n_node
        row_selected = pair_index // n_node
        pair = torch.cat((row_selected,col_selected),-1)
        return selected_likelihood, pair

class TSP_improveCritic(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        dim_model = int(model_config['dim_model'])
        dim_hidden = int(model_config['dim_hidden'])
        dim_k = int(model_config['dim_k'])
        dim_v = int(model_config['dim_v'])
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        self.L_embedder = GraphEmbedding(dim_model, usePosEncoding=True)
        self.L_encoder = nn.Sequential(*(TransformerEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_project_graph = nn.Linear(dim_model, dim_model, bias=False)
        self.L_project_node = nn.Linear(dim_model, dim_model, bias=False)
        self.L_decoder = nn.Sequential(
                    nn.Linear(dim_model, dim_hidden, bias = False),
			        nn.ReLU(inplace = False),
                    nn.Linear(dim_hidden, 1, bias = False))

    def forward(self, problems, solution_indexes):
        n_node = problems.size(1)
        x_embed = self.L_embedder(problems, solution_indexes)
        x_encode = self.L_encoder(x_embed) # torch.Size([2, 50, 128])
        # embd graph and node features
        max_pooling = x_encode.max(1)[0] # max Pooling
        graph_feature = self.L_project_graph(max_pooling)[:, None, :]
        node_feature = self.L_project_node(x_encode)
        # pass it through decoder
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([batch_size, n_node, 128])
        value = self.L_decoder(fusion.mean(dim=1))
        return value

class TSP_improveWrapped:
    def __init__(self, env, trainer, device, config):
        self.env = env
        self.trainer = trainer
        self.device = device
        self.actor = TSP_improve(config).to(self.device)
        self.trainer.passIntoParameters(self.actor.parameters(), config)
        self.T = int(config['t'])

    # given a problem size and batch size will train the model
    def train_batch(self, n_batch, p_size, path=None):
        # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
        problems = self.env.gen(n_batch, p_size) if (path is None) else self.env.load(path, n_batch) # generate or load problems
        problems = torch.tensor(problems, device=self.device, dtype=torch.float)
        # setup inital parameters
        state = self.env.getStartingState(n_batch, p_size, self.device)
        best_so_far = torch.tensor(self.env.evaluate(problems, state), device=self.device, dtype=torch.float)
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        exchange = None
        # tracking stuff for information
        state_history = []
        reward_history = [] #INFO
        prob_history = []
        # run through the model
        self.actor.train()
        for _ in range(p_size ** self.T):
            # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
            probability, exchange = self.actor(problems, state, exchange)
            # print("forward", prof.key_averages().table(sort_by="self_cpu_time_total"))
            cost = torch.tensor(self.env.evaluate(problems, state), device=self.device, dtype=torch.float)
            # reward = cost - best_so_far is negatie, otherwise reward = 0
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            state = self.env.step(state, exchange)
            state_history.append(state)
            prob_history.append(probability)
            reward_history.append(best_for_now - best_so_far)
            best_so_far = best_for_now
        # train -  Get discounted R
        expected_returns = []
        reward_history = reward_history[::-1]
        # next_return = 0
        next_return = self.trainer.useBaseline(problems, state).squeeze(-1)
        for r in reward_history:
            next_return = next_return * 0.9 + r
            expected_returns.append(next_return)
        actor_loss, baseline_loss = self.trainer.train(problems, torch.stack(state_history, 0), torch.stack(expected_returns[::-1], 0), torch.stack(prob_history, 0))
        return best_so_far, actor_loss, baseline_loss

    # given a batch size and problem size will test the model
    def test(self, n_batch, p_size, path=None, sample_count=None):
        problems = self.env.gen(n_batch, p_size) if (path is None) else self.env.load(path, n_batch) # generate or load problems
        problems = torch.tensor(problems, device=self.device, dtype=torch.float)
        # setup inital parameters
        state = self.env.getStartingState(n_batch, p_size, self.device)
        best_so_far = torch.tensor(self.env.evaluate(problems, state), device=self.device, dtype=torch.float) #[batch_size]
        exchange = None
        self.actor.eval()
        for _ in range(0, p_size ** self.T):
            #pass through model
            _, exchange = self.actor(problems, state, exchange)
            state = self.env.step(state, exchange)
            cost = torch.tensor(self.env.evaluate(problems, state), device=self.device, dtype=torch.float)
            best_so_far = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            #setup for next round
        return best_so_far

    def save(self):
        model_dict = {}
        model_dict['actor_model_state_dict'] = self.actor.state_dict()
        model_dict.update(self.trainer.save())
        return model_dict

    def load(self, checkpoint):
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.trainer.load(checkpoint)
