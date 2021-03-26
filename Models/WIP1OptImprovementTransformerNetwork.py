import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, TransformerEncoderL

class Decoder(nn.Module):
    def __init__(self, dim_model, dim_k, dim_v, n_head):
        self.MMA = SkipConnection(MultiHeadAttention(dim_model, dim_k, dim_v, n_head))
        self.norm_1 = Normalization(dim_model)
        self.linear = nn.Linear(dim_model, 1)

    def forward(inputs):
        dec_output = self.MMA(inputs) #how is the multihead attention pulled together? mean? addition?
        dec_output = self.norm_1(dec_output)
        dec_output = self.linear(dec_output).squeeze() #TODO
        return dec_output

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
        self.L_encoder = nn.Sequential(*(TransformerEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_project_graph = nn.Linear(dim_model, dim_model, bias=False)
        self.L_project_node = nn.Linear(dim_model, dim_model, bias=False)
        nn.Sequential(*(TransformerEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_decoder = Decoder(dim_model, dim_k, dim_v, n_head)

    def forward(self, problems, solution_indexes, do_sample=True):
        n_node = problems.size(1)
        x_embed = self.L_embedder(problems, solution_indexes)
        x_encode = self.L_encoder(x_embed) # torch.Size([2, 50, 128])
        # embd graph and node features
        max_pooling = x_encode.max(1)[0] # max Pooling
        graph_feature = self.L_project_graph(max_pooling)[:, None, :]
        node_feature = self.L_project_node(x_encode)
        # pass it through decoder
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([batch_size, n_node, 128])
        logits = self.L_decoder(fusion) # # att [n_batch, n_node]
        # select or sample
        action = logits.multinomial(1) if do_sample else logits.max(-1)[1].view(-1,1)
        selected_likelihood = logits.gather(1, pair_index)
        return selected_likelihood, action

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
        initial_result = best_so_far.clone()
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
            probability, action = self.actor(problems, state)
            next_state = self.env.step(state, action)
            # calculate reward
            cost = self.env.evaluate(problems, next_state).to(self.device)
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            # reward = best_for_now - best_so_far #unflipped reward
            reward = cost - initial_result
            #save things
            reward_history.append(reward)
            action_history.append(exchange)
            state_history.append(next_state)
            # train
            R, loss_dict = self.trainer.train(problems, reward, probability, self.actor, self.env)
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
        self.actor.eval()
        while t < self.T:
            #pass through model
            _, action = self.actor(problems, state)
            state = self.env.step(state, action)
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
