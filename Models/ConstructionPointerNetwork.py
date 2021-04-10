import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import math
import random
import numpy as np

from Models.GeneralLayers import *

class BahdanauAttention(nn.Module):
    def __init__(self, dim_model, compatabilit='add'):
        super().__init__() #initialise nn.Modules
        self.L_query = nn.Linear(dim_model, dim_model, bias = True)
        self.L_ref = nn.Conv1d(dim_model, dim_model, 1, 1)
        self.L_V = nn.Parameter(torch.FloatTensor(dim_model).uniform_(-(1. / math.sqrt(dim_model)) , 1. / math.sqrt(dim_model)))

    # inputs:
    #   query [n_batch, dim_model]
    #   ref [n_batch, n_node, dim_model]
    # outputs: U [n_batch, n_node]
    def forward(self, query, ref):
        query = self.L_query(query).unsqueeze(2).repeat(1,1,ref.size(1)) # query: [n_batch, dim_model, n_node]
        ref = self.L_ref(ref.transpose(1, 2)) # ref: [n_batch, dim_model, n_node]
        V = self.L_V.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1) #V: [n_batch, 1, dim_model]
        U = torch.bmm(V, (query + ref).tanh()*10).squeeze(1)
        return U, ref

class Glimpse(nn.Module):
    def __init__(self, dim_model, glimpse_no):
        super().__init__() #initialise nn.Module
        self.n_glimpse = glimpse_no
        self.L_glimpse = BahdanauAttention(dim_model)

    # inputs:
    #   query [n_batch, dim_model]
    #   ref [n_batch, n_node, dim_model]
    # outputs: query [n_batch, dim_model]
    def forward(self, query, ref, mask=None):
        for i in range(self.n_glimpse):
            u, ref2 = self.L_glimpse(query, ref) # ref2: (batch, 128, city_t)
            if mask is not None:
                u = u.masked_fill(mask, float('-inf')) # mask previous actions
            logits = F.softmax(u, dim = 1) # logits: (batch, city_t, 1)
            query = torch.bmm(ref2, logits.unsqueeze(2)).squeeze(2)
        return query

class PtrNet(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Module
        # defining layers and intialising them
        dim_embedding = int(model_config['embedding'])
        dim_model = int(model_config['hidden'])
        n_glimpse = int(model_config['glimpse'])
        self.L_embedder = GraphEmbedding(dim_embedding)
        self.L_encoder = nn.LSTM(dim_embedding, dim_model, batch_first=True)
        self.L_decoder = nn.LSTM(dim_embedding, dim_model, batch_first=True)
        self.L_pointer = BahdanauAttention(dim_model)
        self.L_glimpse = Glimpse(dim_model, n_glimpse)
        self.L_decoder_input = nn.Parameter(torch.FloatTensor(dim_model).uniform_(-(1. / math.sqrt(dim_model)), 1. / math.sqrt(dim_model)))

    # input: problems - [n_batch, n_node, 4]
    # output: logits -  [n_batch, n_node]
    def forward(self, problems, sampling=True):
        """
        Args:
            inputs: [n_batch, n_node, 4]
        """
        n_batch, n_node, _  = problems.size()
        #ENCODE PROBLEM details
        embd_graph = self.L_embedder(problems)
        enc_states, (initial_h, initial_c) = self.L_encoder(embd_graph) # embd_graph & enc_states, [n_batch, n_node, dim_model]
        h_c = (initial_h, initial_c) # h & c, [1, n_batch, dim_model]
        #setup initial variables
        mask = torch.zeros([n_batch, n_node], device=problems.device).bool()
        dec_input = self.L_decoder_input.unsqueeze(0).repeat(n_batch, 1) # decoder_input: [n_batch, dim_embedding]
        action_list, action_probs_list = [], [] #action_list and action_probs_list:  (step x [n_batch])
        while (len(action_list) < problems.size(1)):
            _, h_c = self.L_decoder(dec_input.unsqueeze(1), h_c)
            query = self.L_glimpse(h_c[0].squeeze(0), enc_states, mask)
            logits, _ = self.L_pointer(query, enc_states) # logits is the output of the pointer network, the weights associated with each element in the sequence
            logits = logits.masked_fill(mask, float('-inf')) # mask previous actions
            probs = F.softmax(logits, dim=1) #soft max the probabilities
            actions = probs.multinomial(1).squeeze(1) if sampling else probs.argmax(dim = 1) # pick an action, actions: torch.Size([100])
            # add it to various lists
            action_list.append(actions)
            action_probs_list.append(probs[[x for x in range(len(probs))], actions])
            # action_probs_list.append(probs.gather(0, actions.unsqueeze(dim=1)))
            # update for next loop
            mask = mask.scatter(1, actions.unsqueeze(dim=-1), True)
            dec_input = embd_graph[[i for i in range(n_batch)], actions, :] # takes the corresponding embeddedGraph[actions]
        return torch.stack(action_probs_list, dim=1), torch.stack(action_list, dim=1)

class PntrNetCritic(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Module
        dim_embedding = int(model_config['embedding'])
        dim_model = int(model_config['hidden'])
        n_glimpse = int(model_config['glimpse'])
        n_processing = int(model_config['processing'])
        self.n_process = n_processing
        self.L_embedder = GraphEmbedding(dim_embedding)
        self.L_encoder = nn.LSTM(dim_embedding, dim_model, batch_first=True)
        self.L_glimpse = Glimpse(dim_model, n_glimpse)
        self.L_decoder = nn.Sequential(
                    nn.Linear(dim_model, dim_model, bias = False),
			        nn.ReLU(inplace = False),
                    nn.Linear(dim_model, 1, bias = False))

    def forward(self, problems, states):
        embd_graph = self.L_embedder(problems)
        enc_states, (h, c) = self.L_encoder(embd_graph) # encoder_states: torch.Size([n_batch, n_node, dim_model])
        query = h.squeeze(0) #the hidden step after all the elements of the sequence were processed
        for i in range(self.n_process):
            query = self.L_glimpse(query, enc_states)
        pred_l = self.L_decoder(query).squeeze(-1)
        return pred_l

# TRAIN - given batch_size and problem_size and train on that batch
# TEST - given batch_size and problem_size and test on that batch
# ADDITIONAL_PARAMETERS - give an idea of additional training parameters
# SAVE/LOAD - save or load a copy of the model

# **REINFORCE policy-based method**
#how the model output is returned into reward and probability
class PtrNetWrapped: #wrapper for model
    # creates the everything around the networks
    def __init__(self, env, trainer, device, config): # hyperparameteres
        self.env = env
        self.trainer = trainer
        self.device = device
        self.actor = PtrNet(config).to(self.device)
        self.trainer.passIntoParameters(self.actor.parameters(), config)

    # given a problem size and batch size will train the model
    def train_batch(self, n_batch, p_size, path=None):
        problems = self.env.gen(n_batch, p_size, self.device) if (path is None) else self.env.load(path, n_batch, self.device) # generate or load problems
        # run through the model
        self.actor.train()
        # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
        action_probs_list, action_list = self.actor(problems, sampling=True) #action_probs_list (n_node x [n_batch])
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        # calculate reward and probability
        reward = self.env.evaluate(problems, action_list, self.device)
        probs = action_probs_list.prod(dim=1)
        # use this to train
        actor_loss, baseline_loss = self.trainer.train(problems, action_list.unsqueeze(dim=0), reward.unsqueeze(dim=0), probs.unsqueeze(dim=0))
        return reward, actor_loss, baseline_loss

    # given a batch size and problem size will test the model
    def test(self, n_batch, p_size, path=None):
        problems = self.env.gen(n_batch, p_size, self.device) if (path is None) else self.env.load(path, n_batch, self.device) # generate or load problems
        # run through model
        self.actor.eval()
        action_probs_list, action_list = self.actor(problems, sampling=True)
        reward = self.env.evaluate(problems, action_list, self.device)
        return reward

    def save(self):
        model_dict = {}
        model_dict['actor_model_state_dict'] = self.actor.state_dict()
        model_dict.update(self.trainer.save())
        return model_dict

    def load(self, checkpoint):
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.trainer.load(checkpoint)
