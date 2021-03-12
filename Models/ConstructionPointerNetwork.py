import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import math
import random
import numpy as np

from Models.GeneralLayers import *
from Models.TransformerWIP import TEncoder

class BahdanauAttention(nn.Module):
    def __init__(self, dim_model, compatabilit='add'):
        super().__init__() #initialise nn.Modules
        self.L_query = nn.Linear(dim_model, dim_model, bias = True)
        self.L_ref = nn.Conv1d(dim_model, dim_model, 1, 1)
        self.L_V = nn.Parameter(torch.FloatTensor(dim_model))
        self.setParam(dim_model)

    def setParam(self, dim_model):
        self.L_V.data.uniform_(-(1. / math.sqrt(dim_model)) , 1. / math.sqrt(dim_model))

    # inputs:
    #   query [n_batch, dim_model]
    #   ref [n_batch, seq_len, dim_model]
    # outputs: U [n_batch, seq_len]
    def forward(self, query, ref):
        query = self.L_query(query).unsqueeze(2).repeat(1,1,ref.size(1)) # query: [n_batch, dim_model, seq_len]
        ref = self.L_ref(ref.transpose(1, 2)) # ref: [n_batch, dim_model, seq_len]
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
    #   ref [n_batch, seq_len, dim_model]
    # outputs: query [n_batch, dim_model]
    def forward(self, query, ref, mask=None):
        for i in range(self.n_glimpse):
            u, ref2 = self.L_glimpse(query, ref) # ref2: (batch, 128, city_t)
            if mask is not None:
                u[[[i for i in range(u.size(0))], mask]] = -1e8 #mask it
                # u = u - 1e8 * mask
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
        self.L_decoder_input = nn.Parameter(torch.FloatTensor(dim_model))
        self.setParam(dim_model)

    def setParam(self, dim_model):
        self.L_decoder_input.data.uniform_(-(1. / math.sqrt(dim_model)), 1. / math.sqrt(dim_model))

    # input: problems - [n_batch, seq_len, 4]
    # output: logits -  [n_batch, seq_len]
    def forward(self, problems, sampling=True):
        """
        Args:
            inputs: [n_batch, seq_len, 4]
        """
        n_batch = problems.size(0)
        #ENCODE PROBLEM
        embd_graph = self.L_embedder(problems)
        enc_states, (initial_h, initial_c) = self.L_encoder(embd_graph)
        h_c = (initial_h, initial_c)
        # h & c, [1, n_batch, dim_model]
        # embd_graph & enc_states, [n_batch, seq_len, dim_model]

        action_list = None #a ction_list: ([step, n_batch])
        action_probs_list = [] # probability of each action taken, action_probs_list: (step x [n_batch])
        while (action_list is None or action_list.size(0) != problems.size(1)):
            if action_list is None:
                dec_input = self.L_decoder_input.unsqueeze(0).repeat(n_batch, 1) # decoder_input: [n_batch, dim_embedding]
            else:
                dec_input = embd_graph[[i for i in range(n_batch)], action_list[-1].data, :] # takes the corresponding embeddedGraph[actions]
            _, h_c = self.L_decoder(dec_input.unsqueeze(1), h_c)
            query = self.L_glimpse(h_c[0].squeeze(0), enc_states, action_list)
            # logits is the output of the pointer network, the weights associated with each element in the sequence
            logits, _ = self.L_pointer(query, enc_states)
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
        return action_probs_list, action_list

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
                    nn.Linear(dim_model, 1, bias = True))

    def forward(self, inputs):
        embd_graph = self.L_embedder(inputs)
        enc_states, (h, c) = self.L_encoder(embd_graph) # encoder_states: torch.Size([n_batch, seq_len, dim_model])
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
    def __init__(self, env, trainer, model_config, optimizer_config): # hyperparameteres
        self.env = env
        self.trainer = trainer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PtrNet(model_config).to(self.device)
        self.trainer.passIntoParameters(self.actor.parameters(), optimizer_config)

    # given a problem size and batch size will train the model
    def train(self, n_batch, p_size, path=None):
        problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
        # run through the model
        self.actor.train()
        action_probs_list, action_list = self.actor(problems, sampling=True) #action_probs_list (seq_len x [n_batch])
        # calculate reward and probability
        reward = self.env.evaluate(problems.detach(), action_list.transpose(0, 1)).to(self.device)
        probs = action_probs_list[0]
        for i in range(1, len(action_probs_list)):
            probs = probs * action_probs_list[i]
        # use this to train
        R, loss = self.trainer.train(problems, reward, probs)
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
