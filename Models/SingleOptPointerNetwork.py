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
    def forward(self, problems, actn_sele):
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
            if actn_sele == 'sampling':
                probs = F.softmax(logits, dim=1) #soft max the probabilities
                actions = probs.multinomial(1).squeeze(1) # sample an index for each problem in the batch, actions: torch.Size([100])
            elif actn_sele == 'greedy':
                actions = logits.argmax(dim = 1)
                probs = Variable(torch.zeros(logits.size()), requires_grad = True).to(logits.device)
                probs[:, actions] = 1
            else:
                raise NotImplementedError
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
class TrainPointerNetwork:
    # creates the everything around the networks
    def __init__(self, env, models, config): # hyperparameteres
        # set variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = True if (models[1] is not None) else False
        self.max_g = float(config['maxGrad'])
        self.env = env
        # create models
        self.M_actor = models[0].to(self.device)
        if self.critic and len(models) > 1:
            self.M_critic = models[1].to(self.device)
        # setup optimizer
        self.M_actor_optim = optim.Adam(self.M_actor.parameters(), lr=float(config['actor_lr']))
        self.M_actor_scheduler = torch.optim.lr_scheduler.StepLR(self.M_actor_optim, step_size=1, gamma=float(config['actor_lr_gamma']))
        if self.critic:
            self.M_critic_optim   = optim.Adam(self.M_critic.parameters(), lr=float(config['critic_lr']))
            self.M_critic_scheduler = torch.optim.lr_scheduler.StepLR(self.M_critic_optim, step_size=1, gamma=float(config['critic_lr_gamma']))
            self.mse_loss = nn.MSELoss()
        else:
            self.TRACK_critic_exp_mvg_avg = None

    def additonal_params(self):
        return ['actor_loss', 'critic_loss']

    # given a problem size and batch size will train the model
    def train(self, batch_size, problem_size, data_loc=None):
        # generate problems
        if data_loc is None:
            problems = self.env.gen(batch_size, problem_size).to(self.device)
        else:
            #load the dataset
            problems = self.env.load(data_loc, batch_size).to(self.device)
        # run through the model
        self.M_actor.train()
        action_probs_list, action_list = self.M_actor(problems, 'sampling')
        actual_R = self.env.evaluate(problems.detach(), action_list.transpose(0, 1)).to(self.device)
        if self.critic:
            pred_R = self.M_critic(problems.detach())
            critic_loss = self.mse_loss(actual_R, pred_R)
            self.M_critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.M_critic.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
            self.M_critic_optim.step()
            self.M_critic_scheduler.step()
        else:
            if self.TRACK_critic_exp_mvg_avg is None:
                self.TRACK_critic_exp_mvg_avg = actual_R.detach().mean()
            else:
                self.TRACK_critic_exp_mvg_avg = (self.TRACK_critic_exp_mvg_avg * 0.9) + (0.1 * actual_R.detach().mean())
            pred_R = self.TRACK_critic_exp_mvg_avg
            critic_loss = 0
        # calculate advantage and solution probability to calculate actor loss
        advantage = actual_R - pred_R.detach()
        logprobs = 0
        for prob in action_probs_list:
            logprob = torch.log(prob)
            logprobs += logprob
        logprobs[logprobs < -1000] = 0.
        reinforce = (advantage * logprobs)
        actor_loss = reinforce.mean()
        # update the weights using optimiser
        self.M_actor_optim.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.M_actor.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.M_actor_optim.step() # update weights
        self.M_actor_scheduler.step()
        return actual_R, {'actor_loss': actor_loss, 'critic_loss': critic_loss}

    # given a batch size and problem size will test the model
    def test(self, batch_size, problem_size, data_loc=None):
        self.M_actor.eval()
        # generate problems
        if data_loc is None:
            problems = self.env.gen(batch_size, problem_size).to(self.device)
        else:
            #load the dataset
            problems = self.env.load(data_loc, batch_size).to(self.device)
        action_probs_list, action_list = self.M_actor(problems, 'sampling')
        R = self.env.evaluate(problems, action_list.transpose(0, 1)).to(self.device)
        return R

    # saves the model
    def save(self, path):
        model_dict = {}
        model_dict['actor_model_state_dict'] = self.M_actor.state_dict()
        model_dict['actor_optimizer_state_dict'] = self.M_actor_optim.state_dict()
        model_dict['actor_schedular_state_dict'] = self.M_actor_scheduler.state_dict()
        if self.critic:
            model_dict['critic_model_state_dict'] = self.M_critic.state_dict()
            model_dict['critic_optimizer_state_dict'] = self.M_critic_optim.state_dict()
            model_dict['critic_scheduler_state_dict'] = self.M_critic_scheduler.state_dict()
        else:
            model_dict['critic_exp_mvg_avg'] = self.TRACK_critic_exp_mvg_avg
        torch.save(model_dict, path)

    # loads the model
    def load(self, path):
        checkpoint = torch.load(path)
        self.M_actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.M_actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.M_actor_scheduler.load_state_dict(checkpoint['actor_schedular_state_dict'])
        if self.critic:
            self.M_critic.load_state_dict(checkpoint['critic_model_state_dict'])
            self.M_critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.M_critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        else:
            self.TRACK_critic_exp_mvg_avg = checkpoint['critic_exp_mvg_avg']
