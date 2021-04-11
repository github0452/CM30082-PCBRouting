import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from Models.ConstructionPointerNetwork import PntrNetCritic
from Models.ConstructionTransformer import TransformerCritic
from Models.ImprovementTransformer import TSP_improveCritic

import math
import random
import numpy as np

class ExpMovingAvg:
    def __init__(self, device, baseline_config):
        self.type = 'expMovingAvg'
        self.device = device
        self.exp_mvg_avg = torch.zeros(1).to(device)

    def train(self, pred_return, true_return):
        self.exp_mvg_avg = pred_return * 0.9 + true_return.mean() * 0.1
        return (pred_return - true_return).mean()

    def getBaseline(self, problems, states):
        exp_mvg_avg = self.exp_mvg_avg.expand(problems.size(0))
        return exp_mvg_avg

    def save(self):
        model_dict = {}
        model_dict['critic_exp_mvg_avg'] = self.exp_mvg_avg
        return model_dict

    def load(self, checkpoint):
        self.exp_mvg_avg = checkpoint['critic_exp_mvg_avg']

class Nones:
    def __init__(self, device, baseline_config):
        self.type = 'none'
        self.device = device

    def train(self, pred_return, true_return):
        return 0

    def getBaseline(self, problems, states):
        return torch.zeros_like(states).to(device)

    def save(self):
        return {}

    def load(self, checkpoint):
        pass

class Critic:
    def __init__(self, device, critic_config):
        self.type = 'critic'
        self.device = device
        if critic_config['model'] == 'PointerNetwork':
            self.critic = PntrNetCritic(critic_config).to(device)
        elif critic_config['model'] == 'Transformer':
            self.critic = TransformerCritic(critic_config).to(device)
        elif critic_config['model'] == 'TSP_improve':
            self.critic = TSP_improveCritic(critic_config).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(critic_config['learning_rate']))
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1, gamma=float(critic_config['learning_rate_gamma']))
        self.critic_mse_loss = nn.MSELoss()
        self.max_g = float(critic_config['max_grad'])

    def train(self, pred_return, true_return):
        # train critic
        critic_loss = self.critic_mse_loss(pred_return, true_return.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.critic_optimizer.step()
        self.critic_scheduler.step() #DO IN THE LATER LAYER?
        return critic_loss

    def getBaseline(self, problems, states):
        critic_return = self.critic(problems, states.detach())
        return critic_return

    def save(self):
        model_dict = {}
        model_dict['critic_model_state_dict'] = self.critic.state_dict()
        model_dict['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        model_dict['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        return model_dict

    def load(self, checkpoint):
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

class Reinforce:
    def __init__(self, device, baseline_config):
        self.baseline_type = baseline_config['baseline_type']
        if self.baseline_type == 'ExpMovingAvg':
            self.baseline = ExpMovingAvg(device, baseline_config)
        elif self.baseline_type == 'Critic':
            self.baseline = Critic(device, baseline_config)
        elif self.baseline_type == 'None':
            self.baseline = Nones(device, baseline_config)
        else:
            print("Invalid baseline type")

    def passIntoParameters(self, parameters, optimizer_config):
        self.actor_param = parameters
        self.actor_optimizer = optim.Adam(parameters, lr=float(optimizer_config['learning_rate']))
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1, gamma=float(optimizer_config['learning_rate_gamma']))
        self.max_g = float(optimizer_config['max_grad'])

    # problems: [n_batch, n_nodes, 4]
    # states: [steps, n_batch, n_nodes]
    # return, probabilities: [steps, n_batch]
    def train(self, problems, states, returns, probabilities):
        steps, n_batch, n_nodes = states.size()
        # get baseline
        problems = problems.unsqueeze(dim=1).repeat(1, steps, 1, 1).reshape(-1, n_nodes, 4)
        states = states.transpose(0, 1).reshape(-1, n_nodes)
        baseline = self.baseline.getBaseline(problems, states).view(n_batch, steps).transpose(0, 1)
        # calculate actor loss
        advantage = returns - baseline.detach()
        logprobabilities = torch.log(probabilities)
        reinforce = (advantage * logprobabilities)
        actor_loss = reinforce.mean()
        # train the baseline
        baseline_loss = self.baseline.train(baseline.reshape(-1), returns.reshape(-1))
        # train the actor - update the weights using optimiser
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.actor_optimizer.step() # update weights
        self.actor_scheduler.step()
        return actor_loss, baseline_loss

    def useBaseline(self, problems, states):
        if self.baseline_type != 'Critic':
            return 0
        else:
            return self.baseline.getBaseline(problems, states).detach()

    def save(self):
        model_dict = {}
        model_dict['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
        model_dict['actor_schedular_state_dict'] = self.actor_scheduler.state_dict()
        if not self.baseline_type == 'None':
            model_dict.update(self.baseline.save())
        return model_dict

    def load(self, checkpoint):
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_schedular_state_dict'])
        if not self.baseline_type == 'None':
            self.baseline.load(checkpoint)
