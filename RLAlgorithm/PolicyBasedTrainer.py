import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from Models.ConstructionPointerNetwork import PntrNetCritic

import math
import random
import numpy as np

class Rollout:
    def __init__(self, rollout_count):
        self.rollout_count = rollout_count

    def update(self, problems, model, env):
        # run through model
        model.eval()
        _, action_list = model(problems, sampling=True)
        rollouts = []
        for i in range(self.rollout_count):
            R = env.evaluate(problems, action_list.transpose(0, 1)).to(problems.device)
            rollouts.append(R)
        return torch.mean(torch.stack(rollouts, dim=0), dim=0)

    def save(self):
        return {}

    def load(self, checkpoint):
        pass

class ExpMovingAvg:
    def __init__(self):
        self.TRACK_critic_exp_mvg_avg = None

    def update(self, newReward):
        if self.TRACK_critic_exp_mvg_avg is None:
            self.TRACK_critic_exp_mvg_avg = newReward.detach().mean()
        else:
            self.TRACK_critic_exp_mvg_avg = (self.TRACK_critic_exp_mvg_avg * 0.9) + (newReward.detach().mean() * 0.1)
        return self.TRACK_critic_exp_mvg_avg

    def save(self):
        model_dict = {}
        model_dict['critic_exp_mvg_avg'] = self.TRACK_critic_exp_mvg_avg
        return model_dict

    def load(self, checkpoint):
        self.TRACK_critic_exp_mvg_avg = checkpoint['critic_exp_mvg_avg']

class Reinforce:
    def __init__(self, baseline_config):
        self.baseline_type = baseline_config['baselineType']
        if baseline_config['baselineType'] == 'ExpMovingAvg':
            self.baseline = ExpMovingAvg()
        elif baseline_config['baselineType'] == 'Rollout':
            self.baseline = Rollout(int(baseline_config['rolloutCount']))
        elif baseline_config['baselineType'] == 'None':
            pass
        else:
            print("Invalid baseline type")

    def passIntoParameters(self, parameters, optimizer_config):
        self.actor_param = parameters
        self.actor_optimizer = optim.Adam(parameters, lr=float(optimizer_config['actor_lr']))
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1, gamma=float(optimizer_config['actor_lr_gamma']))
        self.max_g = float(optimizer_config['maxGrad'])

    # reward, probability: [steps, n_batch]
    # problems: [steps, n_batch, n_nodes, 4]
    def train(self, problems, reward, probability, model=None, env=None):
        # reward is what the model is training to minimize
        # probability is the probability that it took whatever set of actions led to this reward
        # problems is so that the critic can analyse it
        steps, n_batch = reward.size()
        assert (probability.size() == reward.size()), "Reward size {0} does not match probability size {1}".format(reward.size(), probability.size())
        if self.baseline_type == 'ExpMovingAvg':
            baseline = self.baseline.update(reward.mean(dim=0))
        elif self.baseline_type == 'Rollout': #TODO
            baseline = self.baseline.update(problems.view(steps*n_batch), model, env).view(steps, n_batch)
        elif self.baseline_type == 'None':
            baseline = torch.zeros_like(reward)
        with torch.autograd.set_detect_anomaly(True):
            advantage = reward - baseline.detach()
            logprobability = torch.log(probability)
            reinforce = (advantage * logprobability)
            actor_loss = reinforce.mean()
            # update the weights using optimiser
            self.actor_optimizer.zero_grad()
            actor_loss.backward() # calculate gradient backpropagation
            torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_g, norm_type=2) # to prevent gradient expansion, set max
            self.actor_optimizer.step() # update weights
            self.actor_scheduler.step()
        return reward, {'actor_loss': actor_loss}

    def additonal_params(self):
        return ['actor_loss']

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

class A2C:
    def __init__(self, critic_config):
        if critic_config['model_type'] == 'PointerNetwork':
            self.wrapped_actor = PntrNetCritic(critic_config)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(critic_config['critic_lr']))
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1, gamma=float(critic_config['critic_lr_gamma']))
        self.critic_mse_loss = nn.MSELoss()

    def passIntoParameters(self, parameters, optimizer_config):
        self.actor_param = parameters
        self.actor_optimizer = optim.Adam(parameters, lr=float(optimizer_config['actor_lr']))
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1, gamma=float(optimizer_config['actor_lr_gamma']))
        self.max_g = float(optimizer_config['maxGrad'])

    def additonal_params(self):
        return ['actor_loss', 'critic_loss']

    def train(self, problems, reward, probability):
        critic_reward = self.critic(problems.detach())
        # train critic
        critic_loss = self.critic_mse_loss(reward, critic_reward)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.critic_optimizer.step()
        self.critic_scheduler.step() #DO IN THE LATER LAYER?
        # train actor
        advantage = reward - critic_reward.detach()
        logprobability = torch.log(probability)
        reinforce = (advantage * logprobability)
        actor_loss = reinforce.mean()
        # update the weights using optimiser
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.actor_optimizer.step() # update weights
        self.actor_scheduler.step() #TOD IN LATER LAYER?
        return actual_R, {'actor_loss': actor_loss, 'critic_loss': critic_loss}

    def save(self):
        model_dict = {}
        model_dict['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
        model_dict['actor_schedular_state_dict'] = self.actor_scheduler.state_dict()
        model_dict['critic_model_state_dict'] = self.critic.state_dict()
        model_dict['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        model_dict['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        return model_dict

    def load(self, checkpoint):
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_schedular_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
