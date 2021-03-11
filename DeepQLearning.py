import copt
import PointerNetwork
import Environments
import copy

#torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
import math

class DeepQLearning:
    def create(self, device, critic, glimpse, env,
            embedding_size, hidden_size, # pntr-ntwrk parameters
            lr=1e-3, target_update=10, gamma=0.9): # hyperparameteres
        self.device = device
        self.env = {'Construction': Environments.Construction()}.get(env, None)
        self.M_policy = PointerNetwork.TDPntrN(embedding_size, hidden_size, glimpse).to(device)
        self.M_target = PointerNetwork.TDPntrN(embedding_size, hidden_size, glimpse).to(device)
        #SOME OF THE PARAMETERS WON@T MATCH, MAKE THEM MATCH
        self.M_policy_optim   = optim.Adam(self.M_policy.parameters(), lr=lr) #implements gradient descent
        #self.optim   = optim.RMSprop(self.policy_model.parameters(), lr=lr)
        # self.memory = ReplayMemory(10000) # NEED TO CREATE

        self.gamma = gamma
        self.target_update = target_update #how often to update target_model
        self.critic = critic

        self.update_exploration=True,
        self.initial_exploration_rate=1
        self.final_exploration_rate=0.05  # 0.05
        self.final_exploration_step=1000  # 40000
        self.epsilon = self.initial_exploration_rate

        self.REPLAYMEM_capacity = 1000
        self.REPLAYMEM_memory = []
        self.REPLAYMEM_position = 0

    # given a problem size and batch size will train the model
    def train(self, batch_size, problem_size):
        # generate problems
        problems = torch.FloatTensor(self.env.genProblems(batch_size, problem_size)).to(self.device)
        # pass through model
        self.M_policy.train()
        self.M_target.eval()
        state = None
        total_loss = 0
        for i in range(problem_size):
            # compute initial_state_value - using policy_NN
            pred_q, actions = self.M_policy(state, problems)
            state = self.env.nextState(state, actions) # get the next state
            R = self.env.evaluate(problems, state).to(self.device) # get the reward
            # compute the next_state_value.detach() - using target_NN
            if (state.size(1) == problem_size): # if next state is not terminal
                next_q = torch.zeros((batch_size, 1)).squeeze().to(self.device)
            else:
                next_q,_ = self.M_target(next_state, problems).detach()
            # calculate the target Q
            target_q = (next_q * self.gamma) + R
            loss = F.smooth_l1_loss(pred_q, target_q)# Compute Huber loss
            total_loss += loss
            # update weights
            self.M_policy_optim.zero_grad()
            loss.backward()
            self.M_policy_optim.step()
        return R, total_loss

    def test(self, batch_size, problem_size):
        # generate problems
        self.M_actor.eval()
        problems = torch.FloatTensor(self.env.genProblems(batch_size, problem_size)).to(self.device)
        # pass through model
        self.policy_model.eval()
        self.target_model.eval()
        # do it seq_len times
        Rs = []
        next_state = None
        for i in range(self.prob_size):
            pred_q, actions = self.M_policy(state, problems)
            state = self.env.nextState(state, actions) # get the next state
            R = self.env.evaluate(problems, state) # get the reward
        return R

# MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = 'Construction' # ['Construction', 'Improvement']
use_critic = False
embedding_size = 256
hidden_size    = 256
lr = 1e-4
glimpse_count = 1
target_update = 10
gamma = 0.9


agent = DeepQLearning()
agent.create(device, use_critic, glimpse_count, env, embedding_size, hidden_size,# pntr-network parameters
        lr = lr, target_update = target_update, gamma=gamma)# hyperparameteres

#TRAINING TESTING DETAILS
n_epochs = 1
n_batch_pEpoch = 100
n_test_pEpoch = 10
train_batch_size = 100
test_batch_size = 1000
test_threshold = 90
prob_size = 5
print("Batch per epoch: {0}, Number of epochs: {1}".format(n_batch_pEpoch, n_epochs))

# TENSORBOARD
long_lr = "{:.1e}".format(lr)
data_loc = f'runs/MNIST/DQN/TEST4-lr{long_lr}-bs{train_batch_size}-n{prob_size}'
t_board = SummaryWriter(data_loc)
test_step = 0


# save model
for epoch in range(n_epochs):
    # load model
    #loop through batches of the test problems
    for batch in range(n_batch_pEpoch):
        R, actor_loss = agent.train(train_batch_size, prob_size)
        R_routed = [x for x in R if (x != 10000)]
        t_board.add_scalar('Train/Actor_loss', actor_loss, global_step = batch)
        t_board.add_scalar('Train/AvgRoutedR', sum(R_routed)/len(R_routed), global_step = batch)
        t_board.add_scalar('Train/AvgR', R.mean().item(), global_step = batch)
        t_board.add_scalar('Train/AvgRouted%', (len(R_routed)/train_batch_size)*100, global_step = batch)
        if use_critic:
            t_board.add_scalar('Train/Critic_loss%', critic_loss, global_step = batch)
        if batch % int(n_batch_pEpoch/n_test_pEpoch) == 0 or batch == (n_batch_pEpoch-1):
            R = agent.test(test_batch_size, prob_size)
            R_routed = [x for x in R if (x != 10000)]
            routed_perc = (len(R_routed)/test_batch_size)*100
            mean_routed = sum(R_routed)/len(R_routed)
            t_board.add_scalar('Test/AvgRoutedR', mean_routed, global_step = test_step)
            t_board.add_scalar('Test/AvgR', R.mean().item(), global_step = test_step)
            t_board.add_scalar('Test/AvgRouted%', (len(R_routed)/test_batch_size)*100, global_step = test_step)
            print("Finished test batch {0}".format(test_step))
            test_step += 1
        if batch_no % self.target_update == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            param1a, param2a = self.target_model.getParams()
            self.policy_model.updateParam(param1a, param2a)
        if self.update_exploration:
            eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
                batch_no / self.final_exploration_step)
            self.epsilon = max(eps, self.final_exploration_rate)
    print("Finished epoch: {0}".format(epoch))
    if (routed_perc >= test_threshold):
        print("EARLY STOPPAGE! Routed perc: {0}, threshold: {1}!".format(routed_perc, test_threshold))
        break
t_board.add_hparams({'batchSize': train_batch_size,
    'learning rate':lr, 'problem size': prob_size},
    {'avgRouted': routed_perc, 'avgReward': mean_routed})

        # generate problems
        # problem = Variable(torch.FloatTensor(genProblemList(1, self.prob_size)))
        # current_state = []
        # is_training_ready = False
        # for i in range(episode):
        #     _, R, actions = self.policy_model(current_state, problems)
        #     next_state = current_state
        #     next_state.append(actions)
        #     next_state = torch.as_tensor(next_state)
        #     self.memory.add(state, action, reward, next_state, done)
        #
        #     if done == True:
        #         #reset
        #         problem = Variable(torch.FloatTensor(genProblemList(1, self.prob_size)))
        #         current_state = []
        #
        #     if self.memory.isFull() == True:
        #         #update the main network
        #         if i % self.update_frequency == 0:
        #             # Sample a batch of transitions
        #             transitions = self.memory.sample(self.batch_size)
        #             # Train on selected batch
        #             self.policy_model.train()
        #             self.target_model.eval()
        #
        #             # compute initial_state_value - using policy_NN
        #             state_action_values, R, actions = self.policy_model(transitions, problems)
        #             next_state.append(actions)
        #             if (len(next_state) < seqLen):
        #                 next_state_value = self.target_model(next_state, problems)[0].detach() #TODO
        #             else:
        #                 next_state_value = torch.zeros((batchSize, 1)).squeeze().detach().to(self.device)
        #             expected_state_action_values = (next_state_value * self.gamma) + R
        #             initial_state_value = state_action_values
        #             # Compute Huber loss
        #             loss = F.smooth_l1_loss(initial_state_value, expected_state_action_values)
        #             # update the weights using optimiser
        #             self.optim.zero_grad() # zero the gradient
        #             loss.backward()#retain_graph=True) # calculate gradient backpropagation
        #             # for param in self.policy_model.parameters():
        #             #     param.grad.data.clamp_(-1, 1)
        #             self.optim.step() #update weights
        #         #update the target network
        #         if batch_no % self.target_update == 0:
        #             self.target_model.load_state_dict(self.policy_model.state_dict())
        #
        #         self.TENSORBOARD_write.add_scalar('Train/Loss', total_loss, global_step = self.TENSORBOARD_test_step)
        #         logStats('train', self.TENSORBOARD_write, self.TENSORBOARD_test_step, R)
        #         self.TENSORBOARD_test_step += 1
