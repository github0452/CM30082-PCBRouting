import os
import sys
import csv
import time
import datetime
import json
# import configparser
import numpy as np
from pathlib import Path
import tracemalloc

print("Env path", sys.path)

import torch
from torch.utils.tensorboard import SummaryWriter

from Misc.Environments import Construction, Improvement
from Models.ConstructionPointerNetwork import PtrNetWrapped
from Models.ImprovementTransformer import TSP_improveWrapped
from Models.ConstructionTransformer import TransformerWrapped
from RLAlgorithm.PolicyBasedTrainer import Reinforce

class TrainTest:
    def __init__(self, config, routableOnly=False):
        print("using config:", config)
        # check device
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        # setup data stuff
        data_path = config['data_path']
        Path(data_path).mkdir(parents=True, exist_ok=True)
        self.csv, self.tensor = None, None
        if bool(config['save_csv']):
            self.csv = {'train': '{0}/train.csv'.format(data_path),
                'test': '{0}/test.csv'.format(data_path)}
        if bool(config['save_tensor']):
            self.tensor = '{0}/tensor'.format(data_path)
        self.model_name = '{0}/checkpoint'.format(data_path)
        # setup testing and training stuff
        self.n_epoch = 0
        self.n_batch = int(config['n_batch'])
        self.n_batch_train_size = int(config['n_batch_train_size'])
        self.n_batch_test_size = int(config['n_batch_test_size'])
        #=-=-=-=-=-=-=ENVIRONMENTS and TRAINER=-=-=-=-===-=-=-=-=-=-=
        env = {'Construction': Construction(routableOnly), 'Improvement': Improvement(routableOnly)}.get(config['environment'])
        trainer = Reinforce(device, config)
        #=-=-=-=-===-=-=-=-=-=-=MODEL=-=-=-=-===-=-=-=-=-=-=
        model_type = config['model']
        if model_type == 'PointerNetwork': self.wrapped_actor = PtrNetWrapped(env, trainer, device, config)
        elif model_type == 'Transformer': self.wrapped_actor = TransformerWrapped(env, trainer, device, config)
        elif model_type == 'TSP_improve': self.wrapped_actor = TSP_improveWrapped(env, trainer, device, config)
        # self.load()

    def train(self, p_size, prob_path=None):
        # loop through batches
        init = self.n_epoch*self.n_batch
        for i in range(init, init+self.n_batch):
            #pass it through reinforcement learning algorithm to train
            R, actor_loss, baseline_loss = self.wrapped_actor.train_batch(self.n_batch_train_size, p_size, path=prob_path)
            R_routed = [x for x in R if (x != 10000)]
            avgR = R.mean().item()
            avgRoutedR = sum(R_routed).item()/len(R_routed) if len(R_routed) > 0 else 10000
            percRouted = len(R_routed)*100/self.n_batch_train_size
            if torch.is_tensor(baseline_loss):
                baseline_loss = baseline_loss.item()
            if torch.is_tensor(actor_loss):
                actor_loss = actor_loss.item()
            if self.csv is not None:
                if not os.path.isfile(self.csv['train']):
                    with open(self.csv['train'], 'w', newline='') as file:
                        csv.writer(file).writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%", "ActorLoss", "BaselineLoss"])
                with open(self.csv['train'], 'a', newline='') as file:
                    csv.writer(file).writerow([i, avgRoutedR, avgR, percRouted, actor_loss, baseline_loss])
            if self.tensor is not None:
                t_board = SummaryWriter(self.tensor)
                t_board.add_scalar('Train/AvgRoutedR', avgRoutedR, global_step = i)
                t_board.add_scalar('Train/AvgR', avgR, global_step = i)
                t_board.add_scalar('Train/AvgRouted%', percRouted, global_step = i)
                t_board.add_scalar('Train/ActorLoss', actor_loss, global_step = i)
                t_board.add_scalar('Train/BaselineLoss', baseline_loss, global_step = i)
        print("Epoch: {0}, Prob size: {1}, avgRoutedR: {2}, percRouted: {3}".format(self.n_epoch, p_size, avgRoutedR, percRouted))
        self.n_epoch += 1

    def test(self, p_size, prob_path=None, sample_count=1):
        # run tests
        R, time = self.wrapped_actor.test(self.n_batch_test_size, p_size, path=prob_path, sample_count=sample_count)
        R_routed = [x for x in R if (x != 10000)]
        avgR = R.mean().item()
        avgRoutedR = sum(R_routed).item()/len(R_routed) if len(R_routed) > 0 else 10000
        percRouted = len(R_routed)*100/self.n_batch_test_size
        time = time/self.n_batch_test_size
        if self.csv is not None:
            if not os.path.isfile(self.csv['test']):
                with open(self.csv['test'], 'w', newline='') as file:
                    csv.writer(file).writerow(["step", "sampling", "AvgRoutedR", "AvgR", "AvgRouted%", "AvgTime"])
            with open(self.csv['test'], 'a', newline='') as file:
                csv.writer(file).writerow([self.n_epoch, sample_count, avgRoutedR, avgR, percRouted, time])
        if self.tensor is not None:
            t_board = SummaryWriter(self.tensor)
            t_board.add_scalar('Test/AvgRoutedR', avgRoutedR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgR', avgR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgRouted%', percRouted, global_step = self.n_epoch)
            t_board.add_scalar('Test/Time', time, global_step = self.n_epoch)
        print("Epoch: {0}, Prob size: {1}, avgRoutedR: {2}, percRouted: {3}, time: {4}"
            .format(self.n_epoch, p_size, avgRoutedR, percRouted, time))

    def save(self):
        path = "{0}-{1}".format(self.model_name, self.n_epoch)
        model_dict = self.wrapped_actor.save() #save training details
        model_dict['n_epoch'] = self.n_epoch
        torch.save(model_dict, path)

    # def extract_number(f):
    # s = re.findall("\d+$",f)
    # return (int(s[0]) if s else -1,f)

    def load(self):
        # max(list_of_files,key=extract_number)
        path = "{0}-{1}".format(self.model_name, 0)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.wrapped_actor.load(checkpoint) #load training details
            self.n_epoch = checkpoint['n_epoch']
            print('Loaded with', self.n_epoch, 'epochs.')
        else:
            print('weights not found for', self.model_name)
#
# config = {
#     'model': 'TSP_improve',
#     'environment': 'Improvement',
#     'data_path': 'runs/ImprTransformerCritic2',
#     'save_csv': False,
#     'save_tensor': False,
#     'n_batch': '10',
#     'n_batch_train_size': '512',
#     'n_batch_test_size': '2048',
#     'baseline_type': 'Critic',
#     'n_layers': '2',
#     'n_head': '1',
#     'dim_model': '128',
#     'dim_hidden': '64',
#     'dim_v': '32',
#     'dim_k': '32',
#     'max_grad': '2',
#     'learning_rate': '1e-4',
#     'learning_rate_gamma': '1',
#     't': '1'
# }
# N_EPOCHS = 1
# N_NODES = 7
# def trace(frame, event, arg):
#     print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#     return trace
# sys.settrace(trace)
purpose = sys.argv[1]
config_location = sys.argv[2]
N_EPOCHS = int(sys.argv[3])
N_NODES = int(sys.argv[4])
FILTER_NOSOLPROB = bool(sys.argv[5])
with open(config_location) as json_file:
    config = json.load(json_file)
agent = TrainTest(config=config, routableOnly=FILTER_NOSOLPROB)
print("Number of epochs: {0}".format(N_EPOCHS))
# for epoch in range(0, n_epochs):
agent.load()
for epoch in range(0, N_EPOCHS):
    if purpose == "test":
        sample_count = int(sys.argv[6])
        dataset = sys.argv[7]
        agent.test(N_NODES, prob_path=dataset, sample_count=sample_count)
        # agent.save()
        # agent.load()
    elif purpose == "train":
        agent.train(N_NODES)#, path=file)
        agent.save()
