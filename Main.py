import os
import sys
import csv
import datetime
import numpy as np
from pathlib import Path
import tracemalloc

import torch
from torch.utils.tensorboard import SummaryWriter

from Misc.Environments import Construction, Improvement
from Models.ConstructionPointerNetwork import PtrNetWrapped
from Models.ImprovementTransformer import TSP_improveWrapped
from Models.ConstructionTransformer import TransformerWrapped
from RLAlgorithm.PolicyBasedTrainer import Reinforce

class TrainTest:
    def __init__(self, config):
        print("using config:", config)
        # check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # setup data stuff
        data_path = config['data_path']
        Path(data_path).mkdir(parents=True, exist_ok=True)
        self.csv, self.tensor = None, None
        if bool(config['save_csv']):
            self.csv = {'train': '{0}/train.csv'.format(data_path),
                'test': '{0}/test.csv'.format(data_path)}
            with open(self.csv['train'], 'w') as file:
                csv.writer(file).writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%", "ActorLoss", "BaselineLoss"])
            with open(self.csv['test'], 'w') as file:
                csv.writer(file).writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%"])
        if bool(config['save_tensor']):
            self.tensor = '{0}/tensor'.format(data_path)
        self.model = '{0}/checkpoint'.format(data_path)
        # setup testing and training stuff
        self.n_epoch = 0
        self.n_batch = int(config['n_batch'])
        self.n_batch_train_size = int(config['n_batch_train_size'])
        self.n_batch_test_size = int(config['n_batch_test_size'])
        #=-=-=-=-=-=-=ENVIRONMENTS and TRAINER=-=-=-=-===-=-=-=-=-=-=
        env = {'Construction': Construction(), 'Improvement': Improvement()}.get(config['environment'])
        trainer = Reinforce(device, config)
        #=-=-=-=-===-=-=-=-=-=-=MODEL=-=-=-=-===-=-=-=-=-=-=
        model_type = config['model']
        if model_type == 'PointerNetwork': self.wrapped_actor = PtrNetWrapped(env, trainer, device, config)
        elif model_type == 'Transformer': self.wrapped_actor = TransformerWrapped(env, trainer, device, config)
        elif model_type == 'TSP_improve': self.wrapped_actor = TSP_improveWrapped(env, trainer, device, config)

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
                baseline_loss = actor_loss.item()
            if self.csv is not None:
                with open(self.csv['train'], 'a') as file:
                    csv.writer(file).writerow([i, avgRoutedR, avgR, percRouted, actor_loss, baseline_loss])
            if self.tensor is not None:
                t_board = SummaryWriter(self.tensor)
                t_board.add_scalar('Train/AvgRoutedR', avgRoutedR, global_step = i)
                t_board.add_scalar('Train/AvgR', avgR, global_step = i)
                t_board.add_scalar('Train/AvgRouted%', percRouted, global_step = i)
                t_board.add_scalar('Train/ActorLoss', actor_loss, global_step = i)
                t_board.add_scalar('Train/BaselineLoss', baseline_loss, global_step = i)

    def test(self, p_size, prob_path=None):
        # run tests
        R = self.wrapped_actor.test(self.n_batch_test_size, p_size, path=prob_path)
        R_routed = [x for x in R if (x != 10000)]
        avgR = R.mean().item()
        avgRoutedR = sum(R_routed).item()/len(R_routed) if len(R_routed) > 0 else 10000
        percRouted = len(R_routed)*100/self.n_batch_test_size
        if self.csv is not None:
            with open(self.csv['test'], 'a') as file:
                csv.writer(file).writerow([self.n_epoch, avgRoutedR, avgR, percRouted])
        if self.tensor is not None:
            t_board = SummaryWriter(self.tensor)
            t_board.add_scalar('Test/AvgRoutedR', avgRoutedR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgR', avgR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgRouted%', percRouted, global_step = self.n_epoch)
        print("Epoch: {0}, Prob size: {1}, avgRoutedR: {2}, percRouted: {3}".format(self.n_epoch, p_size, avgRoutedR, percRouted))

    def epoch(self, p_size, prob_path=None):
        self.n_epoch += 1
        # tracemalloc.start()
        # snapshot1 = tracemalloc.take_snapshot()
        self.train(p_size, prob_path)
        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        # print("[ Top 10 differences ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        self.test(p_size, prob_path)
        # snapshot1 = tracemalloc.take_snapshot()
        # top_stats = snapshot1.compare_to(snapshot2, 'lineno')
        # print("[ Top 10 differences ]")
        # for stat in top_stats[:10]:
            # print(stat)

    def save(self):
        model_dict = self.wrapped_actor.save() #save training details
        model_dict['n_epoch'] = self.n_epoch
        torch.save(model_dict, self.model)

    def load(self):
        checkpoint = torch.load(self.model)
        self.wrapped_actor.load(checkpoint) #load training details
        self.n_epoch = checkpoint['n_epoch']+1

def train_thingy(config):
    #TRAINING TESTING DETAILS
    agent = TrainTest(config=config)
    n_epochs = 10
    prob_size = 5
    print("Number of epochs: {0}".format(n_epochs))
    for epoch in range(0, n_epochs):
        agent.epoch(prob_size)#, path=file)
        agent.save()

def test_generalisation(config):
    agent = TrainTest(config=config)
    # generalisation
    for prob_size in [3, 4, 5, 6, 8]:
        agent.load()
        agent.test(prob_size)

def loadConfigFile(path):
    f = open(path, "r")
    line = f.readline()
    while line != '':
        print(line, type(line))
        #do something with line
        line = f.readline()
    f.close()

    config = configparser.ConfigParser()
    config.read(path)
    print("Reading config from path", path)
    config = dict(config['config'])
    return config


path = sys.argv[1]
print(path)
# path = "runs/ImprovementTransformer.cfg"
config = loadConfigFile(path)
train_thingy(config)
