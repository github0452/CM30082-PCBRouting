import os
import sys, getopt
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
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
        env = {'con': Construction(routableOnly), 'imp': Improvement(routableOnly)}
        trainer = Reinforce(self.device, config)
        #=-=-=-=-===-=-=-=-=-=-=MODEL=-=-=-=-===-=-=-=-=-=-=
        model_type = config['model']
        if model_type == 'PointerNetwork': con_model = PtrNetWrapped(env['con'], trainer, self.device, config)
        elif model_type == 'Transformer': con_model = TransformerWrapped(env['con'], trainer, self.device, config)
        imp_model = TSP_improveWrapped(env['imp'], trainer, self.device, config)
        self.wrapped_actor = { 'con': con_model,  'imp': imp_model   }
        self.load()

    """
    Test Model
    """
    def test(self, p_size, prob_path, sample_count=1):
        # run tests
        R, time, state = self.wrapped_actor['con'].test(self.n_batch_test_size, p_size, path=prob_path, sample_count=1)
        R, time, state = self.wrapped_actor['imp'].test(self.n_batch_test_size, p_size, path=prob_path, sample_count=(sample_count-1), start_s=state)
        R_routed = [x for x in R if (x != 10000)]
        avgR = R.mean().item()
        avgRoutedR = sum(R_routed).item()/len(R_routed) if len(R_routed) > 0 else 10000
        percRouted = len(R_routed)*100/len(R)
        time = time/len(R)
        if self.csv is not None:
            if not os.path.isfile(self.csv['test']):
                with open(self.csv['test'], 'w', newline='') as file:
                    csv.writer(file).writerow(["step", "p_size", "sampling", "AvgRoutedR", "AvgR", "AvgRouted%", "AvgTime"])
            with open(self.csv['test'], 'a', newline='') as file:
                csv.writer(file).writerow([self.n_epoch, p_size, sample_count, avgRoutedR, avgR, percRouted, time])
        if self.tensor is not None:
            t_board = SummaryWriter(self.tensor)
            t_board.add_scalar('Test/AvgRoutedR', avgRoutedR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgR', avgR, global_step = self.n_epoch)
            t_board.add_scalar('Test/AvgRouted%', percRouted, global_step = self.n_epoch)
            t_board.add_scalar('Test/Time', time, global_step = self.n_epoch)
        print("Epoch: {0}, Prob size: {1}, avgRoutedR: {2}, percRouted: {3}, time: {4}"
            .format(self.n_epoch, p_size, avgRoutedR, percRouted, time))

    """
    Load checkpoint
    """
    def load(self):
        # load construction model
        checkpoint = torch.load("{}-con".format(self.model_name), map_location=self.device)
        self.wrapped_actor['con'].load(checkpoint, ignore_trainer=True) #load training details
        self.n_epoch = checkpoint['n_epoch']
        print('Construction model loaded with', self.n_epoch, 'epochs.')
        # load improvement model
        checkpoint = torch.load("{}-imp".format(self.model_name), map_location=self.device)
        self.wrapped_actor['imp'].load(checkpoint, ignore_trainer=True) #load training details
        self.n_epoch = checkpoint['n_epoch']
        print('Improvement model loaded with', self.n_epoch, 'epochs.')

# get arguments
print(sys.argv)
config_location = sys.argv[1]
folder_location = sys.argv[2]
n_epochs = int(sys.argv[3])
p_size = int(sys.argv[4])
filter_no_sol_prob = bool(sys.argv[5])
sample_count = int(sys.argv[6])
dataset = sys.argv[7]
print("Number of epochs: {0}".format(n_epochs))
# load config
with open(config_location) as json_file:
    config = json.load(json_file)
    config['data_path'] = folder_location #overwriting
# run stuff
print("Running with config: ", config)

agent = TrainTest(config=config, routableOnly=filter_no_sol_prob)
for epoch in range(n_epochs):
    agent.test(p_size, dataset, sample_count=sample_count)
