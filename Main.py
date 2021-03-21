import csv
import datetime
import configparser
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from Misc.Environments import Construction, Improvement
from Models.ConstructionPointerNetwork import PtrNetWrapped
from Models.ImprovementTransformer import TSP_improveWrapped
from Models.Transformer import TransformerWrapped
from RLAlgorithm.PolicyBasedTrainer import Reinforce, A2C

class TrainTest:
    def __init__(self, folder):
        path = '{0}/configuration.cfg'.format(folder)
        config = configparser.ConfigParser()
        config.read(path)
        print("Reading config from path", path)
        print(dict({key:dict(value) for (key,value) in config.items()}))
        # load some parameters
        general_config = config['general']
        self.train_batch_size = int(general_config['train_batch_size'])
        self.train_batch_epoch = int(general_config['train_batch_epoch'])
        # generate some parameters
        self.folder = folder
        self.date = datetime.datetime.now().strftime('%m%d_%H_%M')
        self.n_epoch = 0
        #=-=-=-=-=-=-=ENVIRONMENTS=-=-=-=-===-=-=-=-=-=-=
        env = {'Construction': Construction(), 'Improvement': Improvement()
            }.get(general_config['environment'])
        #=-=-=-=-=-=-=TRAINER=-=-=-=-===-=-=-=-=-=-=
        if general_config['trainer'] == 'A2C':
            trainer = A2C(config['critic'])
        elif general_config['trainer'] == 'REINFORCE':
            trainer = Reinforce(config['baseline'])
        #=-=-=-=-===-=-=-=-=-=-=MODEL=-=-=-=-===-=-=-=-=-=-=
        if general_config['model'] == 'PointerNetwork':
            self.wrapped_actor = PtrNetWrapped(env, trainer, config['actor'], config['optimiser'])
        elif general_config['model'] == 'TSP_improve':
            self.wrapped_actor = TSP_improveWrapped(env, trainer, config['actor'], config['optimiser'])
        elif general_config['model'] == 'Transformer':
            self.wrapped_actor = TransformerWrapped(env, trainer, config['actor'], config['optimiser'])

    def train(self, p_size, data_type="tensor", path=None):
        # create files, setup stuff for SAVING DATA
        if data_type is "csv":
            csv_path = '{0}/{1}_train_data.csv'.format(self.folder, self.date)
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%"] + self.mode.additonal_params())
        elif data_type is "tensor":
            tensor_path = '{0}/{1}_running50epoch_tensor'.format(self.folder, self.date)
            t_board = SummaryWriter(tensor_path)
        # setup test data location if needed
        if path is not None:
            if isinstance(path, str):
                path = [path for _ in range(self.train_batch_epoch)]
            elif not isinstance(path, list) or len(path) < self.train_batch_epoch:
                raise NotImplementedError
        else:
            path = [None for _ in range(self.train_batch_epoch)]
        # loop through batches
        for i, path in zip(range((self.n_epoch*self.train_batch_epoch), (self.n_epoch*self.train_batch_epoch)+self.train_batch_epoch), path):
            #pass it through reinforcement learning algorithm to train
            R, loss = self.wrapped_actor.train(self.train_batch_size, p_size, path=path)
            R_routed = [x for x in R if (x != 10000)]
            avgR = R.mean().item()
            if len(R_routed) != 0:
                avgRoutedR = sum(R_routed)/len(R_routed)
                percRouted = len(R_routed)*100/self.train_batch_size
            else:
                avgRoutedR = 10000
                percRouted = 0.
            if data_type is "csv":
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, avgRoutedR, avgR, percRouted] + list(loss))
            elif data_type is "tensor":
                t_board.add_scalar('Train/AvgRoutedR', avgRoutedR, global_step = i)
                t_board.add_scalar('Train/AvgR', avgR, global_step = i)
                t_board.add_scalar('Train/AvgRouted%', percRouted, global_step = i)
                for k in loss.keys():
                    t_board.add_scalar('Train/{0}'.format(k), loss[k], global_step = i)
        self.n_epoch += 1

    def test(self, n_batch, p_size, data_type="tensor", path=None, override_step=None):
        if override_step is not None:
            epoch = override_step
        else:
            epoch = self.n_epoch
        date = datetime.datetime.now().strftime('%m%d_%H_%M')
        # create files, setup stuff
        if data_type is "csv":
            csv_path = '{0}/{1}_test_data.csv'.format(self.folder, self.date)
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%"])
        elif data_type is "tensor":
            tensor_path = '{0}/{1}_running50epoch_tensor'.format(self.folder, self.date)
            t_board = SummaryWriter(tensor_path)
        # run tests
        R = self.wrapped_actor.test(n_batch, p_size, path=path)
        R_routed = [x for x in R if (x != 10000)]
        avgR = R.mean().item()
        if len(R_routed) != 0:
            avgRoutedR = sum(R_routed)/len(R_routed)
            percRouted = len(R_routed)*100/n_batch
        else:
            avgRoutedR = 10000
            percRouted = 0.
        if data_type is "csv":
            writer.writerow([i, avgRoutedR, avgR, percRouted])
        elif data_type is "tensor":
            t_board.add_scalar('Test/AvgRoutedR', avgRoutedR, global_step = epoch)
            t_board.add_scalar('Test/AvgR', avgR, global_step = epoch)
            t_board.add_scalar('Test/AvgRouted%', percRouted, global_step = epoch)
        elif data_type is "console":
            print("Prob size: {0}, avgRoutedR: {1}, percRouted: {2}".format(p_size, avgRoutedR, percRouted))

    def save(self):
        file_path = "{0}/backup-epoch{1}".format(self.folder, self.n_epoch)
        model_dict = self.wrapped_actor.trainer.save() #save training details
        torch.save(model_dict, file_path)

    def load(self, epoch):
        file_path = "{0}/backup-epoch{1}".format(self.folder, epoch)
        checkpoint = torch.load(file_path)
        self.wrapped_actor.trainer.load(checkpoint) #load training details
        self.n_epoch = epoch

# MODEL
folder = 'runs/Transformer'
agent = TrainTest(folder)
# agent.load(11)

#TRAINING TESTING DETAILS
n_epochs = 50
test_n_batch = 1000
prob_size = 5
print("Number of epochs: {0}".format(n_epochs))
file = "datasets/n{0}b1({1}).pkg".format(prob_size, 10)
agent.test(test_n_batch, prob_size, override_step=0)#, path=file)
for j in range(n_epochs):
    #loop through batches of the test problems
    agent.train(prob_size)#, path=file)
    agent.test(test_n_batch, prob_size)#, path=file)
    print("Finished epoch: {0}".format(j))
    agent.save()
agent.save()
# i += 1
# for j in range(i, i+n_epochs):
#     train.train_epoch(n_batch_in_epoch, train_batch_size, test_n_batch, 7, j)
# i += 1
# for j in range(i, i+n_epochs):
#     train.train_epoch(n_batch_in_epoch, train_batch_size, test_n_batch, 9, j)
#
# for prob_size in [3, 4, 5, 6, 8]:
#     agent.load(10)
#     agent.test(1000, prob_size, "console")
