import csv
import datetime
import configparser
import os

import torch
from Misc.TSPEnvironment import TSPEnv
from Misc.Environments import Construction, Improvement
from Models.SingleOptPointerNetwork import PtrNet, PntrNetCritic, TrainPointerNetwork
from Models.ImprovementTransformer import TSP_improve, TrainImprovementModel
from torch.utils.tensorboard import SummaryWriter

class TrainTest:
    def __init__(self, folder):
        path = '{0}/configuration.cfg'.format(folder)
        config = configparser.ConfigParser()
        config.read(path)
        print("Reading config from path", path)
        print(dict({key:dict(value) for (key,value) in config.items()}))
        #get some parameters
        general_config = config['general']
        self.train_batch_size = int(general_config['train_batch_size'])
        self.train_batch_epoch = int(general_config['train_batch_epoch'])
        #CREATE MODEL AND ENVIRONMENT
        if general_config['model'] == 'PointerNetwork':
            actor_model = PtrNet(config['actor'])
            critic_model = PntrNetCritic(config['critic']) if general_config['critic'] == 'True' else None
            train = TrainPointerNetwork
        elif general_config['model'] == 'Transformer':
            actor_model = Transformer(config['actor'])
            critic_model = None
            train = None
        elif general_config['model'] == 'TSP_improve':
            actor_model = TSP_improve(config['actor'])
            critic_model = None
            train = TrainImprovementModel
        models = (actor_model, critic_model)
        env = {'Construction': Construction(), 'Improvement': Improvement(), 'TSP': TSPEnv()
            }.get(general_config['environment'])
        # WRAP THEM IN REINFORCEMENT LEARNING ALGORITHM
        self.modelWithRlAlg = train(env, models, config['optimiser'])
        self.folder = folder
        self.date = datetime.datetime.now().strftime('%m%d_%H_%M')
        self.n_epoch = 0

    def train(self, problem_size, data_type="tensor", data_loc=None):
        # create files, setup stuff
        if data_type is "csv":
            csv_path = '{0}/{1}_train_data.csv'.format(self.folder, self.date)
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step", "AvgRoutedR", "AvgR", "AvgRouted%"] + self.mode.additonal_params())
        elif data_type is "tensor":
            tensor_path = '{0}/{1}_tensor'.format(self.folder, self.date)
            t_board = SummaryWriter(tensor_path)
        # setup test data if needed
        if data_loc is not None:
            if isinstance(data_loc, str):
                data_loc = [data_loc for _ in range(self.train_batch_epoch)]
            elif not isinstance(data_loc, list) or len(data_loc) < self.train_batch_epoch:
                raise NotImplementedError
        else:
            data_loc = [None for _ in range(self.train_batch_epoch)]
        # loop through batches
        for i, path in zip(range((self.n_epoch*self.train_batch_epoch), (self.n_epoch*self.train_batch_epoch)+self.train_batch_epoch), data_loc):
            R, loss = self.modelWithRlAlg.train(self.train_batch_size, problem_size, data_loc=path)
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

    def test(self, batch_size, problem_size, data_type="tensor", data_loc=None, override_step=None):
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
            tensor_path = '{0}/{1}_tensor'.format(self.folder, self.date)
            t_board = SummaryWriter(tensor_path)
        # run tests
        R = self.modelWithRlAlg.test(batch_size, problem_size, data_loc=data_loc)
        R_routed = [x for x in R if (x != 10000)]
        avgR = R.mean().item()
        if len(R_routed) != 0:
            avgRoutedR = sum(R_routed)/len(R_routed)
            percRouted = len(R_routed)*100/batch_size
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
            print("Prob size: {0}, avgRoutedR: {1}, percRouted: {2}".format(problem_size, avgRoutedR, percRouted))

    def save(self):
        file_path = "{0}/backup-epoch{1}".format(self.folder, self.n_epoch)
        self.modelWithRlAlg.save(file_path)

    def load(self, epoch):
        file_path = "{0}/backup-epoch{1}".format(self.folder, epoch)
        self.modelWithRlAlg.load(file_path)
        self.n_epoch = epoch

# MODEL
folder = 'runs/ImprovementTsp'
agent = TrainTest(folder)
# agent.load(9)

#TRAINING TESTING DETAILS
n_epochs = 200
test_batch_size = 10000
prob_size = 20
print("Number of epochs: {0}".format(n_epochs))

# agent.test(test_batch_size, prob_size, override_step=0)
for j in range(n_epochs):
    #loop through batches of the test problems
    agent.train(prob_size)#, data_loc="datasets/n5b10(1).pkg")
    agent.test(test_batch_size, prob_size)#, data_loc="datasets/n5b1(1).pkg")
    #, data_loc="datasets/n5b10(1).pkg")
    if j % 20 == 0:
        agent.save()
    print("Finished epoch: {0}".format(j))
# i += 1
# for j in range(i, i+n_epochs):
#     train.train_epoch(n_batch_in_epoch, train_batch_size, test_batch_size, 7, j)
# i += 1
# for j in range(i, i+n_epochs):
#     train.train_epoch(n_batch_in_epoch, train_batch_size, test_batch_size, 9, j)
#
# for prob_size in [3, 4, 5, 6, 8]:
#     agent.load(9)
#     agent.test(1000, prob_size, "console")
