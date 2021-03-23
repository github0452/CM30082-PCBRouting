#!/usr/bin/env python
# coding: utf-8

# **Checklist**
#  * setup openAI Gym[x]
#  * setup pytorch[x]
#  * build basic pointer network on CPU [ ]
#  * build train function (using policy-based method reinforce) on CPU on construction problem [ ]
#  * make it run on the GPU on construction problem [ ]
#  * do so for the improvement problem [ ]
#  * build train funciton (using DQL)

# **What i want to achieve**
# 1. experiment with different reinforcement learning algorithms
#    * policy-based method reinforcement
#    * value-based Deep Q Learning
#    * MCTS
# 2. generalise to different problem sizes
# 3. compare different problem structures (constrution, improvement)
# 4. compare to baselines

#Check python version
from platform import python_version
print("Python version: ", python_version())
assert python_version() == "3.7.9"


# **Setup OpenAI Gym**
#
# Want to setup openAI environment

#imports
import copt
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

# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECK_GPU_DETAILS = False
if CHECK_GPU_DETAILS:
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    get_ipython().system(' nvcc --version')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

#SAVE AND LOAD MODELS
#CUSTOM DATASETS
#EINSUM - tensors
#Pytorch quick tips - progress bar, calculate mean, weight initialisation, learning rate scheduler

print(torch.__version__)



# solutions: (isSuccessfullyRouted, numPointsRouted, order, measure)
def metrics(problems, TENSORBOARD_write, solutionFunction, *parameters):
    batchSize = len(problems)
    seqLen = len(problems[0])
    start = time.perf_counter()
    solutions = solutionFunction(problems, *parameters)
    end = time.perf_counter()
    routedSolutions = [x for x in solutions if x[0] == 1]
    percRouted = len(routedSolutions)*100/len(solutions)
    if percRouted > 0:
        routedSolutionQual = sum([x[3] for x in routedSolutions])/len(routedSolutions)
        routedSolutionQualPerElement = routedSolutionQual/seqLen
    else:
        routedSolutionQual = 10000
        routedSolutionQualPerElement = 10000
    timeTaken = (end-start)/batchSize
    TENSORBOARD_write.add_hparams({'method': solutionFunction.__name__, 'batchSize': batchSize, 'problem size': seqLen},
        {'avgRouted': percRouted, 'avgRReward': routedSolutionQual, 'avgRRewardPerElement': routedSolutionQualPerElement, 'time': timeTaken})
    return percRouted, routedSolutionQual, routedSolutionQualPerElement, timeTaken

def runMetrics(batch_size, n_node, TENSORBOARD_write):
    baseline = Baselines()
    problems = Environments.genProblemList(batch_size, n_node)
    print("=-=-=-=-=-=Batch size: ", batch_size, ",seq len: ", n_node, "=-=-=-=-=-=")
    print("Brute force", metrics(problems, TENSORBOARD_write, baseline.bruteForce))
    print("Random sampling", metrics(problems, TENSORBOARD_write, baseline.randomSampling))
    print("Nearest neighbour", metrics(problems, TENSORBOARD_write, baseline.NN))
    print("Routable random hc metrics with 1 restart(s)", metrics(problems, TENSORBOARD_write, baseline.RoutableRRHillClimbing))
    print("Routable random hc metrics with 5 restart(s)", metrics(problems, TENSORBOARD_write, baseline.RoutableRRHillClimbing, 5))

def test(batchSize, seqLen):
    routingDistribution = {}
    average = 0
    averageCount = 0
    possibleSolutions = math.factorial(seqLen)
    numSuccessiveRouting = 0
    numPossibleSolutions = 0
    # miniBatch = 10000
    # batchCount = int(batchSize/10000)
    # for i in range(batchCount):
        # print("Batch ", i)
    problems = Environments.genProblemList(batchSize, seqLen)
    for problem in problems:
        # print("=-=-=-=-=Problems ", averageCount, "=-=-=-=-=-=")
        results = copt.bruteForce(problem)
        # (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
        successiveRouting = [eval['measure']/seqLen for eval in results]
        percRouted = (len(successiveRouting)/possibleSolutions)*100
        numSuccessiveRouting += len(successiveRouting)
        numPossibleSolutions += possibleSolutions
        counts = Counter(successiveRouting)
        # update average
        averageCount += 1
        average += (percRouted-average)/averageCount
        routingDistribution.update(counts)
        # print("Routing dict: ",counts)
        # print("Percentage routed: ",percRouted,"%")
    print("=-=-=-=-=Average for problem size", seqLen,"batch size",batchSize,"=-=-=-=-=-=-=")
    print("Average routed: ",average,"%")
    keys = list(routingDistribution.keys())
    keys.sort()
    values = [routingDistribution[key] for key in keys]
    averageValue = sum([value*key for value,key in zip(values, keys)])/sum(values)
    print("Number of solutions ", numSuccessiveRouting)
    print("Number of possible solutions ", numPossibleSolutions)
    print("Average value: ",averageValue)
    plt.plot(keys, values)
    # plt.bar(routingDistribution.keys(), routingDistribution.values())
    plt.show()
#metrics
# - quality of routed solutions
# - % routed

# main only code
# if __name__ == "__main__":
#     TENSOR_BOARD_DATA = f'runs/MNIST/SolutionSpace'
#     TENSORBOARD_write = SummaryWriter(TENSOR_BOARD_DATA)
#     routed_perc = [92.37, 74.498, 60.192, 39.123, 18.478, 6.387, 1.383, 0.237, 0.025]
#     num_solution = [2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
#     average_value = [1074.42, 1089.88, 1105.90, 1122.35, 1136.87, 1147.91, 1154.10, 1159.41, 1164.97]
#     step = 8
#     for i in range(7, len(routed_perc)):
#         print(step, routed_perc[i], num_solution[i], average_value[i])
#         TENSORBOARD_write.add_scalar('SolutionSpace/AverageRouted%', routed_perc[i], global_step = step)
#         TENSORBOARD_write.add_scalar('SolutionSpace/SolutionSpaceSize', num_solution[i], global_step = step)
#         TENSORBOARD_write.add_scalar('SolutionSpace/AverageValuePerPointForRoutedSolutions', average_value[i], global_step = step)
#         step += 1
    # Counter(words).keys() # equals to list(set(words))
    # Counter(words).values() # counts the elements' frequency
    # runMetrics(100, 2, TENSORBOARD_write)
    # runMetrics(100, 3, TENSORBOARD_write)
    # runMetrics(100, 4, TENSORBOARD_write)
    # runMetrics(100, 5, TENSORBOARD_write)
    # runMetrics(100, 6, TENSORBOARD_write)
    # runMetrics(100, 7, TENSORBOARD_write)
    # runMetrics(100, 8, TENSORBOARD_write)
    # runMetrics(100, 9, TENSORBOARD_write)
    # runMetrics(100, 10, TENSORBOARD_write)
