#!/usr/bin/env python
# coding: utf-8

#Check python version
from platform import python_version
print("Python version: ", python_version())
# assert python_version() == "3.7.9"

#imports
import copt
import Environments
import copy
from time import perf_counter
import itertools

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
from Environments import Environment

# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda:5")
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

# time taken
# solution quality
# solution % routed
class Baseline:
    def bruteForce(problems, earlyExit=0):
        solutions = []
        for problem in problems:
            results = copt.bruteForce(problem, earlyExit)
            if len(results) != 0:
                best = results[0]
                solution = (best['success'], best['nRouted'], best['order'], best['measure'])
            else:
                solution = (0, -1, -1, 0.0)
            solutions.append(solution)
        return solutions

    def randomSampling(problems):
        solutions = []
        for problem in problems:
            seqLen = len(problem)
            order = np.random.permutation(seqLen).tolist()
            eval = copt.evaluate(problem,order)
            solution = (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
            solutions.append(solution)
        return solutions

    def NN(problems):
        solutions = []
        for problem in problems:
            seqLen = len(problem)
            order = []
            for elementNo in range(seqLen):
                options = []
                for i in range(seqLen):
                    if i not in order:
                        tempOrder = order.copy()
                        tempOrder.append(i)
                        eval = copt.evaluate(problem, tempOrder)
                        options.append((i, eval['success'], eval['measure']))
                routedOptions = [x for x in options if x[1] == 1]
                routedOptions.sort(key=lambda tup: tup[2], reverse=False)
                if (len(routedOptions) != 0):
                    order.append(routedOptions[0][0])
                else:
                    order.append(options[0][0])
            eval = copt.evaluate(problem,order)
            solution = (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
            solutions.append(solution)
        return solutions

    def RoutableRRHillClimbing(problems, numRestarts=1):
        solutions = []
        # for each problem
        for problem in problems:
            restartSolutions = []
            # restart this number of times
            for restartCount in range(numRestarts):
                seqLen = len(problem)
                # need to first find a initial routable solution
                initialSolutions = copt.bruteForce(problem, 1)
                if len(initialSolutions) != 0:
                    order = copt.bruteForce(problem, 1)[0]['order']
                    eval = copt.evaluate(problem, order)
                    oldMeasure = eval['measure']
                    improvement = True
                    oldMeasure = 0
                    while improvement == True:
                        #consider all the options
                        options = []
                        for i,j in itertools.product(range(seqLen), range(seqLen)):
                            if i != j:
                                tempOrder = order.copy()
                                tempOrder[i],tempOrder[j] = order[j],order[i]
                                eval = copt.evaluate(problem, tempOrder)
                                options.append((i, eval['success'], eval['measure']))
                        routedOptions = [x for x in options if x[1] == 1]
                        routedOptions.sort(key=lambda tup: tup[2], reverse=False)
                        if (len(routedOptions) != 0):
                            newMeasure = routedOptions[0][2]
                            if (newMeasure < oldMeasure):
                                swapElements = routedOptions[0]
                                i,j = swapElements[0],swapElements[1]
                                order[i],order[j] = order[j],order[i]
                                oldMeasure = newMeasure
                            else:
                                improvement=False
                        else:
                            improvement = False
                    eval = copt.evaluate(problem,order)
                    solution = (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
                else:
                    solution = (0, -1, -1, 0.0)
                restartSolutions.append(solution)
            restartSolutions.sort(key=lambda tup: tup[3], reverse=False)
            bestRestartSolution = restartSolutions[0]
            solutions.append(bestRestartSolution)
        return solutions

# solutions: (isSuccessfullyRouted, numPointsRouted, order, measure)
def metrics(data_path, solutionFunction, *parameters):
    env = Environment()
    problems = env.load("datasets/n5b5120.pkg")
    batchSize, seqLen = len(problems), len(problems[0])
    torch.cuda.synchronize(device)
    stime = perf_counter()
    solutions = solutionFunction(problems, *parameters)
    torch.cuda.synchronize(device)
    timeTaken = perf_counter() - stime
    R_routed = [x for x in solutions if x[0] == 1]
    avgR = sum(x[3] for x in solutions)/len(solutions)
    avgRoutedR = sum(x[3] for x in R_routed)/len(R_routed) if len(R_routed) > 0 else 10000
    percRouted = len(R_routed)*100/len(solutions)
    print("Method: ", solutionFunction.__name__)
    print("batchSize:", batchSize, ", problem size:", seqLen)
    print("avgReward: {0}, avgRoutedReward: {1}, percRouted: {2}, timeTaken: {3}".format(avgR, avgRoutedR, percRouted, timeTaken))
    file="ExploreProblemSpace.csv"
    if not os.path.isfile(file):
        with open(file, 'w', newline='') as file:
            csv.writer(file).writerow(["Function", "data_path", "AvgRoutedR", "AvgR", "AvgRouted%", "AvgTime"])
    with open(file, 'a', newline='') as file:
        csv.writer(file).writerow([solutionFunction.__name__, data_path, avgR, avgRoutedR, percRouted, time])

def runMetrics(data_path):
    print("Brute force", metrics(data_path, Baseline.bruteForce))
    print("Random sampling", metrics(data_path, Baseline.randomSampling))
    print("Nearest neighbour", metrics(data_path, Baseline.NN))
    print("Routable random hc metrics with 1 restart(s)", metrics(data_path, Baseline.RoutableRRHillClimbing))
    print("Routable random hc metrics with 5 restart(s)", metrics(data_path, Baseline.RoutableRRHillClimbing, 5))

DATA = ["datasets/n5b5120.pkg", "datasets/n8b5120.pkg"]
for data_path in DATA:
    runMetrics(data_path)

# def test(batchSize, seqLen):
#     routingDistribution = {}
#     average = 0
#     averageCount = 0
#     possibleSolutions = math.factorial(seqLen)
#     numSuccessiveRouting = 0
#     numPossibleSolutions = 0
#     # miniBatch = 10000
#     # batchCount = int(batchSize/10000)
#     # for i in range(batchCount):
#         # print("Batch ", i)
#     problems = Environments.gen(batchSize, seqLen)
#     for problem in problems:
#         # print("=-=-=-=-=Problems ", averageCount, "=-=-=-=-=-=")
#         results = copt.bruteForce(problem)
#         # (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
#         successiveRouting = [eval['measure']/seqLen for eval in results]
#         percRouted = (len(successiveRouting)/possibleSolutions)*100
#         numSuccessiveRouting += len(successiveRouting)
#         numPossibleSolutions += possibleSolutions
#         counts = Counter(successiveRouting)
#         # update average
#         averageCount += 1
#         average += (percRouted-average)/averageCount
#         routingDistribution.update(counts)
#         # print("Routing dict: ",counts)
#         # print("Percentage routed: ",percRouted,"%")
#     print("=-=-=-=-=Average for problem size", seqLen,"batch size",batchSize,"=-=-=-=-=-=-=")
#     print("Average routed: ",average,"%")
#     keys = list(routingDistribution.keys())
#     keys.sort()
#     values = [routingDistribution[key] for key in keys]
#     averageValue = sum([value*key for value,key in zip(values, keys)])/sum(values)
#     print("Number of solutions ", numSuccessiveRouting)
#     print("Number of possible solutions ", numPossibleSolutions)
#     print("Average value: ",averageValue)
#     plt.plot(keys, values)
#     # plt.bar(routingDistribution.keys(), routingDistribution.values())
#     plt.show()
