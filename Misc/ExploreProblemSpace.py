#!/usr/bin/env python
# coding: utf-8

#Check python version
from platform import python_version
print("Python version: ", python_version())
assert python_version() == "3.7.9"

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
from Environments import Environment

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

# time taken
# solution quality
# solution % routed
class Baselines:
    def __init__(self):
        self.device = None

    def bruteForce(self, problems, earlyExit=0):
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

    def randomSampling(self, problems):
        solutions = []
        for problem in problems:
            seqLen = len(problem)
            order = np.random.permutation(seqLen).tolist()
            eval = copt.evaluate(problem,order)
            solution = (eval['success'], eval['nRouted'], eval['order'], eval['measure'])
            solutions.append(solution)
        return solutions

    def NN(self, problems):
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

    def RoutableRRHillClimbing(self, problems, numRestarts=1):
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

    def simulatedAnnealing(self, problems):
        pass

    def tabuSearch(self, problems):
        pass


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

batchSize = 5
prob_size = 7
for i in range(batchSize):
    # print("=-=-=-=-=Problems ", averageCount, "=-=-=-=-=-=")
    problem = copt.getProblem(prob_size) #generate problem
    # check the problem is valid
    invalidPointNo = len([ 1 for x in problem for y in problem if x != y
        and (np.linalg.norm(np.subtract(x, y)[:2],2) < 30
        or np.linalg.norm(np.subtract(x, y)[2:],2) < 30) ])
    if (invalidPointNo == 0):
        results = copt.bruteForce(problem)
        results = [(x['measure'], x['order']) for x in results]
        print(results)
    else:
        print("invalid")
