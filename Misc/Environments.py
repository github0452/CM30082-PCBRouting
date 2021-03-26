import Misc.copt as copt
# import copt
# import Baselines
import torch
import random
import numpy as np
import pickle

class Environment:
    def __init__(self):
        pass

    def gen(self, list_size, prob_size, routableOnly=True):
        problems = []
        invalid = 0
        noSol = 0
        while len(problems) < list_size:
            problem = copt.getProblem(prob_size) #generate problem
            # check the problem is valid
            invalidPointNo = len([ 1 for x in problem for y in problem if x != y
                and (np.linalg.norm(np.subtract(x, y)[:2],2) < 30
                or np.linalg.norm(np.subtract(x, y)[2:],2) < 30) ])
            if (invalidPointNo == 0):
                if (routableOnly and len(copt.bruteForce(problem, 1)) != 0) or not routableOnly:
                    problems.append(problem)
                else:
                    noSol += 1
            else:
                invalid += 1
        print("Invalid problem: {0}, No solution problem: {1}".format(invalid, noSol))
        random.shuffle(problems) # randomly shuffling the data to prevent any bias, e.g. if testing later with different problem sizes
        return torch.FloatTensor(problems)

    # problems: torch.Size([100, 5, 4]), orders [5, 100]
    def evaluate(self, problems, orders, additional=None):
        #convert everything to the right format
        n_node = problems.size(1)
        orders = orders.tolist()
        problems = [[tuple(element) for element in problem] for problem in problems.tolist()]
        reward = []
        i = 0
        for problem,order in zip(problems, orders):
            if len(order) >= len(problem):
                eval = copt.evaluate(problem,order)
                if (eval["success"] == 0):
                    reward.append(10000)
                else:
                    reward.append(eval["measure"]/n_node)
            else:
                reward.append(0)
        # print(reward[0:10])
        # print(torch.Tensor(reward[0:10]))
        return torch.Tensor(reward)

    def load(self, path, batch_size):
        problems = pickle.load( open( path, "rb" ))
        problem_count = problems.size(0)
        if problem_count < batch_size: #repeat if needed
            multiply = -(-batch_size // problem_count)
            problems = problems.repeat(multiply, 1, 1)
        problems = problems[:batch_size] #trim
        return problems

class Construction(Environment):
    def initialState(self, problems):
        pass
    
    def step(self, cur_state, step):
        if cur_state is not None:
            next_state = torch.cat((cur_state, step.unsqueeze(dim=1)), dim=1)
        else:
            next_state = step.unsqueeze(dim=1)
        return next_state

    def isDone(self, cur_state, problems):
        return (cur_state.size(1) == problems.size(1))

class Improvement(Environment):
    def __init__(self):
        pass

    def getStartingState(self, list_size, prob_size):
        initial_solution = torch.linspace(0, prob_size-1, steps=prob_size).expand(list_size, prob_size)
        return initial_solution

    def step(self, cur_state, step):
        device = cur_state.device
        step_num = step.clone().cpu().numpy()
        rec_num = cur_state.clone().cpu().numpy()
        for i in range(cur_state.size()[0]):
            loc_of_first = np.where(rec_num[i] == step_num[i][0])[0][0]
            loc_of_second = np.where(rec_num[i] == step_num[i][1])[0][0]
            # temp = rec_num[i][loc_of_first]
            # rec_num[i][loc_of_first] = rec_num[i][loc_of_second]
            # rec_num[i][loc_of_second] = temp
            if( loc_of_first < loc_of_second):
                rec_num[i][loc_of_first:loc_of_second+1] = np.flip(
                        rec_num[i][loc_of_first:loc_of_second+1])
            else:
                temp = rec_num[i][loc_of_first]
                rec_num[i][loc_of_first] = rec_num[i][loc_of_second]
                rec_num[i][loc_of_second] = temp
        return torch.tensor(rec_num).to(device)

    def isDone(self):
        pass

if __name__ == "__main__":
    batchSize = 1
    seqLen = 3
    env = Construction()
    #experimenting with steps
    # problems = torch.Tensor(env.genProblems(batchSize, seqLen))
    # state = None
    # for x in range(seqLen):
    #     step = torch.full((batchSize,1),x).squeeze()
    #     next_state = env.nextState(state, step)
    #     reward = env.evaluate(problems, next_state)
    #     print("state = ", state)
    #     print("next_state = ", next_state)
    #     print("Reward = ", reward)
    #     state = next_state
    # experimenting with saving files
    # for i in range(1, 11):
    #     file = "n3b1({0}).pkg".format(i)
    #     problems = torch.Tensor(env.gen(batchSize, seqLen))
    #     print([(solution['order'], solution['measure']) for solution in copt.bruteForce([[tuple(element) for element in problem] for problem in problems.tolist()][0])])
    #     pickle.dump( problems, open( file, "wb" ) )
    #     problemsv2 = pickle.load( open( file, "rb" ) )
    #     print(problemsv2[0:5])
    #check the loading works
    # problems = pickle.load( open( "Misc/n5b1000(1).pkg", "rb" ))
    # print(problems.size())
    # problem_count = problems.size(0)
    # if problem_count < batchSize: #repeat if needed
    #     multiply = -(-batchSize // problem_count)
    #     problems = problems.repeat(multiply, 1, 1)
    # problems = problems[:batchSize] #trim
    # print(problems.size())
#CONSTRUCTION PROBLEM
