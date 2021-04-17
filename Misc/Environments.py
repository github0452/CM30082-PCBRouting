import sys
print("Env path", sys.path)

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
        # randomly shuffling the data to prevent any bias, e.g. if testing later with different problem sizes
        random.shuffle(problems)
        return problems

    # problems: torch.Size([100, 5, 4]), orders [5, 100]
    def evaluate(self, problems, orders):
        #convert everything to the right format
        n_node = problems.size(1)
        if torch.is_tensor(orders):
            orders = orders.tolist()
        if torch.is_tensor(problems):
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
        # reward = torch.as_tensor(reward, device=device)
        return reward

    def load(self, path, batch_size=None):
        problems = pickle.load( open( path, "rb" ))
        # if batch_size is not None:
        #     problem_count = len(problems)
        #     if problem_count < batch_size: #repeat if needed
        #         multiply = -(-batch_size // problem_count)
        #         problems = problems.repeat(multiply, 1, 1)
        #     problems = problems[:batch_size] #trim
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

    def getStartingState(self, list_size, prob_size, device):
        initial_solution = torch.linspace(0, prob_size-1, steps=prob_size, device=device).expand(list_size, prob_size)
        return initial_solution

    # cur_state: [n_batch, n_nodes]
    def step(self, cur_state, step):
        step_np = step.cpu().numpy()
        state_np = cur_state.cpu().numpy()
        for action, state in zip(step_np, state_np):
            a_1 = np.where(state == action[0])[0][0]
            a_2 = np.where(state == action[1])[0][0]
            if (a_1 < a_2):
                state[a_1:a_2+1] = np.flip(state[a_1:a_2+1])
            else:
                temp = state[a_1]
                state[a_1] = state[a_2]
                state[a_2] = temp
        state_np = torch.as_tensor(state_np, device=cur_state.device)
        return state_np

    def isDone(self):
        pass

if __name__ == "__main__":
    batchSize = 1
    seqLen = 3
    env = Construction()
    problems = env.gen(5120, 5)
    pickle.dump( problems, open( "n5b5120.pkg", "wb" ) )

    # array = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    # valuesToSwap = torch.tensor([[3, 2], [0, 4]])
    #
    # step_np = valuesToSwap.clone().cpu().numpy()
    # state_np = array.clone().cpu().numpy()
    # state_np = np.insert(state_np, 0, -1, axis=1)
    # for action, state_row in zip(step_np, state_np):
    #     a_1 = np.where(state_row == action[0])[0][0]
    #     state_row[0:a_1-1] = state_row[1:a_1]
    #     state_row[a_1-1] = action[1]
    # # for i in range(array.size(dim=0)):
    # #
    # #     print(np.insert(state_np, [i, a_1], step_np[i][1]))
    # print(torch.tensor(state_np))
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
