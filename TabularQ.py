from Misc.Environments import Improvement
import random
import numpy as np
import torch

class TrainTabularQ:
    def __init__(self):
        self.q_table = {}
        self.gamma = 0.9
        self.env = Improvement()
        self.T = 3
        self.epsilon = 0.5
        self.alpha = 0.2

    def reset(self):
        self.q_table = {}

    def step(self, initialSolution, x, y):
        x_index = np.where(initialSolution==x)
        y_index = np.where(initialSolution==y)
        newSolution = initialSolution.copy()
        newSolution[x_index] = y
        newSolution[y_index] = x
        return newSolution

    def getNextQ(self, problem_size, solution, problem):
        population = [(x, y) for x in range(problem_size) for y in range(problem_size) if x != y]
        possibleNextStates = [self.step(solution.copy(), x, y) for x, y in population]
        possibleNextQ = [self.evaluate(problem, s) for s in possibleNextStates]
        return possibleNextQ

    def evaluate(self, problem, solution):
        return self.env.evaluate(torch.FloatTensor(problem).unsqueeze(0), torch.FloatTensor(solution).unsqueeze(0)).item()

    # given a problem size and batch size will train the model
    def train(self, batch_size, problem_size, data_loc):
        problems = self.env.load(data_loc, batch_size).numpy()
        solutions = self.env.getInitialSolution(batch_size, problem_size).numpy()
        results = []
        for problem, solution in zip(problems, solutions):
            # print(problem)
            best_so_far = self.evaluate(problem.copy(), solution.copy())
            t = 0
            while t < self.T:
                if random.random() < self.epsilon:
                    population = [(x, y) for x in range(problem_size) for y in range(problem_size) if x != y]
                    (x, y) = random.sample(population, 1)[0]
                else:
                    # best action
                    population = [(x, y) for x in range(problem_size) for y in range(problem_size) if x != y]
                    q = np.array(self.getNextQ(problem_size, solution.copy(), problem.copy()))
                    index = np.argmin(q)
                    x, y = population[index]
                #pick an action
                next_solution = self.step(solution.copy(), x, y)
                cost = self.evaluate(problem.copy(), next_solution.copy())
                best_for_now = min(cost, best_so_far)
                reward = best_for_now - best_so_far
                nextQ = np.array(self.getNextQ(problem_size, next_solution.copy(), problem.copy()))
                target = reward + self.gamma * (min(np.amin(nextQ), best_for_now) - best_for_now) - self.q_table.get(tuple(solution), 0)
                self.q_table[tuple(solution)] = self.q_table.get(tuple(solution), 0) + self.alpha * target
                solution = next_solution
                best_so_far = best_for_now
                t += 1
            # print("Batch", best_so_far)
            results.append(best_so_far)
        return results

    # given a batch size and problem size will test the model
    def test(self, batch_size, problem_size, data_loc=None):
        problems = self.env.load(data_loc, batch_size).numpy()
        solutions = self.env.getInitialSolution(batch_size, problem_size).numpy()
        for problem, solution in zip(problems, solutions):
            ss = []
            results = []
            best_so_far = self.evaluate(problem.copy(), solution.copy())
            ss.append(solution.copy())
            results.append(best_so_far)
            t = 0
            while t < self.T:
                #best action
                population = [(x, y) for x in range(problem_size) for y in range(problem_size) if x != y]
                q = np.array(self.getNextQ(problem_size, solution.copy(), problem.copy()))
                index = np.argmin(q)
                x, y = population[index]
                next_solution = self.step(solution.copy(), x, y)
                ss.append(next_solution)
                cost = self.evaluate(problem.copy(), next_solution.copy())
                results.append(cost)
                best_so_far = min(cost, best_so_far)
                solution = next_solution
                t += 1
            print("Result", best_so_far)
            print("Q table", self.q_table)
            print("Solutions", ss)
            print("Results", results)

test = TrainTabularQ()
print("-=-=-=-=-=-=-=-=TRAIN=-=-=-==-=-==-")
print(test.train(100, 5, "datasets/n5b1(1).pkg"))
print("-=-=-=-=-=-=-=-=TEST=-=-=-==-=-==-")
test.test(1, 5, "datasets/n5b1(1).pkg")
test.reset()
print("-=-=-=-=-=-=-=-=TRAIN=-=-=-==-=-==-")
print(test.train(100, 5, "datasets/n5b1(2).pkg"))
print("-=-=-=-=-=-=-=-=TEST=-=-=-==-=-==-")
test.test(1, 5, "datasets/n5b1(2).pkg")
test.reset()
print("-=-=-=-=-=-=-=-=TRAIN=-=-=-==-=-==-")
print(test.train(100, 5, "datasets/n5b1(3).pkg"))
print("-=-=-=-=-=-=-=-=TEST=-=-=-==-=-==-")
test.test(1, 5, "datasets/n5b1(3).pkg")
test.reset()
print("-=-=-=-=-=-=-=-=TRAIN=-=-=-==-=-==-")
print(test.train(100, 5, "datasets/n5b1(4).pkg"))
print("-=-=-=-=-=-=-=-=TEST=-=-=-==-=-==-")
test.test(1, 5, "datasets/n5b1(4).pkg")
test.reset()
print("-=-=-=-=-=-=-=-=TRAIN=-=-=-==-=-==-")
print(test.train(100, 5, "datasets/n5b1(5).pkg"))
print("-=-=-=-=-=-=-=-=TEST=-=-=-==-=-==-")
test.test(1, 5, "datasets/n5b1(5).pkg")
