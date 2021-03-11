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
