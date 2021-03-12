class MonteCarlo:
    def __init__(self, rlAlg):
        self.rlAlg = rlAlg

    def train(self, problems):
        reward, probs = self.rlAlg.actor.train(problems)
        R, loss = self.rlAlg.train(problems, reward, probs)
        return R, loss

class TemporalDifference:
    def __init__(self, rlAlg):
        self.rlAlg = rlAlg

    def train(self, problems):
        problems = problems.to(self.device)
        batch_size, problem_size, _ = problems
        state = self.env.getStartingState(batch_size, problem_size).to(self.device)
        #tracking information about the sequence
        # best_so_far = self.env.evaluate(problems, state) #[batch_size] training relevant #WRAPPERs
        # initial_reward = best_so_far.clone() #INFO #WRAPPER not needed
        action_history = [] #INFO
        state_history = [state]
        reward_history = [] #INFO
        total_reward = 0
        total_loss = 0
        action = None
        # run through the model
        self.rlAlg.actor.train()
        while self.env.isDone():
            likelihood, action = self.rlAlg.actor(problems, state, action)
            next_state = self.env.step(state, action)
            reward = self.env.evaluate(problems, next_state, best_so_far)
            #save things
            reward_history.append(cost)
            action_history.append(action)
            state_history.append(next_state)
            #update for next iteration
            state = next_state
            R, loss = self.rlAlg.train(problems, reward, probs)
            total_reward += R
            total_loss += loss
        return total_reward, total_loss
