class Reinforce:


class A2C:
    def __init__(self, model):
        self.model = model

    def train(self, n_batch, p_size, data_loc=None):
        reward, probs = self.model(n_batch, p_size, data_loc=data_loc)
        advantage = reward - baseline.detach()
        logprobs = 0
        for prob in probs:
            logprob = torch.log(prob)
            logprobs += logprob
        logprobs[logprobs < -1000] = 0.
        reinforce = (advantage * logprobs)
        actor_loss = reinforce.mean()
        # update the weights using optimiser
        self.M_actor_optim.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.M_actor.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.M_actor_optim.step() # update weights
        self.M_actor_scheduler.step()
        return actual_R, {'actor_loss': actor_loss, 'critic_loss': critic_loss}
