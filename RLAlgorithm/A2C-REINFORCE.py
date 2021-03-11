class ExpMovingAvg:
    def __init__(self):
        self.TRACK_critic_exp_mvg_avg = None

    def update(self, newReward):
        if self.TRACK_critic_exp_mvg_avg is None:
            self.TRACK_critic_exp_mvg_avg = newReward.detach().mean()
        else:
            self.TRACK_critic_exp_mvg_avg = (TRACK_critic_exp_mvg_avg.bseline * 0.9) + (newReward.detach().mean() * 0.1)
        return self.TRACK_critic_exp_mvg_avg

class Reinforce:
    def __init__(self, model, baseline=None):
        self.model = actor
        self.optim = None
        self.scheduler = None
        self.baseline = baseline

    def train(self, problems):
        reward, probs = self.model(problemsc)
        if baseline is None:
            advantage = reward
        else:
            advantage = reward - self.baseline.update(reward).detach()
        logprobs = torch.log(probs)
        reinforce = (advantage * logprobs)
        actor_loss = reinforce.mean()
        # update the weights using optimiser
        self.M_actor_optim.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.M_actor.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.M_actor_optim.step() # update weights
        self.M_actor_scheduler.step()
        return actual_R, {'actor_loss': actor_loss}

    def additonal_params(self):
        return ['actor_loss']

class A2C:
    def __init__(self, actor, algorithm_config, critic_config):
        self.actor = actor
        self.critic = PntrNetCritic(config['critic'])
        self.actor_optimizer = optim.Adam(self.M_actor.parameters(), lr=float(config['actor_lr']))
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.M_actor_optim, step_size=1, gamma=float(config['actor_lr_gamma']))
        self.critic_optimizer = optim.Adam(self.M_critic.parameters(), lr=float(config['critic_lr']))
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.M_critic_optim, step_size=1, gamma=float(config['critic_lr_gamma']))
        self.critic_mse_loss = nn.MSELoss()

    def additonal_params(self):
        return ['actor_loss', 'critic_loss']

    def train(self, problems):
        #pass through model and additional method to get reward and probs
        reward, probs = self.actor.train(problems)
        critic_reward = self.critic(problems.detach())
        # train critic
        critic_loss = self.critic_mse_loss(reward, critic_reward)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.critic_optimizer.step()
        self.critic_scheduler.step() #DO IN THE LATER LAYER?
        # train actor
        advantage = reward - critic_reward.detach()
        logprobs = torch.log(probs)
        reinforce = (advantage * logprobs)
        actor_loss = reinforce.mean()
        # update the weights using optimiser
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.actor_optimizer.step() # update weights
        self.actor_scheduler.step() #TOD IN LATER LAYER?
        return actual_R, {'actor_loss': actor_loss, 'critic_loss': critic_loss}
