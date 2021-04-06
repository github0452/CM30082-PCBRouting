import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

from Models.GeneralLayers import GraphEmbedding, SkipConnection, MultiHeadAttention, FeedForward, TransformerEncoderL

class Transformer(nn.Module):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        dim_model = int(model_config['dim_model'])
        dim_hidden = int(model_config['dim_hidden'])
        dim_k = int(model_config['dim_k'])
        dim_v = int(model_config['dim_v'])
        self.L_embedder = GraphEmbedding(dim_model, usePosEncoding=False)
        self.L_encoder = nn.Sequential(*(TransformerEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v) for _ in range(n_layers)))
        self.L_graph_context = nn.Linear(dim_model, dim_model, bias=False)
        self.L_node_context = nn.Linear(dim_model, dim_model, bias=False)
        self.L_decoder_attention = MultiHeadAttention(dim_model*3, dim_k, dim_v, n_head, dim_k_input=dim_model, dim_v_input=dim_model)
        self.W_v_f = nn.Parameter(torch.FloatTensor(size=[1, 1, dim_model]).uniform_())
        self.W_v_l = nn.Parameter(torch.FloatTensor(size=[1, 1, dim_model]).uniform_())

    def forward(self, problems, sampling=True):
        n_batch, n_node, _ = problems.size()
        #ENCODE PROBLEM - invariant so must be problem
        embedding = self.L_embedder(problems, None)
        node_context = self.L_encoder(embedding) # [n_batch, n_node, dim_model]
        graph_context = self.L_graph_context(node_context.mean(dim=1, keepdim=True)) # [n_batch, 1, dim_model]
        #setup initial variables
        first, last = self.W_v_f.repeat(n_batch, 1, 1), self.W_v_l.repeat(n_batch, 1, 1)
        mask = torch.zeros([n_batch, n_node], device=problems.device).bool()
        action_list, action_probs_list = [], [] #action_list and action_probs_list:  (step x [n_batch])
        while (len(action_list) < problems.size(1)):
            context_query = torch.cat([graph_context, last, first], dim=-1)
            q = self.L_decoder_attention(context_query, node_context, node_context, mask=mask)
            #another decoder layer - SINGLE ATTENTION HEAD
            k = self.L_node_context(node_context)
            norm_factor = 1/(q.size(-1)**0.5)
            logits = torch.matmul(q, k.transpose(-2, -1) * norm_factor).squeeze(dim=1)
            logits = logits.tanh()*10 #[n_batch, n_node]
            logits = logits.masked_fill(mask, float('-inf')) #mask stuff
            probs = F.softmax(logits, dim=1) #soft max the probabilities
            actions = probs.multinomial(1).squeeze(1) if sampling else probs.argmax(dim = 1) # pick an action, actions: torch.Size([100])
            # add it to various lists
            action_list.append(actions)
            action_probs_list.append(probs[[x for x in range(len(probs))], actions.data])
            # update for next loop
            mask = mask.scatter(1, actions.unsqueeze(dim=-1), True)
            visit_idx = actions.unsqueeze(-1).repeat(1, 1, node_context.size(-1))
            last = torch.gather(node_context, 1, visit_idx).transpose(0, 1)
            if len(action_probs_list) == 1:
                first = last
        return torch.stack(action_probs_list, dim=1), torch.stack(action_list, dim=1)

class TransformerWrapped:
        # creates the everything around the networks
        def __init__(self, env, trainer, device, model_config, optimizer_config): # hyperparameteres
            self.env = env
            self.trainer = trainer
            self.device = device
            self.actor = Transformer(model_config).to(self.device)
            self.trainer.passIntoParameters(self.actor.parameters(), optimizer_config)

        # given a problem size and batch size will train the model
        def train_batch(self, n_batch, p_size, path=None):
            problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
            # run through the model
            self.actor.train()
            action_probs_list, action_list = self.actor(problems, sampling=True)
            # calculate reward and probability
            reward = self.env.evaluate(problems, action_list).to(self.device)
            probs = action_probs_list.prod(dim=1)
            # use this to train
            loss_dict = self.trainer.train(problems, action_list.unsqueeze(dim=0), reward.unsqueeze(dim=0), probs.unsqueeze(dim=0))
            return reward, loss_dict

        # given a batch size and problem size will test the model
        def test(self, n_batch, p_size, path=None):
            problems = self.env.gen(n_batch, p_size).to(self.device) if (path is None) else self.env.load(path, n_batch).to(self.device) # generate or load problems
            # run through model
            self.actor.eval()
            action_probs_list, action_list = self.actor(problems, sampling=True)
            reward = self.env.evaluate(problems, action_list).to(self.device)
            return reward

        def save(self):
            model_dict = {}
            model_dict['actor_model_state_dict'] = self.actor.actor.state_dict()
            model_dict.update(self.trainer.save())
            return model_dict

        def load(self, checkpoint):
            self.actor.actor.load_state_dict(checkpoint['actor_model_state_dict'])
            self.trainer.load(checkpoint)

# generate problems
# problems = torch.FloatTensor(self.env.genProblems(batch_size, problem_size))
if __name__ == "__main__":
    #__init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout):
    test = Transformer(4, 2, 128, 256, 64, 64)
    testArray = torch.zeros((100, 5, 4))
    testArray2 = torch.zeros((100, 3, 4))
    print(testArray.size())
    testArray = test(testArray, testArray2)
    print(testArray.size())