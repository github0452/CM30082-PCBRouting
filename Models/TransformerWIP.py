import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Models.GeneralLayers import GraphEmbedding, MultiHeadAttention, FeedForward

class TEncoderL(nn.Module):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.L_MHA = MultiHeadAttention(n_head, dim_model, dim_k, dim_v, dropout=dropout)
        self.L_FF = FeedForward(dim_model, dim_hidden, dropout=dropout)

    # inputs: [batch_size, seq_len, embedding_size]
    # outputs: [batch_size, seq_len, embedding_size]
    def forward(self, inputs):
        enc_output =  self.L_MHA(inputs, inputs, inputs) #how is the multihead attention pulled together? mean? addition?
        enc_output = self.L_FF(enc_output)
        return enc_output

class TEncoder(nn.Module):
    def __init__(self, n_layers, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.L_embedder = GraphEmbedding(dim_model)
        #NO POSITIONAL ENCODER
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6) #do we want to normalize input
        self.layer_stack = nn.ModuleList([
            TEncoderL(n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=dropout) for _ in range(n_layers)])

    # problems: [batch_size, seq_len, 4]
    # outputs: [batch_size, seq_len, embedding_size]
    def forward(self, problems):
        node_embedding = self.L_embedder(problems) # enc_output [batch_size, seq_len, embedding_size]
        # maybe dropout and normalization
        enc_output = self.layer_norm(node_embedding)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output

class TDecoderL(nn.Module):
    def __init__(self, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.L_slf_attn = MultiHeadAttention(n_head, dim_model, dim_k, dim_v, dropout=dropout)
        self.L_enc_attn = MultiHeadAttention(n_head, dim_model, dim_k, dim_v, dropout=dropout)
        self.L_positionFF = FeedForward(dim_model, dim_hidden, dropout=dropout)

     #dec_output [batch_size, partial_solution_size, embedding_size]
     #enc_output [batch_size, seq_len, embedding_size]
     #return [batch_size, partial_solution_size, embedding_size]
    def forward(self, dec_input, enc_output):
        dec_output = self.L_slf_attn(dec_input, dec_input, dec_input)
        dec_output = self.L_enc_attn(dec_output, enc_output, enc_output)
        dec_output = self.L_positionFF(dec_output)
        return dec_output

class TDecoder(nn.Module):
    def __init__(self, n_layers, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
        super().__init__() #initialise nn.Modules
        self.L_embedder = GraphEmbedding(dim_model)
        self.L_posEnc = PosEncoding(dim_model)
        self.L_norm = nn.LayerNorm(dim_model, eps=1e-6) #do we wnat to normalize input
        self.layer_stack = nn.ModuleList([
            TDecoderL(n_head, dim_model, dim_hidden, dim_v, dim_k, dropout=dropout) for _ in range(n_layers)])

    # states: [batch_size, partial_solution_size, 4]
    # encoder output: [batch_size, seq_len, embedding_size]
    # return: [batch_size, seq_len, dim_model] - with probabilities
    def forward(self, states, enc_output):
        # process the input
        dec_output = self.L_posEnc(self.L_embedder(states)) #positional encoding
        # process again
        dec_output = self.L_norm(dec_output) #dec_output [batch_size, partial_solution_size, embedding_size]
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output)
        # take dec output and enc_output and turn into return
        return dec_output

class Transformer(nn.Module):
    # def __init__(self, n_layers, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=0.1):
    def __init__(self, model_config):
        super().__init__() #initialise nn.Modules
        n_layers = int(model_config['n_layers'])
        n_head = int(model_config['n_head'])
        dim_model = int(model_config['model'])
        dim_hidden = int(model_config['hidden'])
        dim_k = int(model_config['k'])
        dim_v = int(model_config['v'])
        dropout = float(model_config['dropout'])
        self.L_encoder = TEncoder(n_layers, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=dropout)
        self.L_decoder = TDecoder(n_layers, n_head, dim_model, dim_hidden, dim_k, dim_v, dropout=dropout)
        self.L_attention = BahdanauAttention(dim_model)
        self.L_decoder_input = nn.Parameter(torch.FloatTensor(dim_model))

    # problem points ->
    def forward(self, problems, states):
        n_batch = problems.size(0)
        #ENCODE PROBLEM
        enc_output = self.L_encoder(problems)
        # h & c, [1, n_batch, dim_model]
        # embd_graph & enc_states, [n_batch, seq_len, dim_model]

        action_list = None #action_list: ([step, n_batch])
        action_probs_list = [] # probability of each action taken, action_probs_list: (step x [n_batch])
        while (action_list is None or action_list.size(0) != problems.size(1)):
            if action_list is None:
                # [batch_size, partial_solution_size, 4]
                dec_input = problems[-1] # decoder_input: [n_batch, dim_embedding]
            else:
                dec_input = action_list # takes the corresponding embeddedGraph[actions]
            dec_output = self.L_decoder(dec_input, enc_output) #[batch_size, partial_solution_size, embedding_size]
            b, p, e = dec_output.size()
            dec_query = dec_output.transpose(1, 2).reshape(b*e, p).sum(dim=1).reshape(b, e)#add dim=1 together and use this as a query
            # [batch_size, dim_model]
            results, _ = self.L_attention(dec_query, enc_output)
            logits = F.softmax(results, dim = 1)
            # [batch_size, seq_len] - probabilities
            if action_list is not None: # mask the previous actions
                logits[[[i for i in range(n_batch)], action_list]] = -np.inf
            # pick an action
            if actn_sele == 'sampling':
                probs = F.softmax(logits, dim=1) #soft max the probabilities
                actions = probs.multinomial(1).squeeze(1) # sample an index for each problem in the batch, actions: torch.Size([100])
            elif actn_sele == 'greedy':
                actions = logits.argmax(dim = 1)
                probs = Variable(torch.zeros(logits.size()), requires_grad = True).to(logits.device)
                probs[:, actions] = 1
            else:
                raise NotImplementedError
            # add it to various lists
            if action_list is None:
                action_list = actions.unsqueeze(dim=0)
            else:
                action_list = torch.cat((action_list, actions.unsqueeze(dim=0)), dim=0)
            action_probs_list.append(probs[[x for x in range(len(probs))], actions.data])
        return action_probs_list, action_list


        print("states", states)

        return logits

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
