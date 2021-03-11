class PntrNetV2(nn.Module):
    def __init__(self):
        self.node_embedding_layer
        self.self_attention_layer
        self.FF_layer

    def forward(problem):
        #ENCODER
        embededProblem = self.node_embedding_layer(problem)
        # sinusoidal positional encodings
        N=3 #below each of 3 layers have DIFFERENT WEIGHTS
        for i in range(N):
            advNodeEmbed = self.self_attention_layer(embededProblem)
            #actually just a one head multiattention
            advNodeEmbed = self.FF_layer(advNodeEmbed)
            #transforms each node embedding seperately with shared parameters
            # residual connections/skip connections - the adding the beginning vector onto original
            # - supposed to help with gradient propagation and to avoid the exploding/vanishing problem
            # batch normalization - normalizes data

        #DECODER
        #some sort of compatablity layer to produce node pair selections
        #mask the diagonal since picking same node twice is a bit pointless
        #mask the previous choice to prevent it just undoing stuff - prevent solution cycling
        #softmax it
        #sample - NOT greedy choice



#batch training different sized problems - DUMMY points?
