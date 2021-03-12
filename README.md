# CM30082-PCBRouting

Structure of project:
 - datasets: tensor datasets that can be directly loaded using pickle.load
 - models: contains code for the models and general layers, model files contains 3 main components:
  - specific layers for the model, which inherits nn.Module and its 2 methods 
  - the model class which turns a problem, passes it through model and gets actions (and other details as needed), also inherits nn.Module and its 2 methods
  - the wrapped model class, which combines the components needed to train the model and has 4 methods
   - train: takes in n_batch, p_size and potentially a path, and trains the model, returning some details
   - test: takes in n_batch, p_size and potentially a path, and tests the model, returning some details
   - save: returns a dict of all the things needed to save the model
   - load: takes in a dict and loads the model up
 - RLalgorithm: contains code for the reinforcement learning algorithm, composes of 5 main components: passIntoParameters (to initialise optimizer and schedular for the actor model), train (given problem, reward and probability will train the model), additional params (returns a list of the names the additional params returned), save and load
 - runs: contains data from pytorch tensorboard
