# CM30082-PCBRouting

**Project Structure:**
 - BatchFiles: contains bash files that i used to automate running all the tests
 - datasets: np array datasets that can be directly loaded using pickle.load
 - models: contains code for the models and general layers, containing:
      - specific layers for the model, which inherits nn.Module and its 2 methods 
      - the model class which turns a problem, passes it through model and gets actions (and other details as needed), also inherits nn.Module and its 2 methods
      - the wrapped model class, which combines the components needed to train the model and has 4 methods
            - train: takes in n_batch, p_size and potentially a path, and trains the model, returning some details
            - test: takes in n_batch, p_size and potentially a path, and tests the model, returning some details
            - save: returns a dict of all the things needed to save the model
            - load: takes in a dict and loads the model up
 - RLalgorithm: contains code for the reinforcement learning algorithm, specifically REINFORCE with expMovingAvgBaseline and critic baseline, composes of 5 main components:
   - passIntoParameters (to initialise optimizer and schedular for the actor model),
   - train (given problem, reward and probability will train the model),
   - additional params (returns a list of the names the additional params returned),
   - save and load
 - Misc:
   - CreateHPOConfigs: used to generate the hyperparameter configs for HPO # MAY BE OUT OF DATE
   - CreateTrainingConfigs: used to generate the hyperparameter configs for Training # MAY BE OUT OF DATE
   - Environment: contains code for the environments
   - ExploreProblemSpace: contains code used to explore the problem space
   - utils: contains code for unused replayMemory class
 - runs: contains configurations and weights for models that can just be directly loaded.
 - There are 2 mains that can be called to run code
   - runMultiModel: for runs that use the 2 models stacked
   - runSingleModel: for runs that use a single model

For the wrapped models, they are named slightly differently. Construction transformer is construction attention in dissertation and improvement transformer is improvement attention in dissertation.

 **Training Models:**
 In order to train a model, run the below command:
  + python3 runSingleModel.py train <locationOfConfig> <locationOfDataFolder> <numEpochs> <problemSize> <filterProbsWithNoSol>
 
 For example:
  - training construction pointer:
     + python3 runSingleModel.py train runs/Training/Pointer-config runs/Training/5Pointer 10 5 False
  - training construction attention:
     + python3 runSingleModel.py train runs/Training/Transformer-config runs/Training/5Transformer 10 5 False
  - training improvement attention:
     + python3 runSingleModel.py train runs/Training/Improvement-config runs/Training/5Improvement 10 5 False
 
 **Testing Models:**
 In order to test a single model, run the below command:
  + python3 runSingleModel.py test <locationOfConfig> <locationOfDataFolder> <numEpochs> <problemSize> <filterProbsWithNoSol> <numSolSampling> <datasetLoc>
 In order to test a multimodel, run the below command:
  + python3 runMultiModel.py <locationOfConfig> <locationOfDataFolder> <numEpochs> <problemSize> <filterProbsWithNoSol> <numSolSampling> <datasetLoc>

For example:
  - testing improvement attention:
     + python3 runSingleModel.py test runs/Training/Improvement-config runs/Training/5Improvement 10 5 False 1 datasets/n5b5120.pkg
  - testing multi-model where pointer -> improvement attention:
     + python3 runMultiModel.py test runs/StackedTraining/Pointer-config runs/Training/5Pointer 10 5 False 1 datasets/n5b5120.pkg
