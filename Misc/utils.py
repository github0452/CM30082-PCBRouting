import Environments
import copt
import numpy as np
import time
import Baselines
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import math
# import matplotlib as plt

import torch
from torch.utils.tensorboard import SummaryWriter

#
# Transition = namedtuple(
#     'Transition', ('state', 'action', 'reward', 'state_next', 'done')
# )

class ReplayMemory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = {}
        self._position = 0
        self._filled = False

    def add(self, *args):
        self._memory[self._position] = Transition(*args)
        if self._filled == False and (self._position+1) == self._capacity:
            self._filled = True
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, device=None):
        batch = random.sample(list(self._memory.values()), batch_size)
        next_batch = [torch.stack(tensors).to(device) for tensors in zip(*batch)]
        return next_batch

    def isFull(self):
        return self._filled
