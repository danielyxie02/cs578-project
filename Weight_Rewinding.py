# Import from self-created files
import models
import datasets
import weight_rewinding_support as wr

# Import Torch and other packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import time

# Initialize Model and Check initial sparsity
model = models.LeNet_5()
wr.spars_calc(model)

# Train Initial Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.LeNet_5()
wr.train(model,3)

# Prune Parameters 
sparsities = [1.0 / (2 ** i) for i in range(9)]
print(sparsities)
wr.weight_rewind_prune(model,1-sparsities[6])

# Train Pruned Model
wr.train(model,3)

# Calculate Final Sparsity
wr.spars_calc(model)
