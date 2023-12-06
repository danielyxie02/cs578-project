#NNI packages
import nni
from nni.compression import TorchEvaluator
from nni.compression.pruning import TaylorPruner, AGPPruner
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.speedup import ModelSpeedup


#pytorch packages 
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import DataLoader

#tensor packages 
#from tensorboardX import SummaryWriter
#from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
#from ignite.metrics import Accuracy, Loss
#from ignite.contrib.handlers import ProgressBar

#from models import LeNet_5
#from datasets import get_mnist_dataloaders

import models
import datasets
#import modelsNNI

import time

#imports from Models
import torch
import torch.nn.functional as F
#from torch.optim import Adam
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import DataLoader
import inspect
import math

import agpFunctions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_agp(model, dataset, target_sparsity):

    if model == "Lenet":
        model = models.LeNet_5().to(device)
    else:
        pass
    optimizer = agpFunctions.prepare_optimizer(model)
    if dataset == "MNIST":

        batch_size = 100
        agpFunctions.train(model, optimizer, agpFunctions.training_step, lr_scheduler= None, max_steps=None, max_epochs=3)
        _, test_loader = datasets.get_mnist_dataloaders(batch_size, batch_size)
        print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
        print('Original model after 10 epochs finetuning acc: ', agpFunctions.evaluate(model, test_loader), '%')

        config_list = [{
            'op_types': ['Conv2d'],
            'sparse_ratio':target_sparsity
        }]

        dummy_input = torch.rand(6,1,28,28)
        config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
        optimizer = agpFunctions.prepare_optimizer(model)
        evaluator = TorchEvaluator(agpFunctions.train, optimizer, agpFunctions.training_step)

        sub_pruner = TaylorPruner(model,config_list, evaluator, training_steps=100)
        scheduled_pruner = AGPPruner(sub_pruner, interval_steps=100, total_times=10)

        _, masks = scheduled_pruner.compress(max_steps=100 * 10, max_epochs=None)
        scheduled_pruner.unwrap_model()

        model = ModelSpeedup(model, dummy_input, masks).speedup_model()

        print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
        print(model)

        forward_source = inspect.getsource(model.forward)
        print(forward_source)

        #creating Pruned network with correct paramters 
        class LeNet_5Pruned(nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = nn.Conv2d(1, int(math.ceil(6-(6*target_sparsity))), 5, padding=2)
                self.conv2 = nn.Conv2d(int(math.ceil(6-(6*target_sparsity))), int(math.ceil(16-(16*target_sparsity))), 5)
                self.fc3 = nn.Linear(int(math.ceil(16-(16*target_sparsity))) * 5 * 5, 120)
                self.fc4 = nn.Linear(120, 84)
                self.fc5 = nn.Linear(84, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)

                x = F.relu(self.fc3(x.view(-1, int(math.ceil(16-(16*target_sparsity))) * 5 * 5)))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)

                return x
        #new Pruned Model
        modelPruned = LeNet_5Pruned().to(device)

        print(modelPruned)

        forward_source = inspect.getsource(modelPruned.forward)
        print(forward_source)

        #training pruned model and evaluating it 
        optimizer = agpFunctions.prepare_optimizer(modelPruned)
        agpFunctions.train(modelPruned, optimizer, agpFunctions.training_step, lr_scheduler=None, max_steps=None, max_epochs=3)
            
        print('Pruned model after 10 epochs finetuning acc: ', agpFunctions.evaluate(modelPruned, test_loader), '%')

    else:
        pass



