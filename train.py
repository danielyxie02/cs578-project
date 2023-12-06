import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import models
import train
import time
import sys
import prune
import datasets
import copy
import weight_rewinding_support as wr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# This file holds the train loop for each pruning method,
# since different pruning methods might have different custom training loops
# (or it might not want to use a fancy training loop at all.)

# Training used for random pruning: just using ignite does fine.
def train_generic(which, model, dataset_name, target_sparsity):
	# setup
	optimizer = optim.NAdam(model.parameters())
	loss = nn.CrossEntropyLoss()
	batch_size = 100
	train_loader, val_loader = datasets.get_dataloaders(dataset_name, batch_size, batch_size)  # loads MNIST data
	metrics = { 'accuracy': Accuracy() }  # metrics will be used for evaluating the model

	# ignite lets us abstract away the training loop
	trainer = create_supervised_trainer(model, optimizer, loss, device)
	evaluator = create_supervised_evaluator(model, metrics, device)

	# do pruning here, note: change this if you're testing a different method
	# random and SNIP are both "prune at initialization" methods, so let's capture them here
	# weight rewinding does pruning outside of the training loop, so nothing here
	if which == "random":
		prune.random_prune(model, target_sparsity)
	elif which == "SNIP":
		prune.SNIP_prune(model, target_sparsity, train_loader)

	# This hook logs some information an entire epoch finishes
	total_time = 0
	timer = time.time()
	best_acc = 0
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(trainer):
		nonlocal best_acc, timer, total_time
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		best_acc = max(best_acc, metrics['accuracy'])
		dtime = time.time() - timer
		print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.4f} Best acc so far: {best_acc} Time (seconds): {dtime:.4f}")
		total_time += dtime
		timer = time.time()
	
	trainer.run(train_loader, max_epochs=2)
	print(f"Training finished. Best test accuracy for sparsity {target_sparsity}: {best_acc}, time for 20 epochs (s) = {total_time}")

def train_random(model, dataset_name, target_sparsity=1):
	train_generic("random", model, dataset_name, target_sparsity)

def train_SNIP(model, dataset_name, target_sparsity=1):
	train_generic("SNIP", model, dataset_name, target_sparsity)

def train_rewind(model, dataset_name, target_sparsity=1):
	prune.spars_calc(model)
	train_generic("", model, dataset_name, target_sparsity)
	prune.weight_rewind_prune(model, target_sparsity)
	train_generic("", model, dataset_name, target_sparsity)
	prune.spars_calc(model)  # print sparsity after since weights get reset to 0

def train_rewind_use_old_model(model, dataset_name, target_sparsity=1):
	init_model = copy.deepcopy(model)  # save so we know what weights to rewind to, instead of 0?
	prune.spars_calc(model)
	train_generic("", model, dataset_name, target_sparsity)
	prune.weight_rewind_prune_use_old_model(model, init_model, target_sparsity)
	train_generic("", model, dataset_name, target_sparsity)
	prune.spars_calc(model)  # print sparsity after since weights get reset to 0