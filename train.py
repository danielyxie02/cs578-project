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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# This file holds the train loop for each pruning method,
# since different pruning methods might have different custom training loops
# (or it might not want to use a fancy training loop at all.)

# Training used for random pruning: just using ignite does fine.
def train_random(model, dataset_name, target_sparsity=1):
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
	prune.random_prune(model, target_sparsity)

	# This hook logs some information an entire epoch finishes
	timer = time.time()
	best_acc = 0
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(trainer):
		nonlocal best_acc
		nonlocal timer
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		best_acc = max(best_acc, metrics['accuracy'])
		print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.4f} Best acc so far: {best_acc} Time (seconds): {(time.time() - timer):.4f}")
		timer = time.time()
	
	print(f"Start training for random pruning, sparsity {target_sparsity}")
	trainer.run(train_loader, max_epochs=50)
	print(f"Training finished for random pruning, best test accuracy for sparsity {target_sparsity}: {best_acc}")