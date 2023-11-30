import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from models import LeNet_5
from datasets import get_mnist_dataloaders

import time

# potentially use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
	# all of this is setup
	# model setup
	model = LeNet_5().to(device)
	learning_rate = 0.01 
	optimizer = optim.SGD(
		model.parameters(), 
		lr=learning_rate, 
		momentum=0.9,
		weight_decay=0.0005)  # can possibly be richer
	loss = nn.CrossEntropyLoss()
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 25000, gamma=0.1)  # dynamic lr adjustment used specifically in the SNIP paper
	batch_size = 100

	# dataset setup
	train_loader, val_loader = get_mnist_dataloaders(batch_size, batch_size)  # these are functions/objects that will load the MNIST data correctly

	# evaluation setup
	metrics = { 'accuracy': Accuracy(), 'loss': Loss(loss)}  # metrics will be used for evaluating the model

	# actually create the things that will execute the main training loop
	# ignite is nice in the sense that it allows us to not write our own training loop.. but maybe we want to.
	# anyways, this is just a toy example.
	trainer = create_supervised_trainer(model, optimizer, loss, device)
	evaluator = create_supervised_evaluator(model, metrics, device)

	timer = time.time()

	# we have to define hooks after we define the trainer since hooks rely on the trainer
	@trainer.on(Events.ITERATION_COMPLETED)
	def advance_lr_scheduler():
		lr_scheduler.step()

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(trainer):
		nonlocal timer
		evaluator.run(train_loader)
		metrics = evaluator.state.metrics
		print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(trainer):
		nonlocal timer
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f} Time (seconds) it took for this epoch: {time.time() - timer}")
		timer = time.time()
	
	print("Start training")
	trainer.run(train_loader, max_epochs=50)

if __name__ == "__main__":
	train()
	print("done")