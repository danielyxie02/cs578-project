
# Import Torch and other packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import time
import datasets
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model,epochs):
	# Training Function
	# all of this is setup
	# model setup
	model = model
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
	train_loader, val_loader = datasets.get_mnist_dataloaders(batch_size, batch_size)  # these are functions/objects that will load the MNIST data correctly

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
	trainer.run(train_loader, max_epochs=epochs)



def weight_rewind_prune(model, target_sparsity):
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d):
			prune.l1_unstructured(module, name='weight', amount=target_sparsity)
  
		elif isinstance(module, torch.nn.Linear):
			prune.l1_unstructured(module, name='weight', amount=target_sparsity)
	# Reinitialize weights to 0
	print('Reinitializing Weights')	
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.weight.data.fill_(0)

	# Fix gradients at 0 for those in mask with value 0
	print("Starting to apply mask and fix gradients at 0") 
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			assert(layer.weight.shape == layer.weight_mask.shape)
			# set initial weights in real model according to mask
			layer.weight.data[layer.weight_mask == 0.0] = 0.0

			# tell pruned weights to always have 0 gradient; combined, these params will be forever 0
			# done with some help from
			# https://discuss.pytorch.org/t/use-forward-pre-hook-to-modify-nn-module-parameters/108498/5
			def get_hook(mask):
				def hook(grad):
					return grad * mask
				return hook
			layer.weight.register_hook(get_hook(layer.weight_mask))
	print('Done applying Mask')

def spars_calc(model):
    spars = 0
    full = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            spars += float(torch.sum(layer.weight == 0))
            full += int(layer.weight.nelement())
    print("Global sparsity: {:.2f}%".format(100 * float(spars) / float(full)))
    return spars,full


