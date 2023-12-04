import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# No pruning is one extreme, random pruning is the "other" extreme: 
# we must demonstrate we're doing *better* than random.
# (Otherwise there's no point in doing a "smart" pruning technique!)
# Luckily, PyTorch has a nice tutorial for pruning methods that are as simple as random pruning.
# https://pytorch.org/tutorials/intermediate/pruning_.tutorial.html
# We'll modify the existing model; no deepcopying.
def random_prune(model, target_sparsity):
	prune.global_unstructured(
		[(module, 'weight') for (name, module) in model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)],
		pruning_method=prune.RandomUnstructured,
		amount= 1 - target_sparsity
		)

# SNIP pruning: WIP
def snip_prune(model, target_sparsity, dataloader):
	model = copy.deepcopy(model)  # just in case
	mb_x, mb_y = next(iter(dataloader))  # mb meaning the minibatch used to train the mask
	mb_x = mb_x.to(device)
	mb_y = mb_y.to(device)

	# Create "trainable" weight mask c (we don't actually want to train, but we want the grads after)
	for layer in model.modules():
		print(list(layer.named_parameters()))
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))  # create the c_ij for the existing layer
			nn.init.kaiming_uniform_(layer.weight)  # replace the existing layer with VS-H initialization
			layer.weight.requires_grad = False  # freeze existing layer since we only want to train c
			# This lambda tells forward() to also multiply weights by c
			layer.weight.register_forward_hook = lambda module, input, output: output * layer.c

	# Calculate "connection sensitivities"
	model.zero_grad()
	fwd_result = model.forward(mb_x)
	loss_fn = nn.CrossEntropyLoss()
	loss = loss_fn(fwd_result, mb_y)
	loss.backward()