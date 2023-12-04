import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import copy
import types

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

def snip_forward_conv2d(self, x):
		return F.conv2d(x, self.weight * self.c, self.bias,
						self.stride, self.padding, self.dilation, self.groups)
def snip_forward_linear(self, x):
		return F.linear(x, self.weight * self.c, self.bias)
def get_SNIP_mask(model, target_sparsity, dataloader):
	c_model = copy.deepcopy(model)  # use copy of given model to do the pruning-at-init, since we're creating a new mask
	mb_x, mb_y = next(iter(dataloader))  # mb means the minibatch (used to train c)
	mb_x = mb_x.to(device)
	mb_y = mb_y.to(device)

	# Create "trainable" weight mask c (we don't actually want to train, but we want the grads after)
	for layer in c_model.modules():
		# print(list(layer.named_parameters()))
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.c = nn.Parameter(torch.ones_like(layer.weight))  # create the c_ij for the existing layer
			nn.init.kaiming_uniform_(layer.weight)  # replace the existing layer with VS-H initialization
			layer.weight.requires_grad = False  # freeze existing layer since we only want to train c
			layer.c.requires_grad = True  # explicitly say that c needs gradients
		# Override the forward methods
		if isinstance(layer, nn.Conv2d):
			layer.forward = types.MethodType(snip_forward_conv2d, layer)
		elif isinstance(layer, nn.Linear):
			layer.forward = types.MethodType(snip_forward_linear, layer)

	# Calculate "connection sensitivities"
	c_model.zero_grad()
	fwd_result = c_model.forward(mb_x)
	loss_fn = nn.CrossEntropyLoss()
	loss = loss_fn(fwd_result, mb_y)
	loss.backward()

	# Aggregate dL/dc values
	dLdc_orig = []
	for layer in c_model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			dLdc_orig.append(torch.abs(layer.c.grad))
			# dLdc is a bunch of matrices since they came from weights; flatten and gather them
	dLdc = torch.cat([torch.flatten(mat) for mat in dLdc_orig])

	# Get kth largest dL/dc value as a threshold
	k = int(target_sparsity * len(dLdc))
	topk_dLdc, _ = torch.topk(dLdc, k)
	threshold = topk_dLdc[-1]

	# Create weight mask
	snip_mask = []
	# dLdc_orig has the same dimensions as the layers, so we can use that to build the mask
	for mat in dLdc_orig:
		snip_mask.append((mat >= threshold).float())
	return snip_mask


# SNIP pruning: WIP
def SNIP_prune(model, target_sparsity, dataloader):
	snip_mask = get_SNIP_mask(model, target_sparsity, dataloader)

	print("Starting to apply mask")
	i = 0 
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			assert(layer.weight.shape == snip_mask[i].shape)
			# set initial weights in real model according to mask
			layer.weight.data[snip_mask[i] == 0.0] = 0.0

			# tell pruned weights to always have 0 gradient; combined, these params will be forever 0
			# done with some help from
			# https://discuss.pytorch.org/t/use-forward-pre-hook-to-modify-nn-module-parameters/108498/5
			def get_hook(mask):
				def hook(grad):
					return grad * mask
				return hook
			layer.weight.register_hook(get_hook(snip_mask[i]))
			i += 1  # move on to the next layer