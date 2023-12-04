import torch

import models
import train
import datasets

if __name__ == "__main__":
	print("start")
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# I was thinking of testing sparsities 1, 0.5, 0.25, 0.125, ... 1/(2^7)
	# since the sparsities only really change (get worse) for extreme sparsities
	sparsities = [1.0 / (2 ** i) for i in range(7)]
	for sparsity in sparsities:
		# Reinitialize model just in case pruning methods tinkered around with it	
		# Can manually change the next three lines based on what you want to test
		model = models.LeNet_5().to(device)
		dataset_name = "MNIST"  # we'll probably stick to MNIST
		train.train_random(model, dataset_name, sparsity)  # probably want to manually change this call based on pruning method
	print("done")