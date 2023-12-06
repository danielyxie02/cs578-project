import torch
import torchvision
import torch.nn as nn

import models
import train
import datasets
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, default='')
	parser.add_argument("-d", "--dataset", type=str, default='')
	parser.add_argument("-p", "--pruner", type=str, default='')
	args = parser.parse_args()
	return args

def get_model(model_name, dataset_name):
	match model_name:
		case "LeNet_5":
			return models.LeNet_5()
		case "LeNet_300_100":
			return models.LeNet_300_100()
		case "ResNet18":
			model = torchvision.models.resnet18(num_classes=10)
			if (dataset_name == "MNIST"):  # slight modification for MNIST since it only has 1 channel
				model.conv1 =nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3)
			return model
		case _:
			print("Model name not recognized")

if __name__ == "__main__":
	print("start")
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args = parse_arguments()
	print(f"Settings: model={args.model}, dataset={args.dataset}, pruning method={args.pruner}")
	# I was thinking of testing sparsities 1, 0.5, 0.25, 0.125, ... 1/(2^7)
	# since the sparsities only really change (get worse) for extreme sparsities
	sparsities = [1.0 / (2 ** i) for i in range(9)]
	sparsities = sparsities[::-1]
	for sparsity in sparsities:
		print(f"Training sparsity={sparsity}")
		# Reinitialize model just in case pruning methods tinkered around with it	
		# Can manually change the next three lines based on what you want to test, or fill them in by command line
		dataset_name = args.dataset
		model = get_model(args.model, dataset_name).to(device)
		pruning_method = args.pruner
		match pruning_method:
			case "random":
				train.train_random(model, dataset_name, sparsity)
			case "SNIP":
				train.train_SNIP(model, dataset_name, sparsity)
			case "weight_rewind":
				train.train_rewind(model, dataset_name, sparsity)
			case "weight_rewind_use_old_model":
				train.train_rewind_use_old_model(model, dataset_name, sparsity)
			case _: 
				print("Pruning method not recognized")
	print("done")