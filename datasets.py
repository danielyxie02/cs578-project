from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

def get_mnist_dataloaders(train_batch_size, val_batch_size):

	data_transform = Compose([transforms.ToTensor()])

	train_dataset = MNIST("_dataset", True, data_transform, download=True)
	test_dataset = MNIST("_dataset", False, data_transform, download=False)

	train_loader = DataLoader(
		train_dataset,
		train_batch_size,
		shuffle=True,
		num_workers=2,
		pin_memory=True)
	val_loader = DataLoader(
		test_dataset,
		val_batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True)

	return train_loader, val_loader