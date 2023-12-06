import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import nni

import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#functions

def get_data(dataset, batchsize):
    if dataset == "MNIST":
        train_loader, test_loader = datasets.get_mnist_dataloaders(batchsize, batchsize)
        return train_loader, test_loader

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step,
          lr_scheduler: _LRScheduler, max_steps: int, max_epochs: int):
    assert max_epochs is not None or max_steps is not None
    #train_loader, test_loader = prepare_dataloader()
    #grant addittion ---
    train_loader, test_loader = get_data("MNIST", 100)

    #----
    max_steps = max_steps if max_steps else max_epochs * len(train_loader)
    max_epochs = max_steps // len(train_loader) + (0 if max_steps % len(train_loader) == 0 else 1)
    count_steps = 0

    model.train()
    for epoch in range(max_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = training_step((data, target), model)
            loss.backward()
            optimizer.step()
            count_steps += 1
            if count_steps >= max_steps:
                acc = evaluate(model, test_loader)
                print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')
                return
        acc = evaluate(model, test_loader)
        print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')

def evaluate(model: torch.nn.Module, test_loader):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100 * correct / len(test_loader.dataset)


def training_step(batch, model: torch.nn.Module):
    output = model(batch[0])
    loss = F.cross_entropy(output, batch[1])
    return loss

def prepare_optimizer(model: torch.nn.Module):
    optimize_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = nni.trace(Adam)(optimize_params, lr=0.01)
    return optimizer