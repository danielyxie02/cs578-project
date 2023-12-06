import torch
import models
import agpFunctions
import datasets 
import agpTrain


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print("Start")
    agpTrain.train_agp("Lenet", "MNIST", .5)