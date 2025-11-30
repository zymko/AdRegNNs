import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import matplotlib.colors as mcolors
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# torch.manual_seed(123)

class MNISTDataset(Dataset):
    def __init__(self, dataset, coeff_matrix, std=0.001, mean=0):
        self.std= std
        self.mean = mean
        self.data = dataset
        # self.data_filtered = []
        self.samples = [images for images, targets in dataset ]
        self.targets = [targets for _, targets in dataset ]
        self.coeff_matrix = coeff_matrix
        ## add a linear operator and gaussian noise
        self.noise_samples = [self.coeff_matrix @ images + torch.randn(images.size()) * self.std + self.mean for images in self.samples]

    def __getitem__(self, index):
        samples = self.samples[index]
        targets = self.targets[index]
        noise_samples =self.noise_samples[index]
        return samples, targets, noise_samples

    def __len__(self):
        return len(self.samples)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloaders_mnist(batch_size, coeff_matrix, num_workers=0, drop_last=True, shuffle=False, validation_fraction=None, std=0.001, mean=0):
    """loads the MNIST

    Args:
        batch_size (int): images in one batch
        validation_fraction (float, optional): amount of data to be in validation set. Defaults to None.

    Returns:
        list(torch.tensor): training, validation and test datasets
    """

    train_transforms = transforms.ToTensor()

    # noise_transforms = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(0., 0.001)])

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)


    test_transforms = transforms.ToTensor()
    # valid_dataset = datasets.MNIST(root='data',
    #                                train=True,
    #                                transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:

        train_dataset_filtered=MNISTDataset(train_dataset, coeff_matrix=coeff_matrix, std=std, mean=mean)
        dataset_size = len(train_dataset_filtered)
        num = int(validation_fraction * dataset_size)
        train_indices = torch.arange(0, dataset_size - num)
        valid_indices = torch.arange(dataset_size - num, dataset_size)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=train_dataset_filtered,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler,
                                  drop_last=drop_last,
                                  shuffle=shuffle)

        train_loader = DataLoader(dataset=train_dataset_filtered,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  sampler=train_sampler,
                                  shuffle=shuffle)
    else:
        train_loader = DataLoader(dataset=MNISTDataset(train_dataset, coeff_matrix=coeff_matrix,std=std, mean=mean),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  shuffle=shuffle)


    test_loader = DataLoader(dataset=MNISTDataset(test_dataset, coeff_matrix=coeff_matrix,std=std, mean=mean),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=drop_last,
                             shuffle=shuffle)

    if validation_fraction is None:
        return train_loader,  test_loader
    else:
        return train_loader, valid_loader, test_loader