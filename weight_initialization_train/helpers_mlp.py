import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
torch.pi = 3.141592653589793238462643383279502884

import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch.nn as nn
import random
import itertools
from snntorch import spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
import argparse 
from scipy.stats import norm
from functools import partial


#LOAD DATASETS FUNCTIONS

def load_CIFAR10():
    # Define batch size and data path
    batch_size = 64
    data_path = 'path/to/data/cifar10'
    subset=10

    # Define data transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    
    #utils.data_subset(cifar10_train, subset)
    #utils.data_subset(cifar10_test, subset)
    # Create DataLoaders
    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def load_FashionMNIST():
    #load and transform the dataset
    
    batch_size = 64
    data_path='path/to/data/fashionmnist'
    subset=10
    
    # Define a transform
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

    fashionmnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    fashionmnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    # reduce datasets by 10x to speed up training. Comment the 2 lines below to use the entire dataset (60.000 instead of 6000)
    #utils.data_subset(fashionmnist_train, subset)

    #utils.data_subset(fashionmnist_test, subset)

    # Create DataLoaders
    train_loader = DataLoader(fashionmnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashionmnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, test_loader

def load_MNIST():
    #load and transform the dataset
    
    batch_size = 64
    data_path='path/to/data/mnist'
    subset=10
    
    # Define a transform
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # reduce datasets by 10x to speed up training. Comment the 2 lines below to use the entire dataset (60.000 instead of 6000)
    #utils.data_subset(mnist_train, subset)

    #utils.data_subset(mnist_test, subset)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, test_loader

    

#INIT LAYERS WEIGHTS FUNCTIONS
def kaiming_init_uniform_self(layer):
    if isinstance(layer, nn.Linear):
        #fan_in method
        n_input = layer.in_features
        var_w = 2/n_input
        torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(var_w))
        
def kaiming_init_normal_self(layer):
    if isinstance(layer, nn.Linear):
        #fan_in method
        n_input = layer.in_features
        var_w = 2/n_input
        torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(var_w))

def kaiming_init_uniform(layer):
    if isinstance(layer, nn.Linear):
        #fan_in method default
        torch.nn.init.kaiming_uniform_(layer.weight)

def kaiming_init_normal(layer):
    if isinstance(layer, nn.Linear):
        #fan_in method default
        torch.nn.init.kaiming_normal_(layer.weight)

def xavier_init_uniform(layer):
    if isinstance(layer, nn.Linear):
        #n_input = layer.in_features
        torch.nn.init.xavier_uniform_(layer.weight)

def xavier_init_normal(layer):
    if isinstance(layer, nn.Linear):
        #n_input = layer.in_features
        torch.nn.init.xavier_normal_(layer.weight)


def optimal_init_soft_reset(layer, threshold, beta):
    if isinstance(layer, nn.Linear):
        n_input = layer.in_features
        #compute var_w
        mean_y = 0
        var_y = 1
        probability_below_threshold = norm.cdf(threshold, loc=mean_y, scale=math.sqrt(var_y))
        area = 1 - probability_below_threshold #area from threshold to infinity
        
        #compute expected value of UX
        f_theta = norm.pdf((threshold-mean_y)/var_y)  # standard normal density function
        F_theta = norm.cdf((threshold-mean_y)/var_y)  # standard normal cumulative distribution function
        expected = mean_y + var_y*(f_theta / (1 - F_theta))
        
        
        var_w_optimal = (1-(beta**2)-(threshold**2)*(area*(1-area)) + 2*beta*threshold*expected )/(n_input*area)
        
        torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(var_w_optimal))


def optimal_init(layer, threshold):
    if isinstance(layer, nn.Linear):
        n_input = layer.in_features
        mean_y = 0
        var_y = 1
        probability_below_threshold = norm.cdf(threshold, loc=mean_y, scale=math.sqrt(var_y))
        area_from_threshold_to_infinity = 1 - probability_below_threshold
        var_w_optimal = 1/(n_input*area_from_threshold_to_infinity)
        
        torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(var_w_optimal))
        
def optimal_init_finite_size(layer, threshold):
    if isinstance(layer, nn.Linear):
        n_input = layer.in_features
        mean_y = 0
        var_y = 1
        
        # Compute true tail probability
        true_tail_probability = 1 - norm.cdf(threshold, loc=mean_y, scale=math.sqrt(var_y))
        
        sample_data = np.random.randn(n_input)

        # Compute tail probability from sample data
        computed_tail_probability = np.mean(sample_data > threshold)

        # Perform bootstrapping to estimate confidence interval
        n_bootstraps = 1000
        bootstrapped_tail_probabilities = []
        for _ in range(n_bootstraps):
            bootstrap_sample = np.random.choice(sample_data, size=len(sample_data), replace=True)
            bootstrapped_tail_probability = np.mean(bootstrap_sample > threshold)
            bootstrapped_tail_probabilities.append(bootstrapped_tail_probability)

        # Compute mean and standard deviation of bootstrapped tail probabilities
        mean_bootstrapped_tail_probability = np.mean(bootstrapped_tail_probabilities)
        std_dev_bootstrapped_tail_probability = np.std(bootstrapped_tail_probabilities)

        # Compute confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * (std_dev_bootstrapped_tail_probability / np.sqrt(n_bootstraps))
        lower_bound = mean_bootstrapped_tail_probability - margin_of_error
        upper_bound = mean_bootstrapped_tail_probability + margin_of_error
        
        #Compute fraction true_prob / lower_bound
        max_difference = true_tail_probability - lower_bound
        
        var_w_optimal = 1/(n_input*lower_bound)
        
        torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(var_w_optimal))


def custom_uniform_init(layer, a, center):
    if isinstance(layer, nn.Linear):
        root_k = math.sqrt(1 / layer.in_features)
        torch.nn.init.uniform_(layer.weight, - a * root_k + center, a * root_k + center)


#SPIKE COUNT FUNCTIONS
def spike_count(*layers):
#count the number of spikes per layer: given a tensor of dim [time_steps, batch_size, neurons]
#the function takes the mean of the firsr 2 dim and the sum for the 3rd one
    total_spikes = []
    with torch.no_grad():
        for layer in layers:
            num_neurons = layer.size(2)
            total_spikes.append(layer.mean(dim=0).mean(dim=0).sum(dim=0)/num_neurons)
    return total_spikes
    

def accuracy_mlp(output, targets):
    """Compute accuracy for the MLP model.

    Args:
        output (torch.Tensor): Output tensor from the model.
        targets (torch.Tensor): Target tensor.

    Returns:
        float: Accuracy.
    """
    output = output.to(targets.device)  # Ensure output is on the same device as targets
    _, predicted = output.max(1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

    
    
      