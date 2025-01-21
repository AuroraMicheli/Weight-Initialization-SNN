# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import os
import sys

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch.nn as nn

import itertools
from snntorch import spikegen
from helpers_mlp import *



np.random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
torch.manual_seed(42)  # Set random seed for PyTorch for reproducibility
torch.cuda.manual_seed_all(42) 
'''
class Net(nn.Module):
    def __init__(self, n_layers, threshold, beta, n_neurons, init_func):
        super().__init__()

        # Initialize layers
        beta = beta
        num_inputs = 784
        num_hidden = n_neurons
        num_outputs = 10

        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # First Layer: LIF neurons + linear layer
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_inputs, num_hidden))

        # Hidden Layers: LIF neurons + linear layers
        for _ in range(n_layers - 3):  # Excluding first and last 2 layers
            self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
            self.layers.append(nn.Linear(num_hidden, num_hidden))

        # Last 2 Layers: lif - linear - lif_output
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_hidden, num_outputs))
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))

        if 'threshold' in init_func.__code__.co_varnames and 'beta' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold, beta=beta)
        elif 'threshold' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold)
        else:
            self.init_func = init_func

        self.apply(lambda layer: self.init_func(layer))

    def forward(self, input_tensor, time_steps):
        spike_out = []
        mems = [layer.init_leaky() for layer in self.lif_layers]
        var_mems_list = [[] for _ in range(len(self.layers))]  # Initialize lists for variance values
        grad_mags_list = [[] for _ in range(len(self.layers))]  # Initialize lists for gradient magnitudes

        for step in range(time_steps):
            for i in range(len(self.lif_layers)):
                if i == (len(self.lif_layers) - 1):  # Last LIF layer
                    spk, mems[i] = self.lif_layers[i](cur, mems[i])
                    spike_out.append(spk)
                else:
                    if i == 0:
                        spk, mems[i] = self.lif_layers[i](input_tensor, mems[i])
                    else:
                        spk, mems[i] = self.lif_layers[i](cur, mems[i])

                    # Compute and store variance
                    var_mems = torch.var(mems[i]).item()
                    var_mems_list[i].append(var_mems)

                    # Compute input to next layer (if not last layer)
                    if i < (len(self.lif_layers) - 1):
                        cur = self.layers[i](spk)

        # Stack spike_out into a tensor
        if spike_out:
            spike_out = torch.stack(spike_out, dim=0)
        else:
            spike_out = torch.tensor([], device=input_tensor.device)

        return var_mems_list, spike_out

    def extract_gradients(self):
        grad_mags = []
        for name, param in self.named_parameters():
            if 'weight' in name:  # Check if the parameter name contains 'weight'
                if param.grad is not None:
                    grad_mags.append(param.grad.norm().item())
                else:
                    grad_mags.append(0)
        return np.array(grad_mags)
'''

class Net(nn.Module):
    def __init__(self, n_layers, threshold, beta, n_neurons, init_func):
        super().__init__()

        # Initialize layers
        beta = beta
        num_inputs = 784
        num_hidden = n_neurons
        num_outputs = 10

        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # First Layer: LIF neurons + linear layer
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_inputs, num_hidden))

        # Hidden Layers: LIF neurons + linear layers
        for _ in range(n_layers - 3):  # Excluding first and last 2 layers
            self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
            self.layers.append(nn.Linear(num_hidden, num_hidden))

        # Last 2 Layers: lif - linear - lif_output
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_hidden, num_outputs))
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))

        if 'threshold' in init_func.__code__.co_varnames and 'beta' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold, beta=beta)
        elif 'threshold' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold)
        else:
            self.init_func = init_func

        self.apply(lambda layer: self.init_func(layer))

    def forward(self, input_tensor, time_steps):
        spike_out = []
        mems = [layer.init_leaky() for layer in self.lif_layers]
        var_mems_accum = [0.0] * len(self.lif_layers)  # Initialize accumulators for variance
        var_mems_count = [0] * len(self.lif_layers)  # Initialize counts for variance calculation

        for step in range(time_steps):
            for i in range(len(self.lif_layers)):
                if i == (len(self.lif_layers) - 1):  # Last LIF layer
                    spk, mems[i] = self.lif_layers[i](cur, mems[i])
                    spike_out.append(spk)
                else:
                    if i == 0:
                        spk, mems[i] = self.lif_layers[i](input_tensor, mems[i])
                    else:
                        spk, mems[i] = self.lif_layers[i](cur, mems[i])

                    # Accumulate variance
                    mems_mean_batch = mems[i].mean(dim=0)
                    var_mems_accum[i] += torch.var(mems[i]).item()
                    var_mems_count[i] += 1

                    # Compute input to next layer (if not last layer)
                    if i < (len(self.lif_layers) - 1):
                        cur = self.layers[i](spk)

        # Compute average variance for each layer
        var_mems_list = np.array([var_mems_accum[i] / var_mems_count[i] if var_mems_count[i] > 0 else 0 for i in range(len(var_mems_accum))])

        # Stack spike_out into a tensor
        if spike_out:
            spike_out = torch.stack(spike_out, dim=0)
        else:
            spike_out = torch.tensor([], device=input_tensor.device)

        return var_mems_list, spike_out


    def extract_gradients(self):
        grad_mags = []
        for name, param in self.named_parameters():
            if 'weight' in name:  # Check if the parameter name contains 'weight'
                if param.grad is not None:
                    grad_mags.append(param.grad.norm().item())
                else:
                    grad_mags.append(0)
        return np.array(grad_mags)


def training(net, time_steps, train_loader, test_loader):
    dtype = torch.float
    
    loss_hist = []
    loss_epoch_hist = []
    acc_hist = []
    acc_epoch_hist = []
    test_loss_hist = []
    test_loss_epoch_hist = []
    test_acc_hist = []
    test_acc_epoch_hist = []
    var_mems_train = []
    var_mems_test = []
    grad_mags_train = []
    
    num_epochs = 150
    num_steps = time_steps
    
    # Loss function and optimizer
    loss_fn = SF.ce_count_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


    # Outer training loop
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        loss_epoch = []
        acc_epoch = []
        test_loss_epoch = []
        test_acc_epoch = []
        var_mems_batch_train = []
        var_mems_batch_test = []
        grad_mags_batch_train = []

        
        # Training phase
        net.train()
        for data, targets in train_loader:
            data = torch.flatten(data, 1, 3).to(device)
            targets = targets.to(device)

            # Forward pass
            var_mems_list, spk_out = net(data, time_steps)
            var_mems_batch_train.append(var_mems_list)

            # Initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(spk_out, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()

            # Extract and store gradient magnitudes
            grad_mags = net.extract_gradients()
            grad_mags_batch_train.append(grad_mags)

            # Store loss and accuracy history for future plotting
            loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_out, targets)
            acc_hist.append(acc)

        # Test phase 
        with torch.no_grad():
            net.eval()
            for test_data, test_targets in test_loader:
                test_data = torch.flatten(test_data, 1, 3).to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                var_mems_list_test, spk_out_test = net(test_data, time_steps)
                var_mems_batch_test.append(var_mems_list_test)

                # Test set loss 
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss_fn(spk_out_test, test_targets) 

                test_loss_hist.append(test_loss.item())

                # Test set accuracy
                test_acc = SF.accuracy_rate(spk_out_test, test_targets)
                test_acc_hist.append(test_acc)

        # Compute and store mean TRAIN LOSS over batches
        loss_epoch = np.mean(loss_hist)
        loss_epoch_hist.append(loss_epoch)

        # Compute and store mean TRAIN ACC over batches
        acc_epoch = np.mean(acc_hist)
        acc_epoch_hist.append(acc_epoch)

        # Compute and store mean TEST LOSS over batches
        test_loss_epoch = np.mean(test_loss_hist)
        test_loss_epoch_hist.append(test_loss_epoch)

        # Compute and store mean TEST ACC over batches
        test_acc_epoch = np.mean(test_acc_hist)
        test_acc_epoch_hist.append(test_acc_epoch)

        # Compute and store mean VARIANCE over batches for training
        #print(np.array(var_mems_batch_train).shape)
        var_mems_train_epoch = np.mean(var_mems_batch_train, axis=0)
        
        #var_mems_train_epoch = np.array([np.mean(np.array(var_mems), axis=0) for var_mems in var_mems_batch_train])
        #print(var_mems_train_epoch.shape)
        var_mems_train.append(var_mems_train_epoch)

        # Compute and store mean VARIANCE over batches for test
        var_mems_test_epoch = np.mean(var_mems_batch_test, axis=0)
        #var_mems_test_epoch = [np.mean(np.array(var_mems), axis=0) for var_mems in var_mems_batch_test]
        var_mems_test.append(var_mems_test_epoch)

        # Compute and store mean GRADIENT MAGNITUDES over batches for training
        #grad_mags_train_epoch = np.mean(grad_mags_batch_train, axis=0).tolist()
        grad_mags_train_epoch = np.mean(grad_mags_batch_train, axis=0)
        grad_mags_train.append(grad_mags_train_epoch)
    
    return loss_epoch_hist, test_loss_epoch_hist, acc_epoch_hist, test_acc_epoch_hist, var_mems_train, var_mems_test, grad_mags_train


def main(args):
    n_layers=args.n_layers
    threshold=args.threshold
    beta=args.beta
    n_neurons=args.n_neurons
    time_steps=args.time_steps
    dataset_name = args.dataset_name

    init_functions = {
    'kaiming':kaiming_init_normal,
    'xavier': xavier_init_normal,
    'kaiming_self':kaiming_init_normal_self,
    'optimal_reset': optimal_init_soft_reset,
    'optimal': optimal_init
    # Add other initialization functions here
    }

    init_func_name = args.init_func
    if init_func_name not in init_functions:
        raise ValueError(f"Invalid initialization function: {init_func_name}")
    
    init_func = init_functions[init_func_name]

    if dataset_name == 'mnist':
        train_loader, test_loader = load_MNIST()
        results_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/mnist_optimal'
    elif dataset_name == 'fashionmnist':
        train_loader, test_loader = load_FashionMNIST()
        results_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/fmnist_optimal'
    else:
        raise ValueError(f"Invalid dataset choice: {dataset_name}")

    net=Net(n_layers, threshold, beta, n_neurons, init_func).to(device)
    loss_hist, test_loss_hist, acc_hist, test_acc_hist, var_mems_train, var_mems_test, grad_train= training(net, time_steps, train_loader, test_loader)
    
    stat_filename = f'stat_grad_{dataset_name}_{init_func_name}_layers_{n_layers}_thresh_{threshold}_beta_{beta}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    grad_filename = f'grad_{dataset_name}_{init_func_name}_layers_{n_layers}_thresh_{threshold}_beta_{beta}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    var_filename = f'var_{dataset_name}_{init_func_name}_layers_{n_layers}_thresh_{threshold}_beta_{beta}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    np.save(os.path.join(results_dir, stat_filename), np.vstack((loss_hist, test_loss_hist, acc_hist, test_acc_hist))) #return a file with x=epochs, y_1line=loss, y_2line=test_loss, y_3line=acc, y_4line=test acc
    np.save(os.path.join(results_dir, grad_filename), np.array([ grad_train]))
    np.save(os.path.join(results_dir, var_filename), np.array([ var_mems_train, var_mems_test]))
    #np.save(os.path.join('/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/mnist_optimal', cur_filename), np.array([cur_train, cur_test]))
    #return file with x=epochs, y_1line=layer1, y_2line=layer2 etc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP optimal initialization experiment')
    parser.add_argument('--n_layers', type=int, default=10, help='number of layers')
    parser.add_argument('--threshold', type=float, default=1, help='firing threshold')
    parser.add_argument('--beta', type=float, default=0.5, help='beta')
    parser.add_argument('--n_neurons', type=int, default=100, help='number of neurons in hidden layers')
    parser.add_argument('--init_func', type=str, default='kaiming', help='Initialization function name')
    parser.add_argument('--time_steps', type=int, default=1, help='time steps')
    #parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'fashionmnist'], help='Dataset name')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Dataset name')
    args = parser.parse_args()
    main(args)


