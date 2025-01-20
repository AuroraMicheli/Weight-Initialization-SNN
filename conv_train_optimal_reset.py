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
from helpers_conv import *



np.random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
torch.manual_seed(42)  # Set random seed for PyTorch for reproducibility
torch.cuda.manual_seed_all(42) 

# Check and print if GPU is being used
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

class Net(nn.Module):
    def __init__(self, n_layers, threshold, beta, n_channels, init_func):
        super().__init__()

        # Initialize layers
        beta = beta
        num_channels = n_channels
        num_outputs = 10

        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # First Layer: LIF neurons + conv layer
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)) #HERE 3 vs 1 

        # Hidden Layers: LIF neurons + conv layers
        for _ in range(n_layers - 3):  # Excluding first and last 2 layers
            self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
            self.layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1))

        # Last 2 Layers: lif - lineaer - lif_output
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_channels * 28 * 28, num_outputs)) #HERE 32 vs 28
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))

        if 'threshold' in init_func.__code__.co_varnames and 'beta' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold, beta=beta)
        elif 'threshold' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold)
        else:
            self.init_func = init_func
        
        print(f"Initialization function used: {self.init_func.func.__name__ if isinstance(self.init_func, partial) else self.init_func.__name__}")
        
        self.apply(lambda layer: self.init_func(layer))

    def forward(self, input_tensor, time_steps):
        num_steps = time_steps
        spike_out = []
        #spike_counts_list = [[] for _ in range(len(self.lif_layers))]  # Initialize lists for spike counts

        mems = [layer.init_leaky() for layer in self.lif_layers]
        #var_mems_list = [[] for _ in range(len(self.layers))]  # Initialize lists for variance values

        for step in range(time_steps):
            for i in range(len(self.lif_layers)):
                if i == (len(self.lif_layers) - 1):  # Last LIF layer: only append spike_out
                    spk, mems[i] = self.lif_layers[i](cur, mems[i])
                    spike_out.append(spk)
                else:
                    if i == 0:
                        spk, mems[i] = self.lif_layers[i](input_tensor, mems[i])
                    else:
                        spk, mems[i] = self.lif_layers[i](cur, mems[i])

                    if isinstance(self.layers[i], nn.Conv2d):
                        cur = self.layers[i](spk)  # Apply convolutional layer
                    elif isinstance(self.layers[i], nn.Linear):
                    # Flatten the spike tensor before passing to linear layer
                        cur = spk.view(spk.size(0), -1)  

        # Stack spike_out into a tensor
        if spike_out:
            spike_out = torch.stack(spike_out, dim=0)
        else:
            spike_out = torch.tensor([], device=input_tensor.device)

        return spike_out

#net=Net().to(device)

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
    spike_counts_train = []
    spike_counts_test = []
    
    num_epochs = 200
    num_steps = time_steps
    
    # Loss function and optimizer
    loss_fn = SF.ce_count_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay = 0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Outer training loop
    for epoch in range(num_epochs):
        print(epoch)
        loss_epoch = []
        acc_epoch = []
        test_loss_epoch = []
        test_acc_epoch = []


        # Training phase
        net.train()
        for data, targets in train_loader:
            #data = torch.flatten(data, 1, 3).to(device)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            spk_out = net(data, time_steps)

            # Initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(spk_out, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()

            # Store loss and accuracy history for future plotting
            loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_out, targets)
            acc_hist.append(acc)

        # Test phase
        with torch.no_grad():
            net.eval()
            for test_data, test_targets in test_loader:
                #test_data = torch.flatten(test_data, 1, 3).to(device)
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                spk_out_test = net(test_data, time_steps)


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
        print('loss', loss_epoch)
        # Compute and store mean TRAIN ACC over batches
        acc_epoch = np.mean(acc_hist)
        acc_epoch_hist.append(acc_epoch)
        print('acc', acc_epoch)
        # Compute and store mean TEST LOSS over batches
        test_loss_epoch = np.mean(test_loss_hist)
        test_loss_epoch_hist.append(test_loss_epoch)

        # Compute and store mean TEST ACC over batches
        test_acc_epoch = np.mean(test_acc_hist)
        test_acc_epoch_hist.append(test_acc_epoch)


    return loss_epoch_hist, test_loss_epoch_hist, acc_epoch_hist, test_acc_epoch_hist



def main(args):
    n_layers=args.n_layers
    threshold=args.threshold
    beta=args.beta
    n_channels=args.n_channels
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
    elif dataset_name == 'cifar10':
        train_loader, test_loader = load_CIFAR10()
        results_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/cifar10_optimal'
    else:
        raise ValueError(f"Invalid dataset choice: {dataset_name}")

    net=Net(n_layers, threshold, beta, n_channels, init_func).to(device)
    loss_hist, test_loss_hist, acc_hist, test_acc_hist = training(net, time_steps, train_loader, test_loader)
    
    stat_filename = f'stat_{dataset_name}_{init_func_name}_layers_{n_layers}_thresh_{threshold}_beta_{beta}_channels_{n_channels}_timesteps_{time_steps}.npy'
    #var_y_filename = f'var_spk_{dataset_name}_{init_func_name}_layers_{n_layers}_thresh_{threshold}_beta_{beta}_channels_{n_channels}_timesteps_{time_steps}.npy'
    #cur_filename = f'cur_mnist_{init_func_name}_layers_{n_layers}_thresh_{threshold}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    np.save(os.path.join(results_dir, stat_filename), np.vstack((loss_hist, test_loss_hist, acc_hist, test_acc_hist))) #return a file with x=epochs, y_1line=loss, y_2line=test_loss, y_3line=acc, y_4line=test acc
    #np.save(os.path.join(results_dir, var_y_filename), np.array([var_mems_train, var_mems_test, spike_counts_train, spike_counts_test]))
    #np.save(os.path.join('/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/mnist_optimal', cur_filename), np.array([cur_train, cur_test]))
    #return file with x=epochs, y_1line=layer1, y_2line=layer2 etc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP optimal initialization experiment')
    parser.add_argument('--n_layers', type=int, default=10, help='number of layers')
    parser.add_argument('--threshold', type=float, default=1, help='firing threshold')
    parser.add_argument('--beta', type=float, default=0.5, help='beta')
    parser.add_argument('--n_channels', type=int, default=64, help='number of channels in hidden layers')
    parser.add_argument('--init_func', type=str, default='kaiming', help='Initialization function name')
    parser.add_argument('--time_steps', type=int, default=1, help='time steps')
    #parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'fashionmnist'], help='Dataset name')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Dataset name')
    args = parser.parse_args()
    main(args)


