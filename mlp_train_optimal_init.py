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

class Net(nn.Module):
    def __init__(self, n_layers, threshold, n_neurons, init_func):
        super().__init__()

        # Initialize layers
        beta = 0.5
        num_inputs = 784
        num_hidden = n_neurons
        num_outputs = 10

        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # First Layer: LIF neurons + linear layer
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
        self.layers.append(nn.Linear(num_inputs, num_hidden))

        # Hidden Layers: LIF neurons + linear layers
        for _ in range(1, n_layers - 1):  # Excluding first and last layers
            self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))
            self.layers.append(nn.Linear(num_hidden, num_hidden))

        # Last Layer: LIF neurons
        self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract'))

        if 'threshold' in init_func.__code__.co_varnames:
            self.init_func = partial(init_func, threshold=threshold)
        else:
            self.init_func = init_func

        self.apply(lambda layer: self.init_func(layer))

    def forward(self, input_tensor, time_steps):
        num_steps = time_steps
        spike_out = []

        spike_vectors = []  # To store spike vectors for every layer
        mems = [layer.init_leaky() for layer in self.lif_layers]

        var_y_list = [[] for _ in range(len(self.layers))]  # Initialize list of lists

        for step in range(time_steps):
            for i in range(len(self.layers)):
                if i == 0:
                    spk, mems[i] = self.lif_layers[i](input_tensor, mems[i])
                    cur = self.layers[i](spk)
                    var_cur = torch.var(cur).item()
                    var_y_list[i].append(var_cur)
                else:
                    spk, mems[i] = self.lif_layers[i](cur, mems[i])
                    cur = self.layers[i](spk)
                    var_cur = torch.var(cur).item()
                    var_y_list[i].append(var_cur)

                    if i == (len(self.layers) - 1):  # Check if it's the last layer
                        spike_out.append(spk)

        if spike_out:  # Check if spike_out is not empty
            spike_out = torch.stack(spike_out, dim=0)
        else:
            spike_out = torch.tensor([], device=input_tensor.device)

        return var_y_list, spike_out

#net=Net().to(device)

def training(net, time_steps):
    
    dtype = torch.float
    
    loss_hist = []
    loss_epoch_hist=[]
    acc_hist = []
    acc_epoch_hist=[]
    
    test_loss_hist = []
    test_loss_epoch_hist=[]
    test_acc_hist = []
    test_acc_epoch_hist=[]

    var_y_train = []
    var_y_test = []
    
    #simulation parameters
    num_epochs = 150
    num_steps = time_steps
    
    #loss and optimizer
    loss_fn = SF.ce_count_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    #load data
    train_loader, test_loader = load_MNIST()
    data, targets = next(iter(train_loader))
  

    # Outer training loop
    for epoch in range(num_epochs):
        print(epoch)
        acc=0
        total=0
        var_y_batch_train = []  # Collect variance data for each batch in this epoch
        var_y_batch_test = [] 
        train_batch = iter(train_loader) 
        
        for data, targets in train_batch:
            data = torch.flatten(data,1,3).to(device)
            targets = targets.to(device)
            pixel_variance = torch.var(data)
            print(data.shape)
            print('input variance', pixel_variance)
           
            # forward pass
            net.train()
            var_y_list, spk_out = net(data, time_steps)
            var_y_batch_train.append(var_y_list) 
   

            # initialize the loss & sum over time
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
            
            #Test set
        with torch.no_grad():
            net.eval()
            for test_data, test_targets in test_loader:
                test_data = torch.flatten(test_data, 1, 3).to(device)
                test_targets = test_targets.to(device)
                # Test set forward pass
                var_y_list_test, spk_out_test = net(test_data, time_steps)
                var_y_batch_test.append(var_y_list_test)
                
                # Test set loss 
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss_fn(spk_out_test, test_targets) 

                test_loss_hist.append(test_loss.item())

                # Test set accuracy
                test_acc = SF.accuracy_rate(spk_out_test, test_targets)
                test_acc_hist.append(test_acc)

	
        #computing and storing the mean TRAIN LOSS over batches (1 value per epoch)
        loss_epoch=np.mean(loss_hist)
        loss_epoch_hist.append(loss_epoch)
        del loss_epoch
        
       # prin('done first epoch')
        
        #computing and storing the mean TRAIN ACC over batches (1 value per epoch)
        acc_epoch=np.mean(acc_hist)
        acc_epoch_hist.append(acc_epoch)
        del acc_epoch

        #computing and storing the mean TEST LOSS over batches (1 value per epoch)
        test_loss_epoch=np.mean(test_loss_hist)
        test_loss_epoch_hist.append(test_loss_epoch)
        del test_loss_epoch
        
        #computing and storing the mean TEST ACC over batches (1 value per epoch)
        test_acc_epoch=np.mean(test_acc_hist)
        test_acc_epoch_hist.append(test_acc_epoch)
        del test_acc_epoch

        var_y_train.append(np.mean(np.array(var_y_batch_train), axis=0))  # Append variance data for this epoch
        var_y_test.append(np.mean(np.array(var_y_batch_test), axis=0))

    return loss_epoch_hist, test_loss_epoch_hist, acc_epoch_hist, test_acc_epoch_hist, var_y_train, var_y_test

def main(args):
    n_layers=args.n_layers
    threshold=args.threshold
    n_neurons=args.n_neurons
    time_steps=args.time_steps

    init_functions = {
    'kaiming':kaiming_init,
    'optimal': optimal_init,
    'optimal_fs' : optimal_init_finite_size,
    # Add other initialization functions here
    }

    init_func_name = args.init_func
    if init_func_name not in init_functions:
        raise ValueError(f"Invalid initialization function: {init_func_name}")
    
    init_func = init_functions[init_func_name]

    net=Net(n_layers, threshold, n_neurons, init_func).to(device)
    loss_hist, test_loss_hist, acc_hist, test_acc_hist, var_y_train, var_y_test = training(net, time_steps)
    
    stat_filename = f'stat_mnist_{init_func_name}_layers_{n_layers}_thresh_{threshold}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    var_y_filename = f'var_y_mnist_{init_func_name}_layers_{n_layers}_thresh_{threshold}_neurons_{n_neurons}_timesteps_{time_steps}.npy'
    np.save(os.path.join('/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/mnist_optimal', stat_filename), np.vstack((loss_hist, test_loss_hist, acc_hist, test_acc_hist))) #return a file with x=epochs, y_1line=loss, y_2line=test_loss, y_3line=acc, y_4line=test acc
    np.save(os.path.join('/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/mlpSNN/weights_variance/results/mnist_optimal', var_y_filename), np.array([var_y_train, var_y_test]))
    #return file with x=epochs, y_1line=layer1, y_2line=layer2 etc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP optimal initialization experiment')
    parser.add_argument('--n_layers', type=int, default=10, help='number of layers')
    parser.add_argument('--threshold', type=float, default=1, help='firing threshold')
    parser.add_argument('--n_neurons', type=int, default=100, help='number of neurons in hidden layers')
    parser.add_argument('--init_func', type=str, default='kaiming', help='Initialization function name')
    parser.add_argument('--time_steps', type=int, default=1, help='time steps')
    args = parser.parse_args()
    main(args)


