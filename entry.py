import os
import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from load_mnist import get_dataloader
from models import SimpleMLP, SimpleCNN, SimpleVIT, BetterCNN

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default="./data/train.csv", help="Path to Train File.")
    parser.add_argument("--test", type=str, default="./data/test.csv", help="Path to Test File")
    parser.add_argument("--sample_sub", type=str, default="./data/sample_submission.csv", help="Path to Sample Submission")
    parser.add_argument("--out", type=str, default="./out/", help="Path to output models & predictions")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment Name for naming of sub files")
    parser.add_argument("--visualize", action='store_const', default=False, const=True, help="Generate Graphs of all models with torchviz")

    parser.add_argument("--model", type=str, default="CNN", help="One of MLP, CNN, VIT")
    parser.add_argument("--dropout_proba", type=float, default="0.1", help="Dropout probability to use")
    parser.add_argument("--split", type=float, default=0.9, help="Train/Val percentage split")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility Seed for Numpy & Torch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size to use for Training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate for Training")
    parser.add_argument("--n_epochs", type=int, default=15, help="Epochs to train for")
    parser.add_argument("--init_func", type=str, default="normal", help="Weight init function - One of [normal, uniform, zeros, xavier, kaiming]")
    parser.add_argument("--uniform_low", type=float, default="-0.1", help="Minimum value to sample from for uniform init")
    parser.add_argument("--uniform_high", type=float, default="0.1", help="Maximum value to sample from for uniform init")

    args = parser.parse_args()
    return args

MODELS = {"MLP": SimpleMLP, "CNN": SimpleCNN, "VIT": SimpleVIT, "CNN2": BetterCNN}

def get_model(args):
    """Create model"""
    def init_weights(module, init_range=0.02, init_func=nn.init.normal_, **kwargs):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init_func(module.weight.data, **kwargs)#mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = MODELS[args.model](dropout_proba=args.dropout_proba)

    # Weight Initilization
    if args.init_func == "normal":
        model.apply(lambda x: init_weights(x, init_func=nn.init.normal_, std=0.02)) # Default transformer std
    elif args.init_func == "uniform":
        model.apply(lambda x: init_weights(x,init_func=nn.init.uniform_, a=args.uniform_low, b=args.uniform_high))
    elif args.init_func == "zeros":
        model.apply(lambda x: init_weights(x,init_func=nn.init.zeros_))
    elif args.init_func == "xavier":
        model.apply(lambda x: init_weights(x, init_func=nn.init.xavier_normal_))
    elif args.init_func == "kaiming":
        model.apply(lambda x: init_weights(x, init_func=nn.init.kaiming_normal_, nonlinearity='relu'))
    else:
        raise NotImplementedError

    # Statistics
    print("Parameters in Model: ", count_parameters(model))

    return model
    

def visualize(model, x, args):
    """
    Create graph of model
    """
    import torchviz
    dot = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(os.path.join(args.out, args.exp))


def train_val(train_dataloader, val_dataloader, args):
    """
    Perform training & validation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    val_loss_min = np.Inf

    for epoch in range(1, args.n_epochs+1):
        # Keep track of train & val loss & accuracy
        train_loss, val_loss = 0.0, 0.0
        train_correct, val_correct = [], []
        
        ###################
        # Train the model #
        ###################
        # Activate Dropout & Co
        model.train()
        for data, target in train_dataloader:
            # Move tensors to correct device (GPU if Cuda available)
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the batch loss & squeeze target to get correct shape
            loss = criterion(output, target.squeeze())
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step
            optimizer.step()
            # Update correct & training loss
            _, pred = torch.max(output, 1)
            train_correct.extend((target.squeeze() == pred).detach().tolist())
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # Validate the model #
        ######################
        # Deactivate Dropout & Co
        model.eval()
        for data, target in val_dataloader:
            # Deactivate autograd
            with torch.no_grad():
                # Move tensors to correct device (GPU if Cuda available)
                data, target = data.to(device), target.to(device)
                # Forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # Calculate the batch loss
                loss = criterion(output, target.squeeze())
                # Update correct & validation loss 
                _, pred = torch.max(output, 1)
                val_correct.extend((target.squeeze() == pred).tolist())
                val_loss += loss.item()*data.size(0)
                    
        # Calculate average losses
        train_loss = train_loss/len(train_dataloader.dataset)
        val_loss = val_loss/len(val_dataloader.dataset)
        
        # Calculate accuracy
        train_acc = sum(train_correct) / len(train_correct)
        val_acc = sum(val_correct) / len(val_correct)
            
        # Print training/validation statistics 
        print('Epoch: {} \tTrain Loss: {:.6f} \tTrain Accuracy: {:.6f} \tVal Loss: {:.6f} \tVal Accuracy: {:.6f}'.format(
            epoch, train_loss, train_acc, val_loss, val_acc))
        
        # Save model if validation loss has decreased, i.e. early stopping
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            val_loss_min,
            val_loss))
            torch.save(model.state_dict(), os.path.join(args.out, 'model_{}.pt'.format(args.exp)))
            val_loss_min = val_loss


def predict(test_dataloader, args):
    """
    Perform testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MODELS[args.model]()
    model.load_state_dict(torch.load(os.path.join(args.out, 'model_{}.pt'.format(args.exp))))
    model = model.to(device)

    test_correct = []
    # Deactivate Dropout & Co
    model.eval()
    for data, target in test_dataloader:
        # Deactivate autograd
        with torch.no_grad():
            # Move tensors to correct device (GPU if Cuda available)
            data, target = data.to(device), target.to(device)
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Update correct
            _, pred = torch.max(output, 1)
            test_correct.extend((target.squeeze() == pred).tolist())
    test_acc = sum(test_correct) / len(test_correct)
    print('Final Test Accuracy of Best Epoch: {:.6f}'.format(test_acc))
        
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args)

    # Get Graphs of models
    if args.visualize:
        x = next(iter(train_dataloader))
        for model in MODELS.values():
            args.exp = str(model.__name__)
            visualize(model(), x, args)
    # Train, Validate, Test
    else:
        train_val(train_dataloader, val_dataloader, args)
        predict(test_dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)