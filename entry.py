import os
import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from load_mnist import get_dataloader
from models import SimpleMLP, SimpleCNN, SimpleVIT

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default="./data/train.csv", help="Path to Train File.")
    parser.add_argument("--test", type=str, default="./data/train.csv", help="Path to Test File")
    parser.add_argument("--sample_sub", type=str, default="./data/sample_submission.csv", help="Path to Sample Submission")
    parser.add_argument("--out", type=str, default="./out/", help="Path to output models & predictions")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment Name for naming of sub files")
    parser.add_argument("--visualize", action='store_const', default=False, const=True, help="Generate Graphs of all models with torchviz")

    parser.add_argument("--configure", action='store_const', default=False, const=True, help="Configure own presets (by default reproduces lab report table)")
    parser.add_argument("--model", type=str, default="CNN", help="One of MLP, CNN, VIT")
    parser.add_argument("--split", type=int, default=0.9, help="Train/Val percentage split")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility Seed for Numpy & Torch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size to use for Training")
    parser.add_argument("--init_func", type=str, default="normal", help="Weight init function - One of [normal, xavier, kaiming]")

    args = parser.parse_args()
    return args

MODELS = {"MLP": SimpleMLP, "CNN": SimpleCNN, "VIT": SimpleVIT}

def get_model(args):
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

    model = MODELS[args.model]()

    # Initilization
    if args.init_func == "normal":
        model.apply(lambda x: init_weights(x, init_func=nn.init.normal_, std=0.02)) # Default transformer std
    elif args.init_func == "uniform":
        model.apply(lambda x: init_weights(x,init_func=nn.init.uniform_, a=0.0, b=1.0))
    elif args.init_func == "zeros":
        model.apply(lambda x: init_weights(x,init_func=nn.init.zeros_, a=0.0, b=1.0))
    elif args.init_func == "xavier":
        model.apply(lambda x: init_weights(x, init_func=nn.init.xavier_normal_))

    # Statistics
    print("Parameters in Model: ", count_parameters(model))

    return model
    

def visualize(model, x, args):
    import torchviz
    dot = torchviz.make_dot(model(x), params=dict(model.named_parameters())).render()
    dot.format = 'png'
    dot.render(os.path.join(args.out, args.exp))


def train_val(train_dataloader, val_dataloader, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 20
    val_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # Keep track of train & val loss & accuracy
        train_loss = 0.0
        val_loss = 0.0
        
        train_correct = []
        val_correct = []
        
        ###################
        # Train the model #
        ###################
        model.train()
        for data, target in train_dataloader:
            # Move tensors to correct device (GPU if Cuda available)
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss & squeeze target to get correct shape
            loss = criterion(output, target.squeeze())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update correct & training loss
            _, pred = torch.max(output, 1)
            train_correct.extend((target.squeeze() == pred).detach().tolist())
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in val_dataloader:
            # Deactivate autograd
            with torch.no_grad():
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target.squeeze())
                # update correct & validation loss 
                _, pred = torch.max(output, 1)
                val_correct.extend((target.squeeze() == pred).tolist())
                val_loss += loss.item()*data.size(0)
                    
        # calculate average losses
        train_loss = train_loss/len(train_dataloader.dataset)
        val_loss = val_loss/len(val_dataloader.dataset)
        
        # calculate accuracy
        train_acc = sum(train_correct) / len(train_correct)
        val_acc = sum(val_correct) / len(val_correct)
            
        # print training/validation statistics 
        print('Epoch: {} \tTrain Loss: {:.6f} \tTrain Accuracy: {:.6f} \tVal Loss: {:.6f} \tVal Accuracy: {:.6f}'.format(
            epoch, train_loss, train_acc, val_loss, val_acc))
        
        # save model if validation loss has decreased, i.e. early stopping
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            val_loss_min,
            val_loss))
            torch.save(model.state_dict(), os.path.join(args.out, 'model_{}.pt'.format(args.exp)))
            val_loss_min = val_loss


def predict(test_dataloader, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MODELS[args.model]()
    model.load_state_dict(torch.load(os.path.join(args.out, 'model_mnist.pt')))
    model.to(device)

    model.eval()
    preds = []

    # iterate over test data
    with torch.no_grad():
        for data, _ in test_dataloader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # Add to preds
            preds.extend(pred.cpu().tolist())

    sample_sub = pd.read_csv(args.sample_sub)
    sample_sub["Label"] = preds
    print("Submission Head:\n", sample_sub.head(5))
    sample_sub.to_csv(os.path.join(args.out, args.exp), index=False)


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args)

    # Get Graphs of models & exit
    if args.visualize:
        x = next(iter(train_dataloader))
        for model in MODELS.values():
            args.exp = str(model.__name__)
            visualize(model(), x, args)
        exit()

    # Train Models
    if args.configure:
        train_val(train_dataloader, val_dataloader, args)
        predict(test_dataloader, args)
    # Reproduce table
    else:
        pass

if __name__ == "__main__":
    args = parse_args()
    main(args)