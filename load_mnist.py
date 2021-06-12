import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader   

class MNISTDataset(Dataset):
    """MNIST Dataset Wrapper"""
    def __init__(self, df, transform=None):
        
        self.labels = df[["label"]].values
        self.imgs = df.drop("label", axis=1).values.reshape((len(df), 28, 28, 1))

        self.transform = transform
                  
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, image = self.labels[idx], self.imgs[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.as_tensor(image, dtype=torch.float), torch.as_tensor(label, dtype=torch.long)
    
    def visualize(self, idx=0):
        print("Label: {}\n".format(self.labels[idx]))
        plt.imshow(self.imgs[idx,:,:], cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.show()
                

def get_dataloader(args):
    """Return train,val,test dataloders"""
    # Read data
    df = pd.read_csv(args.train)
    df = df.sample(frac=1, random_state=args.seed)

    train_dataset = MNISTDataset(df[:int(args.split*len(df))])
    val_dataset = MNISTDataset(df[int(args.split*len(df)):])

    train_dataset.visualize()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    df_test = pd.read_csv(args.test)
    df_test["label"] = 0 # Create dummy labels, which won't be used
    test_dataset = MNISTDataset(df_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
