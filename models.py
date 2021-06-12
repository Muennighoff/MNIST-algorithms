import torch
import torch.nn as nn
import torch.nn.functional as F


### Simple Algorithms to solve MNIST ###
### Which one will stand the test of time? ###


### Simple Multilayer Perceptron ###
    
class SimpleMLP(nn.Module):
    """
    Simple MLP; Could benefit from normalizations

    Default Params: 1885432
    """
    def __init__(self, dropout_proba=0.1):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(28*28, 28*28*2)
        self.fc2 = nn.Linear(28*28*2, 28*28//2)
        self.fc3 = nn.Linear(28*28//2, 28*28//8)
        self.fc4 = nn.Linear(28*28//8, 10)
        
        self.dropout = nn.Dropout(p=dropout_proba)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1) # BSx784
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x


### Simple Convolutional Neural Network ###

class SimpleCNN(nn.Module):
    """
    Simple CNN - To improve, could add BatchNorm2d & more dropouts

    Default Params: 274882
    """
    def __init__(self, dropout_proba=0.1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_pad = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.fc1 = nn.Linear(4*4*64, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(p=dropout_proba)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[-1], x.shape[1], x.shape[2])) # > BSx1x28x28 
        
        x = self.pool(F.relu(self.conv1(x))) # > 4x14x14 
        x = self.pool(F.relu(self.conv2(x))) # > 16x7x7
        x = self.pool_pad(F.relu(self.conv3(x)))  # > 64x4x4
        
        # Reshape & FCs
        x = x.view(-1, 4*4*64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BetterCNN(nn.Module):
    """
    SimpleCNN + Normalizations

    Default Params: 275562
    """
    def __init__(self, dropout_proba=0.1):
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_pad = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.fc1 = nn.Linear(4*4*64, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(p=dropout_proba)

        self.norm2d_1 = nn.BatchNorm2d(4)
        self.norm2d_2 = nn.BatchNorm2d(16)
        self.norm2d_3 = nn.BatchNorm2d(64)

        self.norm1d_1 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[-1], x.shape[1], x.shape[2])) # > BSx1x28x28 
        
        x = self.pool(F.relu(self.norm2d_1(self.conv1(x)))) # > 4x14x14
        x = self.pool(F.relu(self.norm2d_2(self.conv2(x)))) # > 16x7x7
        x = self.pool_pad(F.relu(self.norm2d_3(self.conv3(x))))  # > 64x4x4
        
        # Reshape & FCs
        x = x.view(-1, 4*4*64)
        x = self.dropout(x)
        x = F.relu(self.norm1d_1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

### Simple Vision Transformer ###

class SelfAttention(nn.Module):
    """
    Simplified Attention without heads and no extra hidden dim, i.e. only a sequence of length dim
    (Original Attention creates a hidden dim for each word/pixel with nn.Embedding; but better performance w/o this for pixels)

    Default Params: 4314418
    """
    def __init__(self, dim=28*28, inner_dim=28*28):
        super().__init__()

        self.inner_dim = inner_dim
        
        self.tokeys = nn.Linear(dim, inner_dim, bias=False)
        self.toqueries = nn.Linear(dim, inner_dim, bias=False)
        self.tovalues = nn.Linear(dim, inner_dim, bias=False)
        
        if dim != inner_dim:
            self.scale = nn.Linear(inner_dim, dim)
        else:
            self.scale = nn.Identity()
        
    def forward(self, x):
        """
        x: bs, dim
        out: bs, dim
        """
        b, _ = x.size()
        
        # Create Q,K,V
        queries = self.toqueries(x).view(b, self.inner_dim, 1)
        keys = self.tokeys(x).view(b, self.inner_dim, 1)
        values = self.tovalues(x).view(b, self.inner_dim, 1)

        # To save computational memory, take the 4th root
        queries = queries / (1 ** (1/4))
        keys = keys / (1 ** (1/4))
        
        # Compute dot product of q & the transpose of k (like dot of x & xT)
        dot = torch.bmm(queries, keys.transpose(1,2))        
        # dot has now size (b*h, t, t) & is equal to our raw weights
        
        # Take the softmax
        dot = F.softmax(dot, dim=2)
        
        # dot now contains [0, 1] normalized row weights
        out = torch.bmm(dot, values).view(b, self.inner_dim)
        
        # Scale back out
        return self.scale(out) # (bsxt)
    
# Much simplified version of https://openreview.net/pdf?id=YicbFdNTTy
class SimpleVIT(nn.Module):
    """
    Simplified Version of the Vision Transformer with only one transformer layer

    Default Params: 4314418
    """
    def __init__(self, dim=28*28, inner_dim=28*28, dropout_proba=0.1):
        super(SimpleVIT, self).__init__()

        # This is quite different from normal transformers but a try at capturing 2D information
        # Since attention is position invariant, we need to somehow encode the position of pixels
        # Note that these embeddings are only for the first row & column of the data
        # TO get positional embeddings for each pixel it should be int(dim**(1/2)), int(dim**(1/2)) instead
        # However I somehow got better results with the one below on this specific task
        self.pos_embedding_w = nn.Parameter(torch.randn(1, int(dim**(1/2)), 1, 1))
        self.pos_embedding_h = nn.Parameter(torch.randn(1, 1, int(dim**(1/2)), 1))

        # Default from https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py
        self.dropout = nn.Dropout(dropout_proba)
        
        self.norm = nn.LayerNorm(dim)
        
        self.attention = SelfAttention(dim, inner_dim)
  
        # Simple MLP after Transformer; Use GELU activation as slightly outperforms RELU due to smoother rounding at 0
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            self.dropout,
            nn.Linear(dim*2, dim),
            self.dropout
        )
        
        self.classifier = nn.Linear(dim, 10)
        
    def transformer(self, x):
        # Norm & Attention & Dropout with Residual Connection
        x = self.dropout(self.attention(self.norm(x))) + x
        
        # Norm & MLP with Residual Connection
        x = self.mlp(self.norm(x)) + x
        
        return x
    
    def forward(self, x):
        bs, h, w, c = x.shape # BS, Height, Width, Channels
        x += self.pos_embedding_h + self.pos_embedding_w
        x = x.reshape((bs, h*w))

        # For bigger images than MNIST, make h*w smaller here via e.g. patches
        
        # Transformer incl. attention
        x = self.transformer(x)
        
        # Classify
        x = self.classifier(self.norm(x))
        
        return x