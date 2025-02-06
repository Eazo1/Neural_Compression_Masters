import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tarfile

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import combinations, product

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

import residual_stack


# Define Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        #self.embeddings.weight.data.uniform_(-1000, 1000)

    def forward(self, x):
        # Flatten input
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(x_flat, self.embeddings.weight.t())
        )

        # Get the encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

