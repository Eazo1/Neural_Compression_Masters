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


# Define residual stack
class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1)
    
    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(num_hiddens, num_residual_hiddens)
             for _ in range(num_residual_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)
