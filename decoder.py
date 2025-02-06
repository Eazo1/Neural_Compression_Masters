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


# Define Decoder
class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, input_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        
        # Initialize residual stack as a module
        self.residual_stack = residual_stack.ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=1, kernel_size=4, stride=2, padding=1)
        #self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=2, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)  # Use the residual stack
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        x = F.interpolate(x, size=(150, 150), mode="bilinear", align_corners=False)  # Resize to exact size
        return x