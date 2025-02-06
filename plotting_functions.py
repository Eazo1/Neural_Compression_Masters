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


# Function to denormalize images for visualization
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean

def display_images(original_images, reconstructed_images, num_images=16, step=0):
    """
    Displays original and reconstructed images side by side.

    Parameters:
    - original_images (torch.Tensor): Original images tensor of shape [batch, channels, height, width].
    - reconstructed_images (torch.Tensor): Reconstructed images tensor of the same shape.
    - num_images (int): Number of images to display.
    """
    # Apply 2D Fourier Transform on the images (shifted for visualization)
    #original_images = torch.fft.ifft2(original_images)
    #reconstructed_images = torch.fft.ifft2(reconstructed_images)
    
    # Take only the real part of the Fourier-transformed images
    original_images_real = original_images #original_images.real  # Extract real part
    reconstructed_images_real = reconstructed_images #reconstructed_images.real  # Extract real part
    
    # Denormalize images (adjust mean and std as per your data)
    original_images_real = denormalize(original_images_real, mean=(0.0031,), std=(0.0352,))
    reconstructed_images_real = denormalize(reconstructed_images_real, mean=(0.0031,), std=(0.0352,))
    
    # Convert images to numpy
    original_images_real = original_images_real.cpu().numpy()
    reconstructed_images_real = reconstructed_images_real.cpu().numpy()
    
    # Handle dimensions:
    # If images have multiple channels, you might want to combine them or select one channel for visualization.
    # Here, we'll average across the channel dimension if there are multiple channels.
    if original_images_real.ndim == 4 and original_images_real.shape[1] > 1:
        original_images_real = original_images_real.mean(axis=1)  # Shape: [batch, height, width]
    elif original_images_real.ndim == 4 and original_images_real.shape[1] == 1:
        original_images_real = original_images_real.squeeze(1)  # Shape: [batch, height, width]
    
    if reconstructed_images_real.ndim == 4 and reconstructed_images_real.shape[1] > 1:
        reconstructed_images_real = reconstructed_images_real.mean(axis=1)  # Shape: [batch, height, width]
    elif reconstructed_images_real.ndim == 4 and reconstructed_images_real.shape[1] == 1:
        reconstructed_images_real = reconstructed_images_real.squeeze(1)  # Shape: [batch, height, width]
    
    # Ensure the number of images to display does not exceed the batch size
    batch_size = original_images_real.shape[0]
    num_images = min(num_images, batch_size)
    
    # Create a figure with 2 rows: Original and Reconstructed
    f, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        # Display original images
        ax = axes[0, i] if num_images > 1 else axes[0]
        ax.imshow(original_images_real[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Original', fontsize=12)
        
        # Display reconstructed images
        ax = axes[1, i] if num_images > 1 else axes[1]
        ax.imshow(reconstructed_images_real[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    save_directory = '/share/nas2_3/amahmoud/week5/galaxy_out/reconstructed_vqvae.pdf'
    # Log the figure with wandb
    wandb.log({"Reconstructed Images": wandb.Image(f)}, step=None)
    plt.savefig(save_directory)
    plt.show()
    plt.close(f)