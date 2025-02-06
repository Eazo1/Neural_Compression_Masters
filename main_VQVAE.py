import os
import subprocess
import tempfile
import sys

#sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import tarfile

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange

import numpy as np
import pandas as pd
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

from scipy.fftpack import ifft2
from scipy.fftpack import fft2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import combinations, product, cycle

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.mixture import GaussianMixture
import wandb


from MiraBest_N import MiraBest_N, MBFRFull, MBFRConfident, MBFRUncertain, MBHybrid, MBRandom
from datasets import RGZ108k

###
import residual_stack
import encoder
import decoder
import vectorquantizer
import plotting_functions
###

# GALAXY ZOO


batch_size = 4
image_size = 150
num_training_updates = 10000
num_hiddens = 256
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 256
commitment_cost = 0.25
vq_use_ema = False  # Not implementing EMA
decay = 0.99
learning_rate = 2e-4

weight_original = 1.0
weight_log = 1.0 - weight_original

print(torch.__version__)            
print(torch.cuda.is_available())     # Should return True if CUDA is properly installed
print(torch.version.cuda)            # Confirm CUDA version used by PyTorch
print(torch.cuda.get_device_name(0)) # Confirm the name of the GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wandb.login(key="7391c065d23aad000052bc1f7a3a512445ae83d0")
wandb.init(
    config={
        "embedding_dim": 64,
        "num_embeddings": 256,
        "architecture": "VQ-VAE",
        "dataset": "CIFAR-10",
        "num_training_updates": 250000,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_original": weight_original,
    },
    reinit=True,

)

from datasets import RGZ108k

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def log_transform(x):
    return torch.log1p(x)

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure only one tensor is returned
        return torch.tensor(self.data[idx], dtype=torch.float32).to(self.device)


def fit_mog(image_data):
    # Get the spatial coordinates (x, y) and the intensity (pixel value)
    h, w = image_data.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w))
    
    # Flatten the coordinates and pixel values into a 2D array of shape (num_pixels, 3)
    pixels = np.vstack([x_coords.ravel(), y_coords.ravel(), image_data.ravel()]).T
    
    # Fit a Gaussian Mixture Model to the spatial coordinates and pixel values
    gmm = GaussianMixture(n_components=2)  # You can change n_components to fit more Gaussians
    gmm.fit(pixels)
    
    return gmm

#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5,), (0.5,))
#])

# Load precomputed datasets

# Save train data as memory-mapped NumPy arrays
train_data = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_train_original.pth")

for image in train_data:
    print('image shape: ', image.shape)

np.save('/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy', np.stack([t.numpy() for t in train_data]))

train_data_log = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_train_log.pth")
np.save('/share/nas2_3/amahmoud/week5/galaxy_out/train_data_log.npy', np.stack([t.numpy() for t in train_data_log]))

valid_data_original = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_valid_original.pth")
np.save('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', np.stack([t.numpy() for t in valid_data_original]))

valid_data_log = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_valid_log.pth")
np.save('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_log.npy', np.stack([t.numpy() for t in valid_data_log]))


# Load and memory map other datasets
train_data_mmap = np.load('/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy', mmap_mode='r')
train_data_log_mmap = np.load('/share/nas2_3/amahmoud/week5/galaxy_out/train_data_log.npy', mmap_mode='r')
valid_data_original_mmap = np.load('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', mmap_mode='r')
valid_data_log_mmap = np.load('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_log.npy', mmap_mode='r')

# Create datasets
train_dataset = MemoryMappedDataset(train_data_mmap, device)
train_log_dataset = MemoryMappedDataset(train_data_log_mmap, device)
valid_dataset = MemoryMappedDataset(valid_data_original_mmap, device)
valid_log_dataset = MemoryMappedDataset(valid_data_log_mmap, device)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
train_loader_log = DataLoader(train_log_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
valid_loader_log = DataLoader(valid_log_dataset, batch_size=4, shuffle=False)

# Debug: Inspect the shape of a batch from train_loader
for batch in train_loader:
    print("Shape of a batch from train_loader:", batch.shape)  # Check shape
    break

# Debug: Inspect the shape of a batch from train_loader_log
for batch in train_loader_log:
    print("Shape of a batch from train_loader_log:", batch.shape)  # Check shape
    break

print('Valid batch info etc:')

for batch in valid_loader:
    print(f"Batch type: {type(batch)}")
    if isinstance(batch, (tuple, list)):
        print(f"Batch length: {len(batch)}")
        print(f"First element shape: {batch[0].shape}")
    else:
        print(f"Batch shape: {batch.shape}")
    break


print("DataLoaders are ready!")

# Create infinite iterators for training
#train_loader_iter_original = cycle(precomputed_train_fft_original)
#train_loader_iter_log = cycle(precomputed_train_fft_log)


# Build the models


encoder_vq = encoder.Encoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
pre_vq_conv1 = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1).to(device)
vq_vae = vectorquantizer.VectorQuantizer(num_embeddings, embedding_dim, commitment_cost).to(device)
decoder_vq = decoder.Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, input_dim=embedding_dim).to(device)

# Optimizer
optimizer = optim.Adam(
    list(encoder_vq.parameters())
    + list(pre_vq_conv1.parameters())
    + list(vq_vae.parameters())
    + list(decoder_vq.parameters()),
    lr=learning_rate,
)

scheduler = CosineAnnealingLR(optimizer, T_max=num_training_updates, eta_min=1e-6)


# Training Loop
# Initialize arrays to store iterations, training, and validation metrics
iterations = []
valid_iterations = []
train_res_recon_error = []
train_res_perplexity = []
train_bits_per_dim = []  # To store bits per dimension
valid_res_recon_error = []
valid_res_perplexity = []
valid_bits_per_dim = []  # To store validation bits per dimension

# Set models to training mode
encoder_vq.train()
decoder_vq.train()
vq_vae.train()

# Create an infinite iterator over the train_loader
#train_loader_iter = cycle(train_loader)

print("beginning training loop")

# Initialize the update counter
i = 0
# Start the training loop
while i < num_training_updates:
    for (images_original), (images_log) in zip(train_loader, train_loader_log):
        images_original = images_original.to(device).squeeze(2)
        images_log = images_log.to(device).squeeze(2)

        # Forward pass through encoder and quantizer for original channel
        z_original = encoder_vq(images_original)
        z_original = pre_vq_conv1(z_original)
        quantized_original, vq_loss_original, perplexity_original = vq_vae(z_original)

        # Reconstruct original images
        x_recon_original = decoder_vq(quantized_original)

        # Forward pass through encoder and quantizer for log-scaled channel
        z_log = encoder_vq(images_log)
        z_log = pre_vq_conv1(z_log)
        quantized_log, vq_loss_log, perplexity_log = vq_vae(z_log)

        # Reconstruct log-scaled images
        x_recon_log = decoder_vq(quantized_log)

        # Reconstruction loss (NMSE)
        data_variance_original = torch.var(images_original)
        data_variance_log = torch.var(images_log)

        recon_loss_original = F.mse_loss(x_recon_original, images_original, reduction='sum') / data_variance_original
        recon_loss_log = F.mse_loss(x_recon_log, images_log, reduction='sum') / data_variance_log

        recon_loss = weight_original * recon_loss_original + weight_log * recon_loss_log
        vq_loss = weight_original * vq_loss_original + weight_log * vq_loss_log
        total_loss = recon_loss + vq_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Calculate bits per dimension
        nll_original = F.mse_loss(x_recon_original, images_original, reduction='sum')
        nll_log = F.mse_loss(x_recon_log, images_log, reduction='sum')

        bits_per_dim_original = nll_original / (22500 * np.log(2))
        bits_per_dim_log = nll_log / (22500 * np.log(2))
        bits_per_dim = weight_original * bits_per_dim_original + weight_log * bits_per_dim_log

        # Log metrics
        wandb.log({
            "train/total_loss": total_loss.item(),
            "train/perplexity_original": perplexity_original.item(),
            "train/perplexity_log": perplexity_log.item(),
            "train/bits_per_dim": bits_per_dim.item()
        })

        # Track losses, perplexity, bits per dimension, and iterations for training
        train_res_recon_error.append(total_loss.item())
        train_res_perplexity.append((perplexity_original.item() + perplexity_log.item()) / 2)
        train_bits_per_dim.append(bits_per_dim.item())
        iterations.append(i)

        # Increment the counter
        i += 1

        # Print training status every 100 iterations
        if i % 100 == 0:
            print(f"{i} iterations, loss: {np.mean(train_res_recon_error[-100:]):.3f}, "
                  f"perplexity: {np.mean(train_res_perplexity[-100:]):.3f}, "
                  f"bits per dimension: {np.mean(train_bits_per_dim[-100:]):.5f}")

            del images_original, images_log, z_original, z_log, quantized_original, quantized_log, x_recon_original, x_recon_log
            torch.cuda.empty_cache()

        # Perform validation every 1000 iterations
        if i % 1000 == 0:
            encoder_vq.eval()
            decoder_vq.eval()
            vq_vae.eval()
            scheduler.step()

            valid_recon_error = []
            valid_perplexity = []
            valid_bits_per_dim_batch = []

            with torch.no_grad():
                # Iterate over both original and log-transformed validation loaders
                for (val_images_original), (val_images_log) in zip(valid_loader, valid_loader_log):
                    val_images_original = val_images_original.to(device).squeeze(2)
                    val_images_log = val_images_log.to(device).squeeze(2)

                    # Forward pass for validation data
                    # Original images
                    z_original = encoder_vq(val_images_original)
                    z_original = pre_vq_conv1(z_original)
                    quantized_original, vq_loss_original, perplexity_original = vq_vae(z_original)
                    x_recon_original = decoder_vq(quantized_original)

                    # Log-transformed images
                    z_log = encoder_vq(val_images_log)
                    z_log = pre_vq_conv1(z_log)
                    quantized_log, vq_loss_log, perplexity_log = vq_vae(z_log)
                    x_recon_log = decoder_vq(quantized_log)

                    # Reconstruction loss
                    data_variance_original = torch.var(val_images_original)
                    data_variance_log = torch.var(val_images_log)

                    recon_loss_original = F.mse_loss(x_recon_original, val_images_original, reduction='sum') / data_variance_original
                    recon_loss_log = F.mse_loss(x_recon_log, val_images_log, reduction='sum') / data_variance_log

                    recon_loss = weight_original * recon_loss_original + weight_log * recon_loss_log
                    vq_loss = weight_original * vq_loss_original + weight_log * vq_loss_log
                    loss = recon_loss + vq_loss

                    # Bits per dimension
                    nll_original = F.mse_loss(x_recon_original, val_images_original, reduction='sum')
                    nll_log = F.mse_loss(x_recon_log, val_images_log, reduction='sum')

                    bits_per_dim_original = nll_original / (22500 * np.log(2))
                    bits_per_dim_log = nll_log / (22500 * np.log(2))
                    bits_per_dim = weight_original * bits_per_dim_original + weight_log * bits_per_dim_log

                    # Accumulate validation metrics
                    valid_recon_error.append(loss.item())
                    valid_perplexity.append((perplexity_original.item() + perplexity_log.item()) / 2)
                    valid_bits_per_dim_batch.append(bits_per_dim.item())

            del val_images_original, val_images_log, z_original, z_log, quantized_original, quantized_log, x_recon_original, x_recon_log
            torch.cuda.empty_cache()

            # Log validation metrics
            wandb.log({
                "validation/loss": np.mean(valid_recon_error),
                "validation/perplexity": np.mean(valid_perplexity),
                "validation/bits_per_dim": np.mean(valid_bits_per_dim_batch)
            })

            valid_res_recon_error.append(np.mean(valid_recon_error))
            valid_res_perplexity.append(np.mean(valid_perplexity))
            valid_bits_per_dim.append(np.mean(valid_bits_per_dim_batch))
            valid_iterations.append(i)

            print(f"Validation loss: {np.mean(valid_recon_error):.3f}, "
                  f"perplexity: {np.mean(valid_perplexity):.3f}, "
                  f"bits per dimension: {np.mean(valid_bits_per_dim_batch):.5f}")

            # Set models back to training mode
            encoder_vq.train()
            decoder_vq.train()
            vq_vae.train()

        # Break the loop if the maximum number of updates is reached
        if i >= num_training_updates:
            break






# Evaluation (optional)
encoder_vq.eval()
decoder_vq.eval()
vq_vae.eval()

encoder_vq.eval()
decoder_vq.eval()
vq_vae.eval()

with torch.no_grad():
    for batch in valid_loader:
        # Since batch is a tensor, directly use it
        images_original = batch.to(device).squeeze(2)  # Squeeze unnecessary dimensions

        # Forward pass
        z = encoder_vq(images_original)
        z = pre_vq_conv1(z)
        quantized, vq_loss, perplexity = vq_vae(z)
        x_recon = decoder_vq(quantized)

        # Compute loss
        nll = F.mse_loss(x_recon, images_original, reduction='sum')

        # Compute bits per dimension
        bits_per_dim = nll / (22500 * np.log(2))  # Adjusted denominator

        print('Bits per dimension:', bits_per_dim.item())
        break  # Remove this break to evaluate the full validation set




f = plt.figure(figsize=(16, 8))

# First subplot: Training and Validation Loss
ax1 = f.add_subplot(1, 2, 1)
ax1.plot(iterations, train_res_recon_error, label='Train Loss')
ax1.plot(valid_iterations, valid_res_recon_error, label='Validation Loss', marker='o')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.legend()

# Second subplot: Training and Validation Perplexity
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(iterations, train_res_perplexity, label='Train Perplexity')
ax2.plot(valid_iterations, valid_res_perplexity, label='Validation Perplexity', marker='o')
ax2.set_title('Training and Validation Perplexity')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Perplexity')
ax2.legend()

# Display the plots


save_directory = '/share/nas2_3/amahmoud/week5/galaxy_out/loss_curves.pdf'
plt.savefig(save_directory)
plt.show()

with torch.no_grad():
    z = encoder_vq(images_original)
    z = pre_vq_conv1(z)
    quantized, _,_ = vq_vae(z)
    x_recon = decoder_vq(quantized)
    
    # Display original and reconstructed images
    plotting_functions.display_images(images_original, x_recon, num_images=8, step = i)


# Save the Encoder, Decoder, and Optimizer
save_directory = '/share/nas2_3/amahmoud/week5/galaxy_out/'
torch.save({
    'encoder_vq_state_dict': encoder_vq.state_dict(),
    'decoder_vq_state_dict': decoder_vq.state_dict(),
    'vq_vae_state_dict': vq_vae.state_dict(),
}, os.path.join(save_directory, 'vqvae_model.pth'))

wandb.finish()

print("Model and training metrics saved successfully.")

