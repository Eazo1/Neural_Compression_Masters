import os
import numpy as np
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io
from scipy.stats import norm
from mtl_helpers import kl_divergence_two_gaussians, matrix_sqrt, wasserstein2_two_multivariate_gaussians, \
    generate_correlated_noise, add_correlated_noise, generate_noisy_copies, compute_covariance_torch, \
        load_noised_data, load_unnoised_data, chi_squared_covariance

from mtl_helpers import MemoryMappedDataset, MemoryMappedDatasetWithCorrelatedNoise

###

# Import the trained mtl model from mtl_autoencoder_model.pt
from encoder import Encoder
from decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, input_dim=num_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters (must match those used during saving)
num_hiddens = 256
num_residual_layers = 2
num_residual_hiddens = 32
learning_rate = 2e-4

autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
model_save_path = '/share/nas2_3/amahmoud/week5/sem2work/mtl_autoencoder_model.pt'
autoencoder.load_state_dict(torch.load(model_save_path, map_location=device))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False  # Disable gradients to prevent training
    
autoencoder_OG = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
model_save_path2 = '/share/nas2_3/amahmoud/week5/galaxy_out/autoencoder_model.pth'
autoencoder_OG.load_state_dict(torch.load(model_save_path2, map_location=device))
autoencoder_OG.eval()
for param in autoencoder_OG.parameters():
    param.requires_grad = False  # Disable gradients to prevent training

#

# Load example data from the dataset
valid_dataset = load_unnoised_data('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', device=None)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

# Encode the example images
# AE_images = torch.cat(example_images)
# AE_images = AE_images.to(device)
# AE_images = autoencoder.encoder(AE_images)

# # Decode the encoded images
# AE_images = autoencoder.decoder(AE_images)

# from artificial_noisy_main_autoencoder import plot_four_histograms_fit_residual

# Plot the original images, the encoded images, the decoded images, and the residuals
all_real_pixels = []
all_recon_pixels = []
all_recon_pixels_OG = []


autoencoder.eval()
autoencoder_OG.eval()
with torch.no_grad():
    for val_images in valid_loader:
        # Fix shapes as before
        if val_images.dim() == 5:
            val_images = val_images.squeeze(2)
        elif val_images.dim() == 3:
            val_images = val_images.unsqueeze(1)

        val_images = val_images.to(device)

        # Forward pass through the model
        val_recon = autoencoder.encoder(val_images)
        val_recon = autoencoder.decoder(val_recon)
        
        val_recon_OG = autoencoder_OG.encoder(val_images)
        val_recon_OG = autoencoder_OG.decoder(val_recon_OG)

        # Flatten and move to CPU NumPy
        real_np = val_images.detach().cpu().numpy().ravel()
        recon_np = val_recon.detach().cpu().numpy().ravel()
        recon_np_OG = val_recon_OG.detach().cpu().numpy().ravel()

        # Accumulate in Python lists
        all_real_pixels.append(real_np)
        all_recon_pixels.append(recon_np)
        all_recon_pixels_OG.append(recon_np_OG)

# Concatenate everything into one array each
all_real_pixels = np.concatenate(all_real_pixels)
all_recon_pixels = np.concatenate(all_recon_pixels)
all_recon_pixels_OG = np.concatenate(all_recon_pixels_OG)


def plot_histograms(
    real_pixels: np.ndarray,
    recon_pixels: np.ndarray,
    recon_pixels_OG: np.ndarray,
    #std_forced: float = NOISE_STD,
    #mean_forced: float = NOISE_MEAN,
    num_bins: int = 50
):

    # ----------------------------
    #  1) Linear-scale comparison
    # ----------------------------
    #min_clip = 0  # Adjust as needed
    #max_clip = 255 # Adjust as needed
    #real_pixels = np.clip(real_pixels, min_clip, max_clip)
    #recon_pixels = np.clip(recon_pixels, min_clip, max_clip)
    
    all_pixels = np.concatenate([real_pixels, recon_pixels, recon_pixels_OG])
    min_val, max_val = np.min(all_pixels), np.max(all_pixels)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))

    ax1.hist(real_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Real (Noisy)')
    ax1.hist(recon_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Reconstructed Chi2')
    ax1.hist(recon_pixels_OG, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Reconstructed OG')
    ax1.set_title("1) Linear Scale: Real vs. Recons")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right")
    

    # ----------------------------
    #  2) Log-scale comparison
    # ----------------------------
    ax2.hist(real_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Real (Noisy)')
    ax2.hist(recon_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Reconstructed Chi2')
    ax2.hist(recon_pixels_OG, bins=num_bins, range=(min_val, max_val),
             alpha=0.4, label='Reconstructed OG')
    ax2.set_yscale('log')
    ax2.set_title("2) Log Scale: Real vs. Recons")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Log(Count)")
    ax2.legend(loc="upper right")

    return fig

his_fig = plot_histograms(all_real_pixels, all_recon_pixels, all_recon_pixels_OG)
plt.show()  # or plt.show(block=False) if in a notebook
plt.savefig('/share/nas2_3/amahmoud/week5/sem2work/visfig.pdf')
plt.close(his_fig)

###

print('Finished')