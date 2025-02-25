import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import io
from scipy.stats import norm

#
NOISE_MEAN = 0.0 # Mean of the Gaussian noise
NOISE_STD = 4.0 # Standard deviation of the Gaussian noise
GAMMA = 10000  # Hyperparameter for the KL divergence term
#GAMMA = 0
#

# Function to compute KL divergence between two univariate Gaussians
def kl_divergence_two_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    return np.log(sigma_2 / sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2) - 0.5


def plot_gaussian_histogram(pixel_values, num_bins, non_recon_pixel_values):
    
    mean_pixels = np.mean(pixel_values)
    std_pixels = np.std(pixel_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    ax.hist(pixel_values, bins=num_bins, density=True, alpha=0.5, color='b', label="Reconstructed Pixels")
    ax.hist(non_recon_pixel_values, bins=num_bins, density=True, alpha=0.5, color='r', label="OG Pixels")

    # Fitted Gaussian Curve
    x = np.linspace(min(non_recon_pixel_values), max(non_recon_pixel_values), 1000)
    ax.plot(x, norm.pdf(x, mean_pixels, std_pixels), 'r-', label=f"Fitted Gaussian (μ={mean_pixels:.2f}, σ={std_pixels:.2f})")
    ax.plot(x, norm.pdf(x, NOISE_MEAN, NOISE_STD), 'r-', label=f"OG NOISE Gaussian (μ={NOISE_MEAN:.2f}, σ={NOISE_STD:.2f})")

    # Labels & Title
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Pixel Intensities, gamma={GAMMA}, D_KL={kl_divergence_two_gaussians(mean_pixels, std_pixels, NOISE_MEAN, NOISE_STD):.2f}")
    ax.legend()

    return fig

def display_images(images: torch.Tensor, 
                   recon_images: torch.Tensor, 
                   num_images: int = 8, 
                   step: int = 0):
    # Limit how many images we actually plot
    num_images = min(images.size(0), num_images)

    # Create a Matplotlib figure with two rows (for original + recon) 
    # and 'num_images' columns
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(2 * num_images, 5))
    
    # If there is only one image, 'axes' won't be a 2D array, so we can fix that
    if num_images == 1:
        axes = np.array(axes).reshape(2, 1)

    # Move Tensors to CPU and convert to NumPy
    images_np = images.detach().cpu().numpy()
    recon_np = recon_images.detach().cpu().numpy()

    for i in range(num_images):
        # Original image
        original_img = images_np[i]
        # Reconstructed image
        recon_img = recon_np[i]
        
        if original_img.shape[0] == 1:
            # Grayscale
            original_img = original_img.squeeze(0)  # -> shape [H, W]
            recon_img = recon_img.squeeze(0)
            cmap = "gray"
        else:
            # Color (e.g., 3 channels)
            # Convert from [C, H, W] -> [H, W, C]
            original_img = original_img.transpose(1, 2, 0)
            recon_img = recon_img.transpose(1, 2, 0)
            cmap = None  # no colormap for RGB

        # Top row: original
        axes[0, i].imshow(original_img, cmap=cmap)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12)

        # Bottom row: reconstructed
        axes[1, i].imshow(recon_img, cmap=cmap)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=12)

    plt.suptitle(f"Step: {step}", fontsize=16)
    plt.tight_layout()
    return fig

# Function to add Gaussian noise
def add_gaussian_noise(images, mean=NOISE_MEAN, std=NOISE_STD):
    noise = torch.normal(mean=mean, std=std, size=images.size()).to(images.device)
    integer_noise = torch.floor(noise)
    noisy_images = images + integer_noise
    return noisy_images

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a tensor in the shape stored in the npy file.
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Modified Dataset class to add noise once for each image
class MemoryMappedDatasetWithNoise(MemoryMappedDataset):
    def __init__(self, mmap_data, device, mean=0.0, std=4.0):
        super().__init__(mmap_data, device)
        self.mean = mean
        self.std = std
        self.noisy_data = self.add_noise_to_data(mmap_data)

    def add_noise_to_data(self, data):
        # Add Gaussian noise to the entire dataset once
        noisy_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        noisy_data = add_gaussian_noise(noisy_data, mean=self.mean, std=self.std)
        #noisy_data = torch.abs(noisy_data)
        return noisy_data

    def __getitem__(self, idx):
        # Return the noisy data instead of original data
        return self.noisy_data[idx]

# Data loading paths
train_data_path = '/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy'
valid_data_path = '/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy'

train_data_mmap = np.load(train_data_path, mmap_mode='r')
valid_data_mmap = np.load(valid_data_path, mmap_mode='r')

# Create datasets and loaders
train_dataset = MemoryMappedDatasetWithNoise(train_data_mmap, device=None)
valid_dataset = MemoryMappedDatasetWithNoise(valid_data_mmap, device=None)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# --- Model Components ---
from encoder import Encoder
from decoder import Decoder
import plotting_functions  # your custom plotting module

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, input_dim=num_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Setup parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_hiddens = 256
num_residual_layers = 2
num_residual_hiddens = 32
learning_rate = 2e-4
num_training_updates = 2000

wandb.login(key="7391c065d23aad000052bc1f7a3a512445ae83d0")
wandb.init(
    project="Standard AE",
    config={
        "Gamma (KL hyperparameter)": GAMMA,
        "embedding_dim": 64,
        "num_embeddings": 256,
        "architecture": "AE",
        "dataset": "CIFAR-10",
        "num_training_updates": 250000,
        "learning_rate": learning_rate,
    },
    reinit=True,
)

# Instantiate AE and optimizer
autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
train_losses = []
iteration = 0
autoencoder.train()
print("Starting training...")
while iteration < num_training_updates:
    for images in train_loader:
        # Debug print: show shape before processing.
        print("Shape of images before processing:", images.shape)
        
        # If the tensor has 5 dimensions (e.g., [batch, 1, 1, H, W]), remove the extra dimension.
        if images.dim() == 5:
            images = images.squeeze(2)  # Remove the extra dimension at index 2.
        # If images come in as 3D (i.e., missing the channel dimension), add one.
        elif images.dim() == 3:
            images = images.unsqueeze(1)
        
        images = images.to(device)

        optimizer.zero_grad()
        recon = autoencoder(images)
        
        recon_pixels = recon.detach().cpu().numpy().ravel()
        
        mean_recon = np.mean(recon_pixels)
        std_recon = np.std(recon_pixels)
        
        kl_loss = kl_divergence_two_gaussians(mean_recon, std_recon, NOISE_MEAN, NOISE_STD)
        
        mse_loss = F.mse_loss(recon, images, reduction='sum')
        
        gamma = GAMMA
        loss = mse_loss + gamma * kl_loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        wandb.log({"train/loss": loss.item()})
        
        wandb.log({
                    "train/loss_components.mse": mse_loss.item(),
                    "train/loss_components.kl": (kl_loss).item(),
                    "train/loss_components.g*kl": (gamma * kl_loss).item()
                    })
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, training loss: {loss.item():.4f}")
        
        if iteration % 1000 == 0:
            print(f"Generating gaussian hist at iteration {iteration}...")

            # Use only the pixels from the latest batch (instead of accumulating all)
            current_batch_pixels = recon_pixels  # This is already flattened

            # Plot the histogram for this batch only
            fig_hist = plot_gaussian_histogram(current_batch_pixels, num_bins=50, non_recon_pixel_values=images.detach().cpu().numpy().ravel())

            #wandb.log({"D_KL/loss": kl_loss})
            
            # Log to Weights & Biases
            wandb.log({f"histogram/iteration_{iteration}": wandb.Image(fig_hist)})

            # Close the figure to free memory
            plt.close(fig_hist)
        
        if iteration >= num_training_updates:
            break
          
    # Validation loop
    autoencoder.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        num_batches = 0
        for val_images in valid_loader:
            if val_images.dim() == 5:
                val_images = val_images.squeeze(2)
            elif val_images.dim() == 3:
                val_images = val_images.unsqueeze(1)
            val_images = val_images.to(device)
            
            recon_val = autoencoder(val_images)
            
            recon_pixels = recon_val.detach().cpu().numpy().ravel()
            
            mean_recon = np.mean(recon_pixels)
            std_recon = np.std(recon_pixels)
        
            kl_loss = kl_divergence_two_gaussians(mean_recon, std_recon, NOISE_MEAN, NOISE_STD)
        
            mse_loss = F.mse_loss(recon, images, reduction='sum')
        
            gamma = GAMMA
            loss_val = mse_loss + gamma * kl_loss
            total_val_loss += loss_val.item()
            num_batches += 1
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        wandb.log({"validation/loss": avg_val_loss})
        print(f"Validation loss: {avg_val_loss:.4f}")
    autoencoder.train()

# ---- Evaluation and Plotting ----
autoencoder.eval()
with torch.no_grad():
    for images in valid_loader:
        # Fix shape if needed
        if images.dim() == 5:
            images = images.squeeze(2)
        elif images.dim() == 3:
            images = images.unsqueeze(1)
        images = images.to(device)
        
        recon_images = autoencoder(images)  # Inference on the validation batch
        
        break

fig_recon = display_images(images, recon_images, num_images=8, step=iteration)

wandb.log({"reconstruction_images": wandb.Image(fig_recon)})

plt.close(fig_recon)

def plot_four_histograms_fit_residual(
    real_pixels: np.ndarray,
    recon_pixels: np.ndarray,
    std_forced: float = NOISE_STD,
    mean_forced: float = NOISE_MEAN,
    num_bins: int = 50
):

    # ----------------------------
    #  1) Linear-scale comparison
    # ----------------------------
    min_clip = 0  # Adjust as needed
    max_clip = 255 # Adjust as needed
    #real_pixels = np.clip(real_pixels, min_clip, max_clip)
    #recon_pixels = np.clip(recon_pixels, min_clip, max_clip)
    
    all_pixels = np.concatenate([real_pixels, recon_pixels])
    min_val, max_val = np.min(all_pixels), np.max(all_pixels)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    ax1.hist(real_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.5, label='Real (Noisy)')
    ax1.hist(recon_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.5, label='Reconstructed')
    ax1.set_title("1) Linear Scale: Real vs. Recon")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right")
    

    # ----------------------------
    #  2) Log-scale comparison
    # ----------------------------
    ax2.hist(real_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.5, label='Real (Noisy)')
    ax2.hist(recon_pixels, bins=num_bins, range=(min_val, max_val),
             alpha=0.5, label='Reconstructed')
    ax2.set_yscale('log')
    ax2.set_title("2) Log Scale: Real vs. Recon")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Log(Count)")
    ax2.legend(loc="upper right")

    # ----------------------------------------------
    #  3) Linear-scale with forced/best-fit Gaussians
    # ----------------------------------------------
    hist_data_real, bin_edges_real = np.histogram(real_pixels, bins=num_bins)
    bin_centers_real = 0.5 * (bin_edges_real[:-1] + bin_edges_real[1:])
    bin_width_real = bin_edges_real[1] - bin_edges_real[0]
    N_real = len(real_pixels)

    # Forced Real Gaussian
    gauss_forced_real = (norm.pdf(bin_centers_real, loc=mean_forced, scale=std_forced) * N_real * bin_width_real)

    hist_data_recon, bin_edges_recon = np.histogram(recon_pixels, bins=num_bins)
    bin_centers_recon = 0.5 * (bin_edges_recon[:-1] + bin_edges_recon[1:])
    bin_width_recon = bin_edges_recon[1] - bin_edges_recon[0]
    N_recon = len(recon_pixels)

    # Best-Fit Recon Gaussian (best fit real)
    mean_recon = np.mean(recon_pixels)
    std_recon = np.std(recon_pixels)
    gauss_bestfit_recon = (norm.pdf(bin_centers_recon, loc=mean_recon, scale=std_recon) * N_recon * bin_width_recon)

    ax3.hist(real_pixels, bins=num_bins, alpha=0.5, label='Real Data')
    ax3.plot(bin_centers_real, gauss_forced_real, 'r-', lw=2,
             label=f"Forced Gauss(μ={mean_forced}, σ={std_forced:.2f})")

    ax3.hist(recon_pixels, bins=num_bins, alpha=0.5, label='Reconstructed Data')
    ax3.plot(bin_centers_recon, gauss_bestfit_recon, 'g-', lw=2,
             label=f"Best-Fit Gauss(μ={mean_recon:.2f}, σ={std_recon:.2f})")

    ax3.set_title("3) Linear Scale + Forced/Best-Fit Gaussians")
    ax3.set_xlabel("Pixel Intensity")
    ax3.set_ylabel("Count")
    ax3.legend(loc="upper right")

    # -----------------------------------------------------------
    #  4) Residual = (Fit - Data) for Real and Recon, side by side
    # -----------------------------------------------------------
    # Real
    residual_real = -gauss_forced_real + hist_data_real
    # Recon
    residual_recon = -gauss_bestfit_recon + hist_data_recon

    # We can plot both on the same subplot with alpha overlap:
    ax4.bar(bin_centers_real, residual_real,
            width=bin_width_real, alpha=0.5, label='(Real Data - Forced Fit)', color='C0')
    ax4.bar(bin_centers_recon, residual_recon,
            width=bin_width_recon, alpha=0.5, label='(Recon Data - Fit)', color='C1')

    ax4.axhline(0, color='k', linestyle='--')
    ax4.set_title("4) Residual: (Fit - Data)")
    ax4.set_xlabel("Pixel Intensity")
    ax4.set_ylabel("Diff in Count")
    ax4.set_yscale('log')
    ax4.set_xlim(left=20)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    return fig
  
#real_pixels = images.detach().cpu().numpy().ravel()
#recon_pixels = recon_images.detach().cpu().numpy().ravel()
all_real_pixels = []
all_recon_pixels = []

autoencoder.eval()
with torch.no_grad():
    for val_images in valid_loader:
        # Fix shapes as before
        if val_images.dim() == 5:
            val_images = val_images.squeeze(2)
        elif val_images.dim() == 3:
            val_images = val_images.unsqueeze(1)

        val_images = val_images.to(device)

        # Forward pass through the model
        val_recon = autoencoder(val_images)

        # Flatten and move to CPU NumPy
        real_np = val_images.detach().cpu().numpy().ravel()
        recon_np = val_recon.detach().cpu().numpy().ravel()

        # Accumulate in Python lists
        all_real_pixels.append(real_np)
        all_recon_pixels.append(recon_np)

# Concatenate everything into one array each
all_real_pixels = np.concatenate(all_real_pixels)
all_recon_pixels = np.concatenate(all_recon_pixels)

# -----------------------------------------
# NOW PLOT THE FOUR HISTOGRAMS
# -----------------------------------------
fig_four = plot_four_histograms_fit_residual(
    real_pixels=all_real_pixels,     # entire dataset, real/noisy
    recon_pixels=all_recon_pixels,   # entire dataset, reconstructed
    std_forced=NOISE_STD,                  # your known noise std
    mean_forced=NOISE_MEAN,                 # forced mean
    num_bins=50
)

# Log to Weights & Biases
wandb.log({"four_histograms": wandb.Image(fig_four)})

# Close the figure to free resources
plt.show()  # or plt.show(block=False) if in a notebook
plt.close(fig_four)
