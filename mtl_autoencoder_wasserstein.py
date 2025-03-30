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
        load_noised_data, load_unnoised_data, chi_squared_covariance, add_correlated_noise_halton

from mtl_helpers import MemoryMappedDataset, MemoryMappedDatasetWithCorrelatedNoise

#
#NOISE_MEAN = 0.0 # Mean of the Gaussian noise
#NOISE_STD = 4.0 # Standard deviation of the Gaussian noise
#GAMMA = 10000  # Hyperparameter for the KL divergence term (to be removed, replaced with Wasserstein-2 term)
#KAPPA = 0.01  
#DELTA = 1  # Hyperparameter for the chi-square term
#GAMMA = 0
DIGAMMA = 1 # Hyperparameter for the Wasserstein-2 term

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

correlation_matrix_path = "precomputed/point_spread_covariance_matrix2.pt"
correlation_matrix = torch.load(correlation_matrix_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
Sigma_orig = correlation_matrix.to(dtype=torch.float32)
Sigma_orig = Sigma_orig.float()

OG_mean_vector_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/images_mean_vector2.pt"
OG_mean_vector = torch.load(OG_mean_vector_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
OG_mean_vector = OG_mean_vector.float()

cholesky_factor_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/point_spread_cholesky_factor2.pt"
cholesky_factor = torch.load(cholesky_factor_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
cholesky_factor = cholesky_factor.float()

correlation_matrix_sqrt_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/covariance_matrix_sqrt2.pt"
correlation_matrix_sqrt = torch.load(correlation_matrix_sqrt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
correlation_matrix_sqrt = correlation_matrix_sqrt.float()

# Create datasets and loaders
train_dataset = load_unnoised_data('/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy', device=None)
valid_dataset = load_unnoised_data('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', device=None)

batch_size = 1 # Global batch size definition
accumulation_steps = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print('Data loaded successfully.')

# --- Model Components ---
from encoder import Encoder
from decoder import Decoder
#import plotting_functions

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

num_training_updates = 500

# for the monte carlo in the training loop
num_noisy_samples = 500

wandb.login(key="7391c065d23aad000052bc1f7a3a512445ae83d0")
wandb.init(
    project="Covar_AE",
    config={
        "DIGAMMA (Wasserstein hyperparameter)": DIGAMMA,
        #"embedding_dim": 64,
        #"num_embeddings": 256,
        "architecture": "AE",
        "dataset": "RGZ",
        #"num_training_updates": 250000,
        "learning_rate": learning_rate,
        "num_noisy_samples": num_noisy_samples,
        "num_training_updates": num_training_updates,
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
    optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
    
    for images in train_loader:
        # Debug print: show shape before processing.
        print("Shape of images before processing:", images.shape)
        
        # If the tensor has 5 dimensions ([batch, 1, 1, H, W]), remove the extra dimension.
        if images.dim() == 5:
            images = images.squeeze(2)  # Remove the extra dimension at index 2.
        # If images come in as 3D (i.e., missing the channel dimension), add one.
        elif images.dim() == 3:
            images = images.unsqueeze(1)
        
        images = images.to(device)
        noised_images = add_correlated_noise(images, cholesky_factor)
                
        recon = autoencoder(noised_images)
        
        mse_loss = F.mse_loss(recon, noised_images, reduction='sum')
        
        w_distance = torch.tensor(0.0, device=device)  
        
        sub_batch_size = num_noisy_samples
        num_sub_batches = num_noisy_samples // sub_batch_size
        
        # monte carlo estimate of the KL divergence
        for i in range(images.size(0)):
            w_distance_sub = torch.tensor(0.0, device=device)
            for _ in range(num_sub_batches):  # Mini-batches to save memory
                noisy_images = torch.stack([add_correlated_noise_halton(images[i], cholesky_factor) for _ in range(sub_batch_size)])
                if noisy_images.dim() == 5:  
                    noisy_images = noisy_images.squeeze(2)

                recon_images = autoencoder(noisy_images) 
                Sigma_recon, mean_image = compute_covariance_torch(recon_images)

                w_distance_sub += wasserstein2_two_multivariate_gaussians(mean_image, Sigma_recon, OG_mean_vector, Sigma_orig, correlation_matrix_sqrt)
                
                # Free up memory for the next batch
                del noisy_images, recon_images, Sigma_recon
                torch.cuda.empty_cache()
            
            w_distance += w_distance_sub
        
        loss = mse_loss + DIGAMMA * w_distance  # No .item() here!
        
        # Accumulate gradients instead of updating every batch
        loss.backward()
        
        # Perform optimizer step and zero the gradients after `accumulation_steps`
        if (iteration + 1) % accumulation_steps == 0:
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients

        train_losses.append(loss.item())
        #wandb.log({"train/loss": loss.item()})
        
        wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_components.mse": mse_loss.item(),
                    "train/loss_components.w_distance": (w_distance).item(),
                    "train/loss_components.F*w_distance": (DIGAMMA * w_distance.item())
                    })
        
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, training loss: {loss.item():.4f}")
                    
            # Validation loop
            autoencoder.eval()
            total_val_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for val_images in valid_loader:
                    if val_images.dim() == 5:
                        val_images = val_images.squeeze(2)
                    elif val_images.dim() == 3:
                        val_images = val_images.unsqueeze(1)
                    val_images = val_images.to(device)
                    
                    noised_val_images = add_correlated_noise(val_images, cholesky_factor)
                
                    recon_val = autoencoder(noised_val_images)
                    
                    mse_loss_val = F.mse_loss(recon_val, val_images, reduction='sum')
                
                    w_distance_val = torch.tensor(0.0, device=device)
                    
                    # monte carlo estimate of the KL divergence
                    for i in range(images.size(0)):
                        w_distance_sub = torch.tensor(0.0, device=device)
                        for _ in range(num_sub_batches):  # Mini-batches to save memory
                            with torch.no_grad(): # Save memory
                                noisy_images = torch.stack([add_correlated_noise_halton(noised_val_images[i], cholesky_factor) for _ in range(sub_batch_size)])
                                if noisy_images.dim() == 5:  
                                    noisy_images = noisy_images.squeeze(2)
                                recon_images = autoencoder(noisy_images) 
                                Sigma_recon, mean_image = compute_covariance_torch(recon_images)

                            w_distance_sub += wasserstein2_two_multivariate_gaussians(mean_image, Sigma_recon, OG_mean_vector, Sigma_orig, correlation_matrix_sqrt)
                            

                            # Free up memory for the next batch
                            del noisy_images, recon_images, Sigma_recon
                            torch.cuda.empty_cache()
                        
                        w_distance_val += w_distance_sub
                        
                    loss_val = mse_loss_val + DIGAMMA * w_distance_val
                    total_val_loss += loss_val.item()
                    num_batches += 1

            avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float("inf")
            wandb.log({"validation/loss": avg_val_loss,
                       "validation/loss_components.mse": mse_loss.item(),
                        "validation/loss_components.w_distance": (w_distance).item(),})
            print(f"Validation loss: {avg_val_loss:.4f}")
            
            autoencoder.train()
            
        if iteration >= num_training_updates:
            break



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
        
        recon_images = autoencoder(images)
        
        break

fig_recon = display_images(images, recon_images, num_images=8, step=iteration)

wandb.log({"reconstruction_images": wandb.Image(fig_recon)})

plt.close(fig_recon)

# Save the model
torch.save(autoencoder.state_dict(), "wasserstein_autoencoder_model.pt")