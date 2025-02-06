import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a tensor in the shape stored in the npy file.
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Data loading paths
train_data_path = '/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy'
valid_data_path = '/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy'

train_data_mmap = np.load(train_data_path, mmap_mode='r')
valid_data_mmap = np.load(valid_data_path, mmap_mode='r')

# Create datasets and loaders
train_dataset = MemoryMappedDataset(train_data_mmap, device=None)
valid_dataset = MemoryMappedDataset(valid_data_mmap, device=None)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# --- Model Components ---
from encoder import Encoder
from decoder import Decoder
import plotting_functions

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
num_training_updates = 1000

wandb.login(key="7391c065d23aad000052bc1f7a3a512445ae83d0")
wandb.init(
    project="Standard AE",
    config={
        "embedding_dim": 64,
        "num_embeddings": 256,
        "architecture": "VQ-VAE",
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
        loss = F.mse_loss(recon, images, reduction='sum')
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        wandb.log({"train/loss": loss.item()})
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, training loss: {loss.item():.4f}")
        if iteration >= num_training_updates:
            break
          
    # Validation loop
    autoencoder.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        num_batches = 0
        for val_images in valid_loader:
            # Apply the same dimension fix as above.
            if val_images.dim() == 5:
                val_images = val_images.squeeze(2)
            elif val_images.dim() == 3:
                val_images = val_images.unsqueeze(1)
            val_images = val_images.to(device)
            recon_val = autoencoder(val_images)
            loss_val = F.mse_loss(recon_val, val_images, reduction='sum')
            total_val_loss += loss_val.item()
            num_batches += 1
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        wandb.log({"validation/loss": avg_val_loss})
        print(f"Validation loss: {avg_val_loss:.4f}")
    autoencoder.train()

# Evaluation (for visualization)
autoencoder.eval()
with torch.no_grad():
    for images in valid_loader:
        # Again, fix the shape if needed.
        if images.dim() == 5:
            images = images.squeeze(2)
        elif images.dim() == 3:
            images = images.unsqueeze(1)
        images = images.to(device)
        recon_images = autoencoder(images)
        break

# Use your previously defined plotting functions
plotting_functions.display_images(images, recon_images, num_images=8, step=iteration)

# Save the model
save_directory = '/share/nas2_3/amahmoud/week5/galaxy_out/'
model_save_path = os.path.join(save_directory, 'autoencoder_model.pth')
torch.save(autoencoder.state_dict(), model_save_path)
print("Model saved to", model_save_path)

wandb.finish()
