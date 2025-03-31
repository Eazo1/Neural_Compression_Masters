import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from mtl_helpers import load_unnoised_data, add_correlated_noise
from encoder import Encoder
from decoder import Decoder

cholesky_factor_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/point_spread_cholesky_factor2.pt"
cholesky_factor = torch.load(cholesky_factor_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
cholesky_factor = cholesky_factor.float()

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, input_dim=num_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_hiddens = 256
num_residual_layers = 2
num_residual_hiddens = 32
sigma = 0.15  # assumed noise std
target_var = 2 * np.square(sigma)  # Theoretical variance of the difference delta

# Load trained AE model

#model_save_path = '/share/nas2_3/amahmoud/week5/sem2work/mtl_autoencoder_model.pt'
model_save_path = '/share/nas2_3/amahmoud/week5/sem2work/wasserstein_autoencoder_model.pt'
#model_save_path = '/share/nas2_3/amahmoud/week5/galaxy_out/autoencoder_model.pth'

autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
autoencoder.load_state_dict(torch.load(model_save_path, map_location=device))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False

# Load data
valid_dataset = load_unnoised_data('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', device=None)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

print('Data Loaded Successfully')

# Collect all residuals
all_deltas = []

for batch in valid_loader:
    if batch.dim() == 5:
        batch = batch.squeeze(2)
    elif batch.dim() == 3:
        batch = batch.unsqueeze(1)
    
    batch = batch.to(device)
    batch = add_correlated_noise(batch, cholesky_factor)

    with torch.no_grad():
        recon = autoencoder(batch)

    delta = batch - recon
    all_deltas.append(delta.cpu().view(-1))  # Flatten and store

# Stack all deltas/differences and compute variance
all_deltas = torch.cat(all_deltas)
empirical_var = torch.var(all_deltas).item()
score = empirical_var / target_var

# Report
print("\n Normalized Noise Difference Metric")
print(f"Empirical variance of the difference: {empirical_var:.6f}.")
print(f"Target variance (2 sigma^2): {target_var:.6f}.")
print(f"Normalized score: {score:.4f}.")
