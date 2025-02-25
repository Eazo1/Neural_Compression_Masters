import os
os.environ['WANDB_MODE'] = 'dryrun'
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from artificial_noisy_main_autoencoder import train_loader
from artificial_noisy_main_autoencoder import Autoencoder
from artificial_noisy_main_autoencoder import kl_divergence_two_gaussians

# Hyperparameters and other variables
learning_rate = 2e-4
num_epochs = 10  # Change to your actual number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_training_updates = 2000
GAMMA = 10000  # This will be overwritten by sweep config

# Define the sweep function that will be run for each hyperparameter set
def sweep_train():
    # Initialize WandB and get the hyperparameters
    wandb.init()
    gamma = wandb.config.Gamma  # This will be updated during the sweep

    # Setup model and optimizer
    autoencoder = Autoencoder(num_hiddens=256, num_residual_layers=2, num_residual_hiddens=32).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    iteration = 0
    autoencoder.train()
    print(f"Starting training with Gamma={gamma}...")

    while iteration < num_training_updates:
        for images in train_loader:  # Assuming you have defined your DataLoader `train_loader`
            images = images.to(device)
            
            optimizer.zero_grad()
            recon = autoencoder(images)
            
            # Compute the KL divergence loss and MSE loss
            recon_pixels = recon.detach().cpu().numpy().ravel()
            mean_recon = np.mean(recon_pixels)
            std_recon = np.std(recon_pixels)
            kl_loss = kl_divergence_two_gaussians(mean_recon, std_recon, NOISE_MEAN, NOISE_STD)
            mse_loss = F.mse_loss(recon, images, reduction='sum')
            
            # Compute the total loss using the current value of Gamma
            loss = mse_loss + gamma * kl_loss
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            wandb.log({"train/loss": loss.item()})
            wandb.log({
                "train/loss_components.mse": mse_loss.item(),
                "train/loss_components.kl": (kl_loss).item(),
                "train/loss_components.g*kl": (kl_loss).item()
            })
            
            iteration += 1

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, training loss: {loss.item():.4f}")
            
            if iteration >= num_training_updates:
                break
    
    wandb.log({"final_train_loss": train_losses[-1]})

# Sweep configuration
sweep_configuration = {
    'method': 'grid',  # Choose method: 'grid' for grid search, 'random' for random search
    'name': 'gamma_sweep',
    'metric': {
        'name': 'train/loss',  # The metric to optimize (we minimize the loss)
        'goal': 'minimize'
    },
    'parameters': {
        'Gamma': {
            'values': [10, 100, 1000, 10000, 20000, 50000, 20000, 50000, 100000, 500000, 1000000, 10000000]  # Different values of Gamma to explore
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_configuration, project="Standard AE")

# Start the sweep
wandb.agent(sweep_id, function=sweep_train, count=1)  # Set count to the number of trials you want
