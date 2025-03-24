import os
import glob
import time
import numpy as np
import torch
import torch.linalg
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io
from scipy.stats import norm

def load_correlation_matrix(path="/share/nas2_3/amahmoud/week5/sem2work/precomputed/point_spread_covariance_matrix2.pt", device="cpu"):
    return torch.load(path).to(device)

def generate_correlated_noise(correlation_matrix, device="cpu"):
    N = correlation_matrix.shape[0]  # This will be H * W
    L = torch.linalg.cholesky(correlation_matrix)  # Cholesky decomposition
    iid_noise = torch.randn(N, device=device)  # Generate IID Gaussian noise
    correlated_noise = L @ iid_noise  # Apply Cholesky factor to introduce correlation
    return correlated_noise

def generate_correlated_noise(correlation_matrix, num_samples=1000, device="cpu"):
    N = correlation_matrix.shape[0]  # The number of dimensions (e.g., number of pixels)

    # Ensure the correlation matrix is on the correct device
    correlation_matrix = correlation_matrix.cpu().numpy()  # Convert to numpy for easier handling

    # Cholesky decomposition to get the lower triangular matrix
    L = np.linalg.cholesky(correlation_matrix)

    # Generate uncorrelated standard normal random variables
    noise = np.random.randn(num_samples, N)  # Shape: (num_samples, N)

    # Multiply by the Cholesky factor to introduce correlation
    correlated_noise = noise @ L.T  # Shape: (num_samples, N)

    return correlated_noise



def precompute_cholesky(correlation_matrix):
    return torch.linalg.cholesky(correlation_matrix)

def add_correlated_noise(images, cholesky_factor):
    B, C, H, W = images.shape  # Get batch size, channels, and image dimensions
    N = H * W  # Total number of pixels (H * W)

    # Generate standard Gaussian noise with shape (B, 1, N)
    noise = torch.randn((B, 1, N), device=images.device, dtype=torch.float64) ### LATER: MAKE SURE OG IS 64 NOT 32

    # Apply Cholesky factor (This step adds correlation)
    correlated_noise = torch.matmul(cholesky_factor, noise.squeeze(1).T).T.unsqueeze(1)

    # Reshape back to (B, 1, H, W) to match image dimensions
    correlated_noise = correlated_noise.view(B, 1, H, W)

    # Add the correlated noise to the images
    return images + correlated_noise


# -------------------------------
# Dataset Classes
# -------------------------------

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a tensor in the shape stored in the npy file.
        return torch.tensor(self.data[idx], dtype=torch.float64)

# Modified Dataset class to add noise once for each image
class MemoryMappedDatasetWithNoise(MemoryMappedDataset):
    def __init__(self, mmap_data, device, cholesky_factor):
        super().__init__(mmap_data, device)
        self.cholesky_factor = cholesky_factor  # Store precomputed Cholesky factor

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float64, device=self.device)
        
        if image.numel() == 0:
            print(f"[ERROR] Skipping empty image at index {idx}")
            return torch.zeros((150, 150), dtype=torch.float64, device=self.device)
        
        # Ensure correct shape: Remove last dim if it's (H, W, 1)
        if image.dim() == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
        
        # Reshape into (1, 1, H, W) for batch processing by the noise function
        noisy_image = add_correlated_noise(image.unsqueeze(0).unsqueeze(0), self.cholesky_factor)
        # Remove batch and channel dimensions before returning
        noisy_image = noisy_image.squeeze(0).squeeze(0)
        return noisy_image

# -------------------------------
# Helper Functions for Covariance (Using PyTorch)
# -------------------------------

def compute_covariance_torch(noisy_images):
    """
    Compute the sample covariance matrix using PyTorch operations.
    
    Parameters:
        noisy_images: torch.Tensor of shape (N, H, W)
    
    Returns:
        cov_matrix: torch.Tensor of shape (num_features, num_features)
        mean_image: torch.Tensor of shape (num_features,)
    """
    N = noisy_images.shape[0]
    # Flatten each image to a vector (H*W elements)
    flattened = noisy_images.view(N, -1)  # shape: (N, num_features)
    # Compute the mean image (as a vector)
    mean_image = torch.mean(flattened, dim=0)
    # Center the data by subtracting the mean
    deviations = flattened - mean_image
    # Compute the sample covariance matrix
    cov_matrix = torch.mm(deviations.t(), deviations) / (N - 1)
    return cov_matrix, mean_image

def generate_noisy_copies(original_image, cholesky_factor, num_copies=50, device=None):
    copies = []
    
    # Convert image to float tensor on the specified device.
    image = original_image.to(dtype=torch.float64, device=device)
    
    # Normalize the image shape.
    # If image is 4D and has shape (1, 1, H, W), squeeze to get (H, W)
    if image.dim() == 4 and image.shape[0] == 1 and image.shape[1] == 1:
        image = image.squeeze(0).squeeze(0)
    # If image is 3D, it might be (1, H, W) or (H, W, 1)
    elif image.dim() == 3:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        elif image.shape[-1] == 1:
            image = image.squeeze(-1)
        else:
            raise ValueError(f"Expected a grayscale image with a singleton channel, but got shape {image.shape}")
    # If not 2D by now, raise an error.
    elif image.dim() != 2:
        raise ValueError(f"Expected a grayscale image with shape (H, W), but got shape {image.shape}")
    
    # Now image should be (H, W).
    for _ in range(num_copies):
        # Add batch and channel dimensions to get shape (1, 1, H, W)
        input_tensor = image.unsqueeze(0).unsqueeze(0)
        noisy_tensor = add_correlated_noise(input_tensor, cholesky_factor)
        # Remove the batch and channel dimensions to return to (H, W)
        noisy_tensor = noisy_tensor.squeeze(0).squeeze(0)
        copies.append(noisy_tensor)
    
    return torch.stack(copies, dim=0)

def check_invertibility(matrix, tolerance=1e-12):
    """
    Checks if a square matrix is invertible based on its determinant and eigenvalues.

    Args:
        matrix (torch.Tensor): The square matrix to check.
        tolerance (float): Tolerance for considering a value as zero.

    Returns:
        bool: True if the matrix is invertible, False otherwise.
        str: A message indicating the reason for invertibility or non-invertibility.
    """

    if not isinstance(matrix, torch.Tensor):
        return False, "Input must be a torch.Tensor."

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False, "Input must be a square matrix."

    try:
        # Check determinant
        #det = torch.linalg.det(matrix)
        #if abs(det) < tolerance:
        #    return False, f"Matrix is not invertible: determinant ({det}) is close to zero."

        # Check eigenvalues (more robust)
        eigenvalues = torch.linalg.eigvalsh(matrix) #eigvalsh is used because covariance matrices are symmetric.
        if torch.any(torch.abs(eigenvalues) < tolerance):
            return False, "Matrix is not invertible: eigenvalue(s) close to zero."

        return True, "Matrix is invertible."

    except RuntimeError as e:
        return False, f"Error during invertibility check: {e}"

# -------------------------------
# Main Code
# -------------------------------

def main():
    # -------------------------------
    # Load Memory-mapped Train Data
    # -------------------------------
    train_data_path = '/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy'
    train_data_mmap = np.load(train_data_path, mmap_mode='r')
    print(f"Loaded {len(train_data_mmap)} training images.")
    
    # -------------------------------
    # Load or Precompute the Cholesky Factor
    # -------------------------------
    # Option 1: Load precomputed factor (if available)
    # cholesky_factor = torch.load("cholesky_factor.pt")  # Adjust path as needed
    
    # Option 2: Compute the Cholesky factor from a correlation matrix:
    correlation_matrix = load_correlation_matrix(device=None)
    cholesky_factor = precompute_cholesky(correlation_matrix)
    
    # -------------------------------
    # Randomly select one image from the train data
    # -------------------------------
    num_train = len(train_data_mmap)
    random_index = torch.randint(0, num_train, (1,)).item()
    print(f"Randomly selected image index: {random_index}")
    original_image_np = train_data_mmap[random_index]
    original_image = torch.tensor(original_image_np, dtype=torch.float64)
    
    # -------------------------------
    # Generate noisy copies of the selected image using PyTorch
    # -------------------------------
    NUM_NOISY_COPIES = 50
    noisy_images = generate_noisy_copies(original_image, cholesky_factor, num_copies=NUM_NOISY_COPIES, device=None)
    print(f"Generated {noisy_images.shape[0]} noisy copies of the selected image.")
    
    # -------------------------------
    # Compute the Covariance Matrix using PyTorch operations
    # -------------------------------
    start_time = time.time()  # Start timing core logic
    cov_matrix, mean_image = compute_covariance_torch(noisy_images)
    print("Computed covariance matrix of shape:", cov_matrix.shape)
    
    # -------------------------------
    # Save the Covariance Matrix and Mean Image
    # -------------------------------
    # Convert to numpy arrays for saving, if desired
    np.save("covariance_matrix_torch.npy", cov_matrix.cpu().numpy())
    np.save("mean_image_torch.npy", mean_image.cpu().numpy())
    print("Covariance matrix and mean image saved to disk.")
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time of main logic: {elapsed_time:.2f} seconds")
    
    invertible_bool, message_bool = check_invertibility(correlation_matrix)
    print(f"Matrix: Invertible={invertible_bool}, Message='{message_bool}'")

if __name__ == '__main__':
    main()
