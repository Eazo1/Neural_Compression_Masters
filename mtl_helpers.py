import os
import numpy as np
import torch
import torch.linalg
from torch.utils.data import DataLoader, Dataset
import scipy
from scipy.stats import norm
from scipy.stats.qmc import Halton

def load_unnoised_data(data_path, device):
    data = np.load(data_path, mmap_mode='r')
    return MemoryMappedDataset(data, device)

def load_noised_data(data_path, device, given_cholesky_factor):
    data = np.load(data_path, mmap_mode='r')
    return MemoryMappedDatasetWithCorrelatedNoise(data, device, given_cholesky_factor)

# Function to compute KL divergence between two univariate Gaussians
def kl_divergence_two_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    return np.log(sigma_2 / sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2) - 0.5


# def matrix_sqrt(M: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the matrix square root of a positive semi-definite matrix M
#     using eigen-decomposition. Assumes M is symmetric PSD.
#     """
#     # Eigen-decomposition: M = V diag(e) V^T
#     e, V = torch.linalg.eigh(M)
#     # Square root of eigenvalues
#     e_sqrt = e.clamp(min=0.0).sqrt()
#     # Reconstruct the square root
#     M_sqrt = (V * e_sqrt) @ V.transpose(-2, -1)
#     return M_sqrt

def matrix_sqrt(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Ensure symmetry and add a very small epsilon for stability
    M = (M + M.transpose(-1, -2)) / 2.0
    M = M + eps * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    
    # Move the matrix to CPU and convert to double precision for better numerical stability
    M_cpu = M.cpu().double()
    
    # Perform eigen-decomposition on CPU
    e, V = torch.linalg.eigh(M_cpu)
    
    # Clamp eigenvalues to avoid negatives due to numerical issues
    e_clamped = torch.clamp(e, min=0.0)
    
    # Compute the square root in double precision
    M_sqrt_cpu = V @ torch.diag(torch.sqrt(e_clamped)) @ V.transpose(-1, -2)
    
    # Move the result back to the original device (GPU) and convert to float32
    return M_sqrt_cpu.to(M.device).float()

def matrix_sqrt_taylor(M: torch.Tensor) -> torch.Tensor: # first order taylor approx
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    return I + 0.5 * (M - I)

# def matrix_sqrt(M: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
#     # Move M to CPU for the heavy computation
#     M_cpu = M.cpu()
#     batch = M_cpu.shape[0] if M_cpu.dim() == 3 else 1
#     I = torch.eye(M_cpu.shape[-1], device=M_cpu.device, dtype=M_cpu.dtype).expand(batch, -1, -1)
    
#     normM = M_cpu.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
#     Y = M_cpu / normM
#     Z = I.clone()
    
#     for i in range(num_iters):
#         T = 0.5 * (3 * I - Z @ Y)
#         Y = Y @ T
#         Z = T @ Z
    
#     Y = Y * torch.sqrt(normM)
#     return Y.to(M.device)  # Move result back to GPU if needed

def wasserstein2_two_multivariate_gaussians(m1: torch.Tensor, K1: torch.Tensor,
                          m2: torch.Tensor, K2: torch.Tensor, correlation_matrix_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared 2-Wasserstein distance between two Gaussians N(m1, K1) and N(m2, K2).
    
    Args:
        m1: Mean of the first distribution (shape [d]).
        K1: Covariance of the first distribution (shape [d, d]).
        m2: Mean of the second distribution (shape [d]).
        K2: Covariance of the second distribution (shape [d, d]).
    
    Returns:
        A scalar tensor representing W_2^2(N(m1,K1), N(m2,K2)).
    """
    # 1) Squared difference of means
    diff = m1 - m2
    mean_sq_term = (diff ** 2).sum()
    
    # 2) Covariance term
    #K2_sqrt = matrix_sqrt(K2)
    # Inside term: K2^(1/2) K1 K2^(1/2)
    inside = correlation_matrix_sqrt @ K1 @ correlation_matrix_sqrt
    inside_sqrt = matrix_sqrt_taylor(inside)
    
    cov_term = torch.trace(K1 + K2 - 2.0 * inside_sqrt)
    
    return mean_sq_term + cov_term

def chi_squared_covariance(Sigma_recon, Sigma_orig):
    """
    Compute the Chi-Squared statistic between the estimated and expected noise covariance matrices.

    Args:
        Sigma_recon (torch.Tensor): Empirical covariance matrix from noisy images.
        Sigma_orig (torch.Tensor): Precomputed correlation matrix.

    Returns:
        float: Chi-Squared statistic.
    """
    device = Sigma_orig.device
    Sigma_recon = Sigma_recon.to(device)

    # Compute difference
    diff = Sigma_recon - Sigma_orig

    # Compute Chi-Squared statistic
    Sigma_orig_inv = torch.linalg.inv(Sigma_orig)  # Safe to use inv() since we verified invertibility
    chi2 = torch.trace(diff.T @ Sigma_orig_inv @ diff)

    return chi2 #.item()  # Return scalar value

def chi_squared_residuals(m_residual, Sigma_orig):
    """
    Compute chi-squared statistic for a single residual vector.
    Args:
        m_residual (torch.Tensor): Residual vector (flattened image).
        Sigma_orig (torch.Tensor): Original covariance matrix.
    Returns:
        torch.Tensor: Chi-squared scalar value.
    """
    Sigma_inv = torch.linalg.inv(Sigma_orig)
    chi2 = m_residual.T @ Sigma_inv @ m_residual
    return chi2

def generate_correlated_noise(correlation_matrix, device="cpu"):
    N = correlation_matrix.shape[0]  # This will be H * W
    L = torch.linalg.cholesky(correlation_matrix)  # Cholesky decomposition
    iid_noise = torch.randn(N, device=device)  # Generate IID Gaussian noise
    correlated_noise = L @ iid_noise  # Apply Cholesky factor to introduce correlation
    return correlated_noise

'''def generate_correlated_noise(correlation_matrix, num_samples=1000, device="cpu"):
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
'''

'''def precompute_cholesky(correlation_matrix):
    return torch.linalg.cholesky(correlation_matrix)

def add_correlated_noise(images, c_factor):
    print("Shape of images before fixing:", images.shape)  # Debugging print

    # Fix the extra singleton dimensions
    images = images.squeeze()  # Remove all dimensions of size 1
    print("Shape of images after fixing:", images.shape)  # Should be (N, C, H, W)

    # Generate correlated noise (make sure it has the correct shape)
    noise = torch.randn_like(images)  # Noise should match images in size

    print("Shape of correlated noise:", noise.shape)

    correlated_noise = torch.matmul(c_factor, noise.view(noise.shape[0], -1).T).T
    correlated_noise = correlated_noise.view(images.shape)

    print("Shape after applying correlated noise:", correlated_noise.shape)

    return images + correlated_noise'''
    
def add_correlated_noise(images, cholesky_factor):
    """
    Adds correlated noise to images while ensuring correct channel dimensions.
    """
    device = cholesky_factor.device
    images = images.to(device)

    # Ensure image is 4D (Batch, Channel, H, W)
    if images.dim() == 3:  
        images = images.unsqueeze(1)  # Add missing channel dim
    
    # Generate noise
    noise = torch.randn_like(images, device=device)

    # Flatten noise for matrix multiplication
    B, C, H, W = images.shape
    noise_flat = noise.view(B, C, -1).transpose(1, 2)  # Shape: (B, Pixels, C)

    # Apply Cholesky transformation
    correlated_noise = torch.matmul(cholesky_factor, noise_flat).transpose(1, 2)  # Shape: (B, C, Pixels)

    # Reshape back to image format
    correlated_noise = correlated_noise.view(B, C, H, W)

    return images + correlated_noise  # Keep batch & channel structure intact

def add_correlated_noise_halton(images, cholesky_factor, seed=42):

    device = cholesky_factor.device
    images = images.to(device)
    
    # Ensure images is 4D: (B, C, H, W)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    B, C, H, W = images.shape
    # Total number of noise values needed per image.
    num_elements = C * H * W
    
    # Create a Halton sampler in one dimension (each sample is a scalar).
    # We generate a total of B * num_elements samples.
    sampler = Halton(d=1, scramble=True, seed=seed)
    u = sampler.random(n=B * num_elements)  # shape: (B*num_elements, 1)
    
    # Map the uniform samples to standard normal using the inverse CDF.
    noise_values = norm.ppf(u)  # still shape: (B*num_elements, 1)
    
    # Convert to a torch.Tensor and reshape to (B, C, H, W)
    noise = torch.tensor(noise_values, dtype=torch.float32, device=device).view(B, C, H, W)
    
    # Flatten noise for the matrix multiplication step.
    # Rearranging to shape: (B, Pixels, C)
    noise_flat = noise.view(B, C, -1).transpose(1, 2)
    
    # Apply the Cholesky transformation.
    # Assuming cholesky_factor is shaped (C, C) and applied per pixel.
    correlated_noise = torch.matmul(cholesky_factor, noise_flat).transpose(1, 2)
    
    # Reshape back to (B, C, H, W)
    correlated_noise = correlated_noise.view(B, C, H, W)
    
    return images + correlated_noise

def compute_covariance_torch(noisy_images):
    """
    Compute the sample covariance matrix using PyTorch operations.
    
    Args:
        noisy_images: torch.Tensor of shape (N, C, H, W)
    
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
    cov_matrix = torch.mm(deviations.T, deviations) / (N - 1)
    
    return cov_matrix, mean_image

def generate_noisy_copies(original_image, cholesky_factor, num_copies=50, device=None):
    copies = []
    
    # Convert image to float tensor on the specified device.
    image = original_image.to(dtype=torch.float32, device=device)
    
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

'''def add_gaussian_noise(images, mean=NOISE_MEAN, std=NOISE_STD):
    noise = torch.normal(mean=mean, std=std, size=images.size()).to(images.device)
    integer_noise = torch.floor(noise)
    noisy_images = images + integer_noise
    return noisy_images
'''

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a tensor in the shape stored in the npy file.
        return torch.tensor(self.data[idx], dtype=torch.float32)

'''# Modified Dataset class to add noise once for each image
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
        return self.noisy_data[idx]'''

class MemoryMappedDatasetWithCorrelatedNoise(MemoryMappedDataset):
    def __init__(self, mmap_data, device, given_cholesky_factor):
        super().__init__(mmap_data, device)
        self.given_cholesky_factor = given_cholesky_factor
        self.noisy_data = self.add_noise_to_data(mmap_data, given_cholesky_factor)

    def add_noise_to_data(self, data, c_factor):
        # Add Gaussian noise to the entire dataset once
        noisy_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        noisy_data = add_correlated_noise(noisy_data, c_factor)
        #noisy_data = torch.abs(noisy_data)
        return noisy_data

    def __getitem__(self, idx):
        # Return the noisy data instead of original data
        return self.noisy_data[idx]
