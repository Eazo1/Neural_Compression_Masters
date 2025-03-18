import torch
import time
import numpy as np

'''    def check_covariance_invertibility(Sigma, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Check if a covariance matrix is easily invertible using PyTorch.
        
        Args:
            Sigma (torch.Tensor): The covariance matrix to check.
            device (str): 'cuda' for GPU, 'cpu' for CPU.

        Returns:
            torch.Tensor: The inverse or pseudo-inverse of Sigma.
        """
        
        # Move to specified device (GPU or CPU)
        Sigma = Sigma.to(device)

        # Compute condition number
        cond_number = torch.linalg.cond(Sigma)
        print(f"Condition Number: {cond_number.item():.2e}")

        # Set a threshold for numerical stability
        if cond_number > 1e12:
            print("⚠️ WARNING: The covariance matrix is nearly singular. Consider using a pseudo-inverse.")

        try:
            # Attempt to compute the inverse
            Sigma_inv = torch.linalg.inv(Sigma)
            print("✅ The covariance matrix is invertible!")
            return Sigma_inv
        except RuntimeError:
            print("❌ The covariance matrix is singular or nearly singular.")
            print("⚠️ Using pseudo-inverse (Moore-Penrose) instead...")
            Sigma_pinv = torch.linalg.pinv(Sigma)
            return Sigma_pinv

    # Load the actual correlation matrix
    correlation_matrix_path = "precomputed/point_spread_correlation_matrix.pt"

    # Load and test invertibility
    correlation_matrix = torch.load(correlation_matrix_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    Sigma_inv_or_pinv = check_covariance_invertibility(correlation_matrix)
'''

# Load the actual correlation matrix (expected noise structure)
correlation_matrix_path = "precomputed/point_spread_correlation_matrix.pt"
correlation_matrix = torch.load(correlation_matrix_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
correlation_matrix = correlation_matrix.to(dtype=torch.float32)

# Load training image data from memory-mapped file
train_data_path = "/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy"
train_data_mmap = np.load(train_data_path, mmap_mode="r")

# Select a random image from the training set
num_train = len(train_data_mmap)
random_index = torch.randint(0, num_train, (1,)).item()
print(f"Randomly selected image index: {random_index}")

original_image_np = train_data_mmap[random_index].copy()  # Ensure a copy is made
test_image = torch.tensor(original_image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
test_image = test_image.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to add correlated noise
def add_correlated_noise(images, cholesky_factor):
    """
    Adds correlated noise to images using a precomputed Cholesky factor.
    
    Args:
        images (torch.Tensor): Input images (shape: (N, C, H, W)).
        cholesky_factor (torch.Tensor): Cholesky factor of covariance matrix (shape: (Pixels, Pixels)).
    
    Returns:
        torch.Tensor: Images with added correlated noise.
    """
    device = cholesky_factor.device
    images = images.to(device)
    images = images.squeeze()  # Remove unnecessary dimensions

    # Generate standard Gaussian noise
    noise = torch.randn_like(images, device=device)

    # Flatten noise to match Cholesky factor dimensions
    H, W = images.shape[-2:]
    noise_flat = noise.view(-1, H * W).T  # Shape: (H*W, 1)

    # Apply Cholesky transformation
    correlated_noise = torch.matmul(cholesky_factor, noise_flat).T  # Shape: (1, H*W)
    correlated_noise = correlated_noise.view(images.shape)  # Reshape back to image format

    return images + correlated_noise

cholesky_factor_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/point_spread_cholesky_factor.pt"
cholesky_factor = torch.load(cholesky_factor_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

# Generate noisy versions of the test image
num_noisy_samples = 10000
print('number of noisy samples:', num_noisy_samples)
noisy_images = torch.stack([add_correlated_noise(test_image, cholesky_factor) for _ in range(num_noisy_samples)])

# Compute empirical covariance matrix from 1000 noised images
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

# Compute empirical covariance matrix
start_time_cov = time.time()
empirical_cov, mean_image = compute_covariance_torch(noisy_images)
elapsed_time_cov = time.time() - start_time_cov

print(f"Computed empirical covariance matrix in {elapsed_time_cov:.4f} seconds")

# Function to compute chi-squared statistic
def chi_squared_test(Sigma_recon, Sigma_orig):
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
    start_time_chi2 = time.time()
    Sigma_orig_inv = torch.linalg.inv(Sigma_orig)  # Safe to use inv() since we verified invertibility
    chi2 = torch.trace(diff.T @ Sigma_orig_inv @ diff)
    elapsed_time_chi2 = time.time() - start_time_chi2

    return chi2.item(), elapsed_time_chi2  # Return scalar value and computation time

# Compute Chi-Squared between empirical covariance and precomputed covariance
chi2_value, elapsed_time_chi2 = chi_squared_test(empirical_cov, correlation_matrix)

print(f"Chi-Squared value: {chi2_value}")
print(f"Computed Chi-Squared statistic in {elapsed_time_chi2:.4f} seconds")

dof = empirical_cov.shape[0]  # Degrees of Freedom
chi2_reduced = chi2_value / dof
print(f"Reduced Chi-Squared: {chi2_reduced}")
