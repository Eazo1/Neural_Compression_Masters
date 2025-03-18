import torch
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

from mtl_helpers import MemoryMappedDataset, MemoryMappedDatasetWithCorrelatedNoise, precompute_cholesky

def find_correlation_matrix(image_size, fwhm_x, fwhm_y, pixel_scale=1.8, device="cpu"):
    """
    Constructs an anisotropic Gaussian correlation matrix using correct RA/DEC pixel scale
    and the physics-informed sigma derived from the point spread function.

    Args:
        image_size (int): The image dimension (assumed square).
        fwhm_x (float): Full width at half maximum in the x direction (arcsec).
        fwhm_y (float): Full width at half maximum in the y direction (arcsec).
        pixel_scale (float): Pixel scale in arcseconds.
        device (str or torch.device): CUDA or CPU device.

    Returns:
        torch.Tensor: (H*W, H*W) correlation matrix.
    """
    # Convert FWHM to sigma
    sigma_x = fwhm_x / 2.355
    sigma_y = fwhm_y / 2.355
    
    # Create pixel coordinate grid
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    pixel_coords = np.stack((x.ravel(), y.ravel()), axis=1)
    
    # Convert pixel coordinates to sky coordinates
    ra = pixel_coords[:, 0] * pixel_scale * u.arcsec
    dec = pixel_coords[:, 1] * pixel_scale * u.arcsec
    
    # Create SkyCoord objects
    sky_coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
    
    # Compute angular separation matrix
    d = sky_coords[:, None].separation(sky_coords[None, :]).arcsec  # (N, N) in arcseconds
    
    # Compute anisotropic correlation matrix
    C = np.exp(-0.5 * ((d**2 / sigma_x**2) + (d**2 / sigma_y**2)))
    
    # Set diagonal to 1
    np.fill_diagonal(C, 1.0)
    
    # Convert to torch tensor
    C_torch = torch.tensor(C, dtype=torch.float32, device=device)
    
    return C_torch

def compute_mean_vector_from_npy(data_path, given_cholesky_factor, device="cpu"):
    """
    Computes the pixel-wise mean vector from a dataset stored in a NumPy .npy file, 
    with correlated noise added to the dataset.

    Args:
        data_path (str): Path to the .npy file containing the dataset.
                         Expected shape: (N, H, W) where N is the number of images.
        given_cholesky_factor: The Cholesky factor used to generate correlated noise.
        device (str or torch.device): Device to move the tensor to.

    Returns:
        torch.Tensor: A flattened mean vector of shape (H*W,).
    """

    # Load the data with memory mapping
    data = np.load(data_path, mmap_mode='r')
    
    # Initialize the dataset with correlated noise
    data_mmap = MemoryMappedDatasetWithCorrelatedNoise(data, device=None, given_cholesky_factor=given_cholesky_factor)
    
    # Compute the mean image across all N images, using the noisy data
    noisy_data = torch.stack([data_mmap[i] for i in range(len(data_mmap))])  # Stack all noisy data
    
    # Compute the mean image across all N images
    mean_image = noisy_data.mean(dim=0)  # shape: (H, W)
    
    # Flatten the mean image to a vector
    mean_vector = mean_image.flatten()  # shape: (H*W,)
    
    # Convert to a torch tensor on the specified device
    return mean_vector.to(device)
        
#################


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 150
    fwhm_x = 6.4
    fwhm_y = 5.4
    
    correlation_matrix = find_correlation_matrix(image_size, fwhm_x, fwhm_y, device=device)
    point_spread_cholesky_factor = precompute_cholesky(correlation_matrix)
    images_mean_vector = compute_mean_vector_from_npy('/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy', point_spread_cholesky_factor, device=device)
    print('images_mean_vector size:', images_mean_vector.size())
    
    noise_mean_vector = torch.zeros_like(images_mean_vector)
    
    # Save to file for reuse
    # Check if the directory exists
    if os.path.exists("precomputed"):
        # Remove all files inside the directory
        for file_name in os.listdir("precomputed"):
            file_path = os.path.join("precomputed", file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete file
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Delete empty subdirectory

    # Recreate the directory (if it was removed)
    os.makedirs("precomputed", exist_ok=True)

    torch.save(correlation_matrix.cpu(), "precomputed/point_spread_correlation_matrix.pt")
    torch.save(point_spread_cholesky_factor.cpu(), "precomputed/point_spread_cholesky_factor.pt")
    torch.save(images_mean_vector.cpu(), "precomputed/images_mean_vector.pt")
    torch.save(noise_mean_vector.cpu(), "precomputed/point_spread_noise_mean_vector.pt") # Zeros vector of the same size as images_mean_vector
    
    print('done')
