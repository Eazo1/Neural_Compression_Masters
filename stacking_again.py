import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from mtl_helpers import load_unnoised_data, add_correlated_noise
from encoder import Encoder
from decoder import Decoder

cholesky_factor_path = "/share/nas2_3/amahmoud/week5/sem2work/precomputed/point_spread_cholesky_factor2.pt"
cholesky_factor = torch.load(cholesky_factor_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
cholesky_factor = cholesky_factor.float()

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "text.usetex": False,                  # Use LaTeX for rendering text
    "font.family": "serif",               # Use serif font for a LaTeX look
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Nimbus Roman No9 L", "Century Schoolbook L", "FreeSerif"],   # Use Computer Modern, the default font for LaTeX
    "axes.labelsize": 12,                 # Axis label font size
    "axes.titlesize": 14,                 # Title font size
    "font.size": 12,                      # General font size
    "legend.fontsize": 12,                # Legend font size
    "xtick.labelsize": 10,                # X-axis tick label font size
    "ytick.labelsize": 10,                # Y-axis tick label font size
    "axes.linewidth": 0.2,                # Axis line width
    "grid.linewidth": 0.2,                # Grid line width
    "xtick.major.size": 5,                # Major tick size
    "xtick.minor.size": 3,                # Minor tick size
    "ytick.major.size": 5,                # Major tick size
    "ytick.minor.size": 3,                # Minor tick size
    "xtick.direction": "in",              # Tick direction inwards
    "ytick.direction": "in",              # Tick direction inwards
    "legend.frameon": True,              # Remove frame from legend
    "figure.dpi": 100,                    # Set DPI for clarity
    "savefig.dpi": 300,                   # Set DPI for saving figures
    "text.latex.preamble": r"\usepackage{amsmath}",  # Allows the use of amsmath package for LaTeX
    "lines.linewidth": 0.4
})

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
#sigma = 0.15e-3  # assumed noise std
target_var = 2 * np.square(sigma)  # Theoretical variance of the difference delta

# Load trained AE model

#model_save_path = '/share/nas2_3/amahmoud/week5/sem2work/mtl_autoencoder_model.pt'
model_save_path2 = '/share/nas2_3/amahmoud/week5/sem2work/wasserstein_autoencoder_model.pt'
model_save_path = '/share/nas2_3/amahmoud/week5/galaxy_out/autoencoder_model.pth'

autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
autoencoder.load_state_dict(torch.load(model_save_path, map_location=device))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
    
autoencoder2 = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens).to(device)
autoencoder2.load_state_dict(torch.load(model_save_path2, map_location=device))
autoencoder2.eval()
for param in autoencoder2.parameters():
    param.requires_grad = False


# Load data
#valid_dataset = load_unnoised_data('/share/nas2_3/amahmoud/week5/galaxy_out/valid_data_original.npy', device=None)
#valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)


print('Data Loaded Successfully')



# Function to denormalize images
def denormalize(tensor, mean, std):
    if tensor.dim() == 4:
        mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
    elif tensor.dim() == 3:
        mean = torch.tensor(mean).reshape(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).reshape(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    return tensor * std + mean

# Function to normalize images (inverse of denormalize)
def normalize(tensor, mean, std):
    if tensor.dim() == 4:
        mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
    elif tensor.dim() == 3:
        mean = torch.tensor(mean).reshape(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).reshape(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    return (tensor - mean) / std

# Function to save stacked images
def save_stacked_image(stacked_array, save_dir, filename):
    """
    Saves the stacked image array as a PNG file.
    Parameters:
    - stacked_array (numpy.ndarray): The stacked image array.
    - save_dir (str): Directory to save the stacked image.
    - filename (str): Filename for the stacked image.
    Returns:
    - save_path (str): Full path where the image was saved.
    """
    if stacked_array is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Ensure the array is in uint8 format
        if stacked_array.dtype != np.uint8:
            stacked_array = np.clip(stacked_array, 0, 255).astype(np.uint8)
        stacked_image = Image.fromarray(stacked_array)
        save_path = os.path.join(save_dir, filename)
        stacked_image.save(save_path)
        return save_path
    else:
        return None

# Function to split and add noise to images
def split_and_add_noise(n, cholesky_factor, std_sigma, output_dir, original_array):
    """
    Generates n noisy images by adding Gaussian noise to the original image.
    Parameters:
    - n (int): Number of noisy images to generate.
    - mu_sigma (float): Mean of the Gaussian noise.
    - std_sigma (float): Standard deviation of the Gaussian noise.
    - output_dir (str): Directory to save noisy images.
    - original_array (numpy.ndarray): Original image array.
    Returns:
    - None
    """
    # Clear the output directory
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for i in range(n):
        # 1) original → PIL → Tensor (C×H×W)
        img = Image.fromarray(original_array)
        img_t = transforms.ToTensor()(img).to(device)

        # 2) add_correlated_noise expects a batch, so it may add a leading dim
        noisy_batch = add_correlated_noise(img_t, cholesky_factor)  
        # noisy_batch.shape == (1, C, H, W)

        # 3) squeeze off the batch dimension
        if noisy_batch.dim() == 4 and noisy_batch.size(0) == 1:
            noisy_t = noisy_batch.squeeze(0)    # now (C, H, W)
        else:
            noisy_t = noisy_batch

        # 4) back to PIL and save
        noisy_pil = transforms.ToPILImage()(noisy_t.cpu())
        noisy_pil.save(os.path.join(output_dir, f'noisy_image_{i+1}.png'))
        
    return None

# Function to stack images via normal stacking (Mean)
def stack_images(image_dir, n, method='mean'):
    """
    Stacks a series of images by computing the mean or median across them.
    Parameters:
    - image_dir (str): Directory where the images are stored.
    - n (int): Number of images to stack.
    - method (str): 'mean' for mean stacking, 'median' for median stacking.
    Returns:
    - stacked_array (numpy.ndarray): The stacked image array.
    """
    # Initialize a list to hold image arrays
    image_arrays = []

    # Loop over the images and load them
    for i in range(1, n + 1):
        image_path = os.path.join(image_dir, f'noisy_image_{i}.png')
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('L')
            image_array = np.array(image, dtype=np.float64)
            image_arrays.append(image_array)
        else:
            print(f"Image {image_path} does not exist. Skipping.")

    # Check if we have images to stack
    if not image_arrays:
        print("No images found to stack.")
        return None

    # Stack images using the specified method
    if method == 'mean':
        stacked_array = np.mean(image_arrays, axis=0)
    elif method == 'median':
        stacked_array = np.median(image_arrays, axis=0)
    else:
        print(f"Unknown stacking method '{method}'. Use 'mean' or 'median'.")
        return None

    # Optionally, clip the stacked array to ensure valid pixel values
    stacked_array = np.clip(stacked_array, 0, 255).astype(np.uint8)

    return stacked_array  # Return the stacked array for further use

# Function to calculate MSE
def calculate_mse(original_array, stacked_array):
    """
    Calculates the Mean Squared Error between the original image and the stacked image.
    Parameters:
    - original_array (numpy.ndarray): Array of the original image.
    - stacked_array (numpy.ndarray): Array of the stacked image.
    Returns:
    - mse (float): The Mean Squared Error between the two images.
    """
    # Ensure the images are the same size
    if original_array.shape != stacked_array.shape:
        print("Error: The original image and stacked image have different dimensions.")
        return None

    # Compute the Mean Squared Error
    mse = np.mean((original_array.astype(np.float64) - stacked_array.astype(np.float64)) ** 2)

    return mse

# Function to stack images via the model (Model-Based Stacking)
def model_stack_images(n, output_dir, device, mean, std):
    """
    Stacks images using the model by encoding, reconstructing, and averaging the reconstructions.
    Renormalization is done only once at the end.
    """
    image_tensors = []

    for i in range(1, n + 1):
        image_path = os.path.join(output_dir, f'noisy_image_{i}.png')
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('L')
            image_tensor = transforms.ToTensor()(image).to(device)  # Shape: (1, H, W)
            # We no longer normalize during each iteration
            image_tensors.append(image_tensor)
        else:
            print(f"Image {image_path} does not exist. Skipping.")

    if not image_tensors:
        print("No images found for model stacking.")
        return None

    # Stack into batch
    batch_tensor = torch.stack(image_tensors, dim=0)  # Shape: (n, 1, H, W)

    with torch.no_grad():
        # Forward pass through the model
        x_recon = autoencoder(batch_tensor)
        x_recon2 = autoencoder2(batch_tensor)

        # Now perform renormalization only once at the end
        x_recon_denorm = denormalize(x_recon, mean, std)  # Denormalize the batch of images
        x_recon_np = x_recon_denorm.cpu().numpy()  # Shape: (n, 1, H, W)
        
        x_recon_denorm2 = denormalize(x_recon2, mean, std)  # Denormalize the batch of images
        x_recon_np2 = x_recon_denorm2.cpu().numpy()  # Shape: (n, 1, H, W)

    # Stack the reconstructed images by averaging
    stacked_recon = np.mean(x_recon_np, axis=0).squeeze()  # Shape: (H, W)
    stacked_recon2 = np.mean(x_recon_np2, axis=0).squeeze()  # Shape: (H, W)

    # Ensure the array is in [0, 255] and uint8
    stacked_recon = np.clip(stacked_recon, 0, 1)  # Assuming denormalize scales to [0,1]
    stacked_recon = (stacked_recon * 255).astype(np.uint8)
    
    stacked_recon2 = np.clip(stacked_recon2, 0, 1)  # Assuming denormalize scales to [0,1]
    stacked_recon2 = (stacked_recon2 * 255).astype(np.uint8)

    return stacked_recon, stacked_recon2  # Return the stacked reconstructions


# Main function
def main():
    # Paths (update these paths as necessary)
    #model_path = '/share/nas2_3/adey/astro/galaxy_out/vqvae_model.pth'
    #model_path = '/share/nas2_3/amahmoud/week5/galaxy_out/vqvae_model.pth'
    # latent_vectors_path = '/share/nas2_3/adey/astro/galaxy_out/latent_vectors.npy'
    # latent_labels_path = '/share/nas2_3/adey/astro/galaxy_out/latent_labels.npy'  # If labels are used

    # Check if files exist
    #if not os.path.exists(model_path):
    #    raise FileNotFoundError(f"Model file not found: {model_path}")
    # if not os.path.exists(latent_vectors_path):
    #     raise FileNotFoundError(f"Latent vectors file not found: {latent_vectors_path}")
    # if not os.path.exists(latent_labels_path):
    #     raise FileNotFoundError(f"Latent labels file not found: {latent_labels_path}")

    # Load original image
    image_path = '/share/nas2_3/adey/astro/saved_image.png'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original image not found: {image_path}")
    original_image = Image.open(image_path).convert('L')
    original_array = np.array(original_image)
    print(f"Original image loaded from {image_path}")

    # Define normalization parameters (ensure these match your model's training)
    # These values are inferred from the denormalize function in your code
    mean = (0.0031,)
    std = (0.0352,)

    # Directories for stacking methods
    output_dir = '/share/nas2_3/amahmoud/week5/sem2work/IMAGES'
    stacked_normal_dir = '/share/nas2_3/amahmoud/week5/sem2work/STACKED_NORMAL'
    stacked_model_dir = '/share/nas2_3/amahmoud/week5/sem2work/STACKED_MODEL'
    
    output_dir2 = '/share/nas2_3/amahmoud/week5/sem2work/IMAGES2'
    stacked_normal_dir2 = '/share/nas2_3/amahmoud/week5/sem2work/STACKED_NORMAL2'
    stacked_model_dir2 = '/share/nas2_3/amahmoud/week5/sem2work/STACKED_MODEL2'

    # Ensure the output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stacked_normal_dir, exist_ok=True)
    os.makedirs(stacked_model_dir, exist_ok=True)

    # Parameters for noise addition
    mu_sigma = 0.  # Zero mean
    std_sigma = 1.  # Standard deviation of the noise

    # Lists to store MSE values for both stacking methods
    mse_normal_list = []
    mse_model_list = []
    
    mse_normal_list2 = []
    mse_model_list2 = []

    # Define n_values (1 to 100 for finer binning)
    n_values = list(range(1, 31))  # n from 1 to 100
# the PSF gives you a gaussian covariance kernel (comes from the distrivtion of samples in the UV/visibility plane), gaussian from the electronic noise which is correlated to the PSF (which just happens to be a gaussian)
# CCD would be poisson noise, but the PSF (covariance) is still gaussian
# would our method work for non-gaussian noise
    # Loop over different values of n
    for n in n_values:
        print(f"\nProcessing n={n}...")

        # Step 1: Generate and save n noisy images
        split_and_add_noise(n, cholesky_factor, std_sigma, output_dir, original_array )

        # Step 2: Normal Stacking (Mean)
        stacked_normal = stack_images(image_dir=output_dir, n=n, method='mean')
        if stacked_normal is not None:
            mse_normal = calculate_mse(original_array, stacked_normal)
            print(f"Normal Stacking - n={n}, MSE={mse_normal}")
            # Save the stacked normal image
            stacked_normal_filename = f'stacked_normal_n{n}.png'
            stacked_normal_path = save_stacked_image(stacked_normal, stacked_normal_dir, stacked_normal_filename)
        else:
            mse_normal = None
            stacked_normal_path = None
            print(f"Normal Stacking - n={n}, MSE=None")
        mse_normal_list.append(mse_normal)

        # Step 3: Model-Based Stacking
        stacked_model, stacked_model2 = model_stack_images(
            n=n,
            output_dir=output_dir,
            device=device,
            mean=mean,
            std=std
        )
        if stacked_model is not None:
            mse_model = calculate_mse(original_array, stacked_model)
            print(f"Model-Based Stacking - n={n}, MSE={mse_model}")
            # Save the stacked model-based image
            stacked_model_filename = f'stacked_model_n{n}.png'
            stacked_model_path = save_stacked_image(stacked_model, stacked_model_dir, stacked_model_filename)
        else:
            mse_model = None
            stacked_model_path = None
            print(f"Model-Based Stacking - n={n}, MSE=None")
        mse_model_list.append(mse_model)
####
        if stacked_model2 is not None:
            mse_model2 = calculate_mse(original_array, stacked_model2)
            print(f"Wass Model-Based Stacking - n={n}, MSE={mse_model2}")
            # Save the stacked model-based image
            stacked_model_filename2 = f'wass_stacked_model_n{n}.png'
            stacked_model_path2 = save_stacked_image(stacked_model2, stacked_model_dir2, stacked_model_filename2)
        else:
            mse_model2 = None
            stacked_model_path2 = None
            print(f"wass_Model-Based Stacking - n={n}, MSE=None")
        mse_model_list2.append(mse_model2)


    # ----------------------------
    # Plotting the MSE Comparison
    # ----------------------------
    print("\nPlotting MSE Comparison between Normal Stacking and Model-Based Stacking...")

    # Filter out None values for plotting
    n_values_plot_normal = [n for n, mse in zip(n_values, mse_normal_list) if mse is not None]
    mse_normal_plot = [mse for mse in mse_normal_list if mse is not None]

    n_values_plot_model = [n for n, mse in zip(n_values, mse_model_list) if mse is not None]
    mse_model_plot = [mse for mse in mse_model_list if mse is not None]
    
    n_values_plot_model2 = [n for n, mse in zip(n_values, mse_model_list2) if mse is not None]
    mse_model_plot2 = [mse for mse in mse_model_list2 if mse is not None]

    # Create the figure
    plt.figure(figsize=(12, 7))

    # Plot Normal Stacking
    plt.scatter(n_values_plot_normal, mse_normal_plot, marker='o', color='black', 
            label='Normal Stacking (Mean)', s=5)

    # Scatter plot for Model-Based Stacking
    plt.scatter(n_values_plot_model, mse_model_plot, marker='o', color='red', 
            label='Model-Based Stacking', s=5)
    
    plt.scatter(n_values_plot_model2, mse_model_plot2, marker='o', color='red', 
            label='Wasserstein Model-Based Stacking', s=5)

    # Title with a formal font
    plt.title(r'Comparison of Mean Squared Error (MSE) vs. Number of Images Stacked (n)', fontsize=16, family='serif')

    # X and Y labels with serif font
    plt.xlabel(r'Number of Images Stacked (n)', fontsize=14, family='serif')
    plt.ylabel(r'Mean Squared Error (MSE)', fontsize=14, family='serif')

    # Adjust the y-axis to start from zero
    plt.ylim(bottom=np.min(np.concatenate((mse_normal_plot,mse_model_plot, mse_model_plot2))))

    # Use a more detailed grid
    plt.grid(True, which='both', axis='both', linestyle='--', color='gray', linewidth=0.5)

    # Minor ticks for a more precise scientific look
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', linestyle=':', color='gray', linewidth=0.5)

    # Clean up the ticks
    plt.tick_params(axis='both', direction='in', length=6, width=1.2)

    # Add legend
    plt.legend(fontsize=12)

    # Save the plot
    stacking_comparison_path = '/share/nas2_3/amahmoud/week5/sem2work/stacking_comparison.png'
    plt.tight_layout()
    plt.savefig(stacking_comparison_path)
    plt.close()
    print(f"Stacking comparison plot saved to {stacking_comparison_path}")

    # ----------------------------
    # Save Final Stacked Images
    # ----------------------------
    print("\nSaving final stacked images...")

    # Save the last stacked normal image
    final_stacked_normal_path = save_stacked_image(stacked_normal, stacked_normal_dir, f'stacked_normal_n{n_values[-1]}.png') if stacked_normal is not None else None

    # Save the last stacked model-based image
    final_stacked_model_path = save_stacked_image(stacked_model, stacked_model_dir, f'stacked_model_n{n_values[-1]}.png') if stacked_model is not None else None
    
    final_stacked_model_path2 = save_stacked_image(stacked_model2, stacked_model_dir2, f'wass_stacked_model_n{n_values[-1]}.png') if stacked_model2 is not None else None

    # Print file paths
    print("\n----- File Paths -----")
    print(f"Original Image: {image_path}")
    if final_stacked_normal_path:
        print(f"Final Stacked Normal Image (n={n_values[-1]}): {final_stacked_normal_path}")
    else:
        print("Final Stacked Normal Image: Not Available")

    if final_stacked_model_path:
        print(f"Final Stacked Model-Based Image (n={n_values[-1]}): {final_stacked_model_path}")
    else:
        print("Final Stacked Model-Based Image: Not Available")
    print("----------------------")

    print('\nCODE FINISHED RUNNING')
    
    if final_stacked_model_path2:
        print(f"Wass Final Stacked Model-Based Image (n={n_values[-1]}): {final_stacked_model_path2}")
    else:
        print("Wass Final Stacked Model-Based Image: Not Available")
    print("----------------------")

    print('\nCODE FINISHED RUNNING')

if __name__ == '__main__':
    main()
