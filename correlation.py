import os
import subprocess
import tempfile
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

from MiraBest_N import MiraBest_N, MBFRFull, MBFRConfident, MBFRUncertain, MBHybrid, MBRandom
from datasets import RGZ108k

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Only if you want optional log scaling

def main():
    """
    Main function to load data, reshape, filter, compute correlation, and plot/save the result.
    """
    # 1) Load the memory-mapped dataset
    #    Adjust the file path to your actual .npy file location.
    file_path = '/share/nas2_3/amahmoud/week5/galaxy_out/train_data.npy'
    train_data_mmap = np.load(file_path, mmap_mode='r')
    print(f"Original shape of train_data_mmap: {train_data_mmap.shape}")

    # 2) Fix the data shape by removing extra singleton dimensions (if present).
    train_data_mmap = train_data_mmap.squeeze()
    print(f"Shape after squeeze: {train_data_mmap.shape}")

    # 3) Ensure we end up with (num_images, height, width).
    if len(train_data_mmap.shape) == 3:
        num_images, height, width = train_data_mmap.shape
    elif len(train_data_mmap.shape) == 4 and train_data_mmap.shape[1] == 1:
        num_images, _, height, width = train_data_mmap.shape
        train_data_mmap = train_data_mmap.reshape(num_images, height, width)
    else:
        raise ValueError(f"Unexpected shape {train_data_mmap.shape}. Cannot proceed.")
    print(f"Final shape (num_images, height, width): {train_data_mmap.shape}")

    # 4) Reshape the data so each image is a single row in a 2D array.
    #    (num_images, height * width)
    reshaped_data = train_data_mmap.reshape(num_images, height * width)
    print(f"Reshaped data shape (num_images, height*width): {reshaped_data.shape}")

    # 5) Replace NaNs or Inf values with 0, if any.
    if np.isnan(reshaped_data).any():
        print("Warning: NaNs detected in the dataset. Replacing with 0.")
        reshaped_data = np.nan_to_num(reshaped_data)

    # 6) Remove low-variance (near-constant) pixels. Set an appropriate threshold.
    std_dev = reshaped_data.std(axis=0)
    variance_threshold = 1e-4
    non_constant_pixels = std_dev > variance_threshold
    reshaped_data_filtered = reshaped_data[:, non_constant_pixels]
    print(f"Shape after removing low-variance pixels: {reshaped_data_filtered.shape}")

    # Double-check for any remaining NaNs after filtering.
    if np.isnan(reshaped_data_filtered).any():
        print("Warning: NaNs still detected after filtering. Replacing with 0.")
        reshaped_data_filtered = np.nan_to_num(reshaped_data_filtered)

    # 7) Compute the correlation matrix among the remaining pixels.
    correlation_matrix = np.corrcoef(reshaped_data_filtered, rowvar=False)
    if np.isnan(correlation_matrix).any():
        print("Warning: NaNs detected in correlation matrix. Replacing with 0.")
        correlation_matrix = np.nan_to_num(correlation_matrix)
    print(f"Correlation matrix computed. Shape: {correlation_matrix.shape}")

    # 8) Plot the correlation matrix.
    plt.figure(figsize=(10, 8))

    # If you want a log scale (caution with negative/zero correlation values), use:
    # plt.imshow(correlation_matrix, cmap="Greens", norm=LogNorm(), interpolation='nearest')
    # Otherwise, a standard linear colormap is typical for correlation:
    plt.imshow(correlation_matrix, cmap="Greens", interpolation='nearest')

    plt.colorbar(label="Correlation")
    plt.title("Correlation Matrix of Pixel Intensities Across All Images")

    # 9) Save the plot as a PDF file (change path/name as you like).
    output_path = '/share/nas2_3/amahmoud/week5/sem2work/correlation_pic.pdf'
    plt.savefig(output_path, dpi=300)
    print(f"Correlation matrix figure saved to: {output_path}")

    # 10) Show the figure if desired (comment out if running headless).
    plt.show()

if __name__ == "__main__":
    main()