#!/bin/bash

#SBATCH --job-name=testing_slurm
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1500GB
#SBATCH --output=/share/nas2_3/amahmoud/week5/test_output/.out/vqvae_train_%j.log
#SBATCH --error=/share/nas2_3/amahmoud/week5/test_output/.err/vqvae_train_%j.err

# Display GPU status
nvidia-smi

# Increase the limit for open file descriptors
ulimit -n 65536

# Set up WandB environment variables
export WANDB_API_KEY="7391c065d23aad000052bc1f7a3a512445ae83d0"
export WANDB_PROJECT="Standard AE"
export WANDB_ENTITY="eazo-ahmad-abdelhakam-the-university-of-manchester"
export WANDB_CACHE_DIR="/share/nas2_3/amahmoud/wandb_cache"
export WANDB_DATA_DIR="/share/nas2_3/amahmoud/astro/wandb_data/"

echo ">> Starting setup"

# Activate the virtual environment
source /share/nas2_3/amahmoud/.venv/bin/activate
echo ">> Environment activated"

# Verify Python version
python --version
/share/nas2_3/amahmoud/.venv/bin/python --version

# Define script and configuration file paths
PYTHON_SCRIPT="/share/nas2_3/amahmoud/week5/sem2work/artificial_noisy_main_autoencoder.py"
SWEEP_CONFIG="/share/nas2_3/amahmoud/week5/sem2work/sweep_config.yaml"

# Initialize the WandB sweep and capture output
temp_file=$(mktemp)
wandb sweep "$SWEEP_CONFIG" > "$temp_file" 2>&1

# Extract the Sweep ID from the output
SWEEP_PATH=$(grep -o 'eazo-ahmad-abdelhakam-the-university-of-manchester/Standard AE/[^ ]*' "$temp_file" | tail -n 1 | tr -d "'")

# Remove the temporary file
rm "$temp_file"

# Check if Sweep ID was successfully retrieved
if [[ -z "$SWEEP_PATH" ]]; then
    echo "Error: Sweep ID not found. Please check your configuration."
    exit 1
fi

# Start the WandB sweep agent
echo ">> Starting sweep agent for Sweep Path: $SWEEP_PATH"
wandb agent "$SWEEP_PATH"
