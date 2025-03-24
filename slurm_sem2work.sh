#!/bin/bash


#SBATCH --job-name=testing_slurm
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/amahmoud/week5/test_output/.out/vqvae_train_%j.log
#SBATCH --error=/share/nas2_3/amahmoud/week5/test_output/.err/vqvae_train_%j.err
#SBATCH --mem=1000GB

ulimit -n 16384

nvidia-smi

#export MPLCONFIGDIR=$HOME/.config/matplotlib
export MPLCONFIGDIR=/tmp/matplotlib-cache

export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8

echo ">>start"
source /share/nas2_3/amahmoud/.venv/bin/activate
echo ">>environment read"
python --version 
/share/nas2_3/amahmoud/.venv/bin/python --version
echo ">>running"
python3 -u /share/nas2_3/amahmoud/week5/sem2work/mtl_autoencoder_wasserstein.py
