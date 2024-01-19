#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --error=/work/dlclarge1/faridk-quantization/latent_quantization/logs/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=qae_old
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/work/dlclarge1/faridk-quantization/latent_quantization/logs/%j_0_log.out
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --time=23:59:59

# command
. ~/.bashrc
conda activate disentangle

cd /work/dlclarge1/faridk-quantization/latent_quantization

python launchers/train_ae.py debug=False
