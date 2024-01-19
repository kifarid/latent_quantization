#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --error=/work/dlclarge1/faridk-quantization/latent_quantization/logs/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=lfq
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --output=/work/dlclarge1/faridk-quantization/latent_quantization/logs/%j_0_log.out
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --time=23:59:59



cd /work/dlclarge1/faridk-quantization/latent_quantization
# command
. ~/.bashrc
conda activate disentangle
#conda install -c nvidia cuda-nvcc -y

python launchers/train_ae.py -cn train_ae_lfq eval.period=500 model_partial.latent_partial.num_latents=20