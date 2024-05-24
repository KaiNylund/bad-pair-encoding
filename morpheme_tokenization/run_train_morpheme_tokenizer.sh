#!/bin/bash
#SBATCH --job-name=mt5-small-paracrawl-morpheme-tokenizer-training
#SBATCH --account=ark
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/bad-pair-encoding/slurm_logs/%j.out
# --constraint=["a40|a100"]

python -u /mmfs1/gscratch/ark/knylund/bad-pair-encoding/morpheme_tokenization/train_morpheme_tokenizer.py