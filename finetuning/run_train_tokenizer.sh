#!/bin/bash
#SBATCH --job-name=mt5-small-paracrawl-tokenizer-training
#SBATCH --account=ark
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/bad-pair-encoding/slurm_logs/%j.out


CUR_PATH="/mmfs1/gscratch/ark/knylund/bad-pair-encoding"
LANGUAGE="Finnish"

python -u ${CUR_PATH}/finetuning/train_tokenizer.py \
    --train_data_path "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-1" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-2" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-3" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-4" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-5" \
    --original_tokenizer "google/mt5-small" \
    --out_dir "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer" \
    --batch_size 1024