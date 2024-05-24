#!/bin/bash
#SBATCH --job-name=mt5-small-paracrawl-finetuning
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/bad-pair-encoding/slurm_logs/%j.out
# --constraint=["a40|a100"]

CUR_PATH="/mmfs1/gscratch/ark/knylund/bad-pair-encoding"
LANGUAGE="German"

python -u ${CUR_PATH}/finetuning/finetune_paracrawl.py \
    --language $LANGUAGE \
    --train_data_path "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-1" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-2" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-3" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-4" \
                      "${CUR_PATH}/paracrawl_data/${LANGUAGE}/train-5" \
    --dev_data_path "${CUR_PATH}/paracrawl_data/${LANGUAGE}/dev-small" \
    --test_data_path "${CUR_PATH}/paracrawl_data/${LANGUAGE}/evaluation" \
    --model "google/mt5-small" \
    --tokenizer "${CUR_PATH}/mt_tokenizers/gpt2_English-${LANGUAGE}_morpheme_tokenizer" \
    --using_morpheme_tokenizer \
    --eval_steps 5000 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --out_dir "${CUR_PATH}/mt_models/mt5-small_English-${LANGUAGE}_${LANGUAGE}_morpheme_tokenizer"