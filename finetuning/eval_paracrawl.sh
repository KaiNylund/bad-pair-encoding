#!/bin/bash
#SBATCH --job-name=mt5-small-paracrawl-finetuning
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/bad-pair-encoding/slurm_logs/%j.out
# --constraint=["a40|a100"]

CUR_PATH="/mmfs1/gscratch/ark/knylund/bad-pair-encoding"
LANGUAGE="Russian"

python -u ${CUR_PATH}/finetuning/finetune_paracrawl.py \
    --language $LANGUAGE \
    --test_data_path "${CUR_PATH}/paracrawl_data/${LANGUAGE}/evaluation" \
    --model "${CUR_PATH}/mt_models/mt5-small_English-Russian_Russian_tokenizer_added_1024/checkpoint-135102/" \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer" \
    --added_tokens 1024 \
    --using_checkpoint \
    --out_dir "${CUR_PATH}/mt_models/mt5-small_English-${LANGUAGE}_${LANGUAGE}_tokenizer_added_1024_eval"