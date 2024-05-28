#!/bin/bash
#SBATCH --job-name=mt5-small-paracrawl-finetuning
#SBATCH --account=ark
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/bad-pair-encoding/slurm_logs/segmentation-%j.out
# --constraint=["a40|a100"]

CUR_PATH="/mmfs1/gscratch/ark/knylund/bad-pair-encoding"
SCRIPT_PATH="${CUR_PATH}/morpheme_coverage/morpheme_segmentation_rates.py"

#### Finnish

# Finnish + default tokenizer
LANGUAGE="Finnish"
EVAL_PATH="${CUR_PATH}/paracrawl_data/${LANGUAGE}/evaluation"
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "google/mt5-small"

# Finnish + Finnish tokenizer
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer"

# Finnish + morpheme tokenizer
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/gpt2_English-${LANGUAGE}_morpheme_tokenizer" \
    --using_morpheme_tokenizer

# Finnish + Finnish tokenizer + 1024 tokens
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer" \
    --added_tokens 1024

# Finnish + morpheme tokenizer + 1024 tokens
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/gpt2_English-${LANGUAGE}_morpheme_tokenizer" \
    --added_tokens 1024 \
    --using_morpheme_tokenizer


#### German

# German + default tokenizer
LANGUAGE="German"
EVAL_PATH="${CUR_PATH}/paracrawl_data/${LANGUAGE}/evaluation"
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "google/mt5-small"

# German + German tokenizer
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer"

# German + morpheme tokenizer
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/gpt2_English-${LANGUAGE}_morpheme_tokenizer" \
    --using_morpheme_tokenizer

# German + German tokenizer + 1024 tokens
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer" \
    --added_tokens 1024

# German + morpheme tokenizer + 1024 tokens
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/gpt2_English-${LANGUAGE}_morpheme_tokenizer" \
    --added_tokens 1024 \
    --using_morpheme_tokenizer


#### Russian

# Russian + default tokenizer
LANGUAGE="Russian"
EVAL_PATH="${CUR_PATH}/paracrawl_data/${LANGUAGE}/evaluation"
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "google/mt5-small"

# Russian + Russian tokenizer
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer"

# German + German tokenizer + 1024 tokens
python -u $SCRIPT_PATH \
    --language $LANGUAGE \
    --test_data_path $EVAL_PATH \
    --tokenizer "${CUR_PATH}/mt_tokenizers/mt5-small_English-${LANGUAGE}_tokenizer" \
    --added_tokens 1024