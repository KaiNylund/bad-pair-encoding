import pandas as pd

import torch
import argparse
import numpy as np
from datasets import Dataset, load_dataset
from evaluate import load
from huggingface_hub import notebook_login
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MT5TokenizerFast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--original_tokenizer", default="google/mt5-small")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    # Load model + tokenizer
    old_tokenizer = MT5TokenizerFast.from_pretrained(args.original_tokenizer)

    train_dataset = load_dataset('text', data_files=args.train_data_path, split="train")

    #train_dataset = load_text_dataset(args.train_data_path)
    #print(train_dataset)
    #train_dataset = train_dataset.remove_columns(["0", "1"])
    #print(train_dataset)
    # Interleave sentences from different langauges
    #train_dataset = [val for pair in zip(list(train_dataset[0]), list(train_dataset[1])) for val in pair]

    def get_training_corpus():
        train_dataset = train_dataset["text"]
        for start_idx in range(0, len(train_dataset), args.batch_size):
            yield train_dataset[start_idx : start_idx + args.batch_size]

    #print(train_dataset)
    #print(len(train_dataset))

    #def get_training_corpus(batch_size=args.batch_size):
    #    for start_idx in range(0, len(train_dataset), batch_size):
    #        yield train_dataset[start_idx : start_idx + batch_size]

    new_tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=len(old_tokenizer))
    new_tokenizer.save_pretrained(args.out_dir)