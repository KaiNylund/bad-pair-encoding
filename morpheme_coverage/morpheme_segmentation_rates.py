import pandas as pd

import torch
import argparse
import numpy as np
from datasets import Dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import os

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--test_data_path", nargs="+", required=True)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--tokenizer", default="google/mt5-small")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--unk_strategy", type=str, default="split")
    parser.add_argument("--added_tokens", type=int, default=None, help="top-k tokens added from the new tokenizer to the default before finetuning")
    parser.add_argument("--overwrite_cache", action="store_true", default=True)
    parser.add_argument("--using_morpheme_tokenizer", action="store_true", default=False)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    args = parser.parse_args()

    print(args.language)
    print(args.model)
    print("--------------------------------------")
    print(args.tokenizer, args.added_tokens)

    # Load model + tokenizer
    prefix = f"translate English to {args.language}: "
    #model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total_default_tokens = 0
    total_num_tokens = 0
    total_num_examples = 0

    # for "split" strategy
    def retokenize(tokens, new_tokenizer, new_tokenizer_vocab):
        re_tokenized = []
        for i in range(len(tokens)):
            if tokens[i] not in new_tokenizer_vocab:
                #print(tokens[i])
                new_toks = new_tokenizer.tokenize(tokens[i], add_special_tokens=False)
                if args.using_morpheme_tokenizer:
                    new_toks = [mtok.replace("Ġ", '▁').replace("Ã¤", "ä").replace("âĢľ", "“") for mtok in new_toks]
                #print(def_de_tok, de_tokens[i])
                if len(new_toks) > 0:
                    if tokens[i][0] != '▁' and new_toks[0] == '▁':
                        new_toks = new_toks[1:]
                    elif new_toks[0][0] == '▁' and len(re_tokenized) > 0 and re_tokenized[-1] == '▁':
                        del re_tokenized[-1]
                    re_tokenized += new_toks
            else:
                re_tokenized.append(tokens[i])
        re_tokenized = (new_tokenizer.convert_tokens_to_ids(re_tokenized) + [1])
        return re_tokenized
    

    # If our model and given tokenizers are different:
    if args.tokenizer != args.model:
        model_tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_tokenizer_vocab = set(model_tokenizer.get_vocab().keys())
        # Add top-k tokens from the new tokenizer to the default tokenizer before finetuning
        if args.added_tokens != None:
            new_tokens = []
            def_emb_idx_to_tok = {v: k for k, v in model_tokenizer.get_vocab().items()}
            new_emb_idx_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}
            cur_tok_idx = 0
            while len(new_tokens) < args.added_tokens:
                new_tok = new_emb_idx_to_tok[cur_tok_idx]
                if args.using_morpheme_tokenizer:
                    new_tok = new_tok.replace("Ġ", '▁').replace("Ã¤", "ä").replace("âĢľ", "“")
                if new_tok not in model_tokenizer_vocab:
                    new_tokens.append(new_tok)
                cur_tok_idx += 1
                if cur_tok_idx > len(tokenizer):
                    break
            print(f"Adding {len(new_tokens)} tokens")
            # update model tokenizer
            tokenizer.add_tokens(list(new_tokens))
            # add new, random embeddings for the new tokens
            #model.resize_token_embeddings(len(tokenizer))

        # retokenize unknowns with the default tokenizer
        #if args.unk_strategy == "split":
        def preprocess(sample):
            global total_default_tokens
            global total_num_tokens
            global total_num_examples
            inputs = prefix + str(sample["0"])
            targets = str(sample["1"])
            model_inputs = {}
            model_inputs["input_ids"] = tokenizer.tokenize(inputs, max_length=args.max_seq_len, truncation=True)
            total_default_tokens += len(model_inputs["input_ids"])
            model_inputs["input_ids"] = retokenize(model_inputs["input_ids"], model_tokenizer, model_tokenizer_vocab)
            labels = tokenizer.tokenize(targets, max_length=args.max_seq_len, truncation=True)
            total_default_tokens += len(labels)
            labels = retokenize(labels, model_tokenizer, model_tokenizer_vocab)
            model_inputs["attention_mask"] = ([1] * len(model_inputs["input_ids"]))
            model_inputs["labels"] = labels
            #print(model_inputs)
            total_num_tokens += len(model_inputs["input_ids"])
            total_num_tokens += len(model_inputs["labels"])
            total_num_examples += 1
            return model_inputs
    else:
        # Preprocess + tokenize paracrawl splits
        def preprocess(sample):
            global total_default_tokens
            global total_num_examples
            inputs = prefix + str(sample["0"])
            targets = str(sample["1"])
            model_inputs = tokenizer(inputs, max_length=args.max_seq_len, truncation=True)
            labels = tokenizer(text_target=targets, max_length=args.max_seq_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            total_default_tokens += len(model_inputs["input_ids"])
            total_default_tokens += len(model_inputs["labels"])
            total_num_examples += 1
            return model_inputs

    def load_dataset(file_names):
        all_files_data = []
        for fname in file_names:
            all_files_data.append(pd.read_csv(fname, sep="\t", header=None, on_bad_lines="skip", engine="python"))
        raw_data = pd.concat(all_files_data)
        raw_data = raw_data.dropna()
        hf_raw_data = Dataset.from_pandas(raw_data)
        return hf_raw_data.map(preprocess,
                                load_from_cache_file=(not args.overwrite_cache),
                                desc="Preprocessing dataset")

    test_dataset = load_dataset(args.test_data_path)
    print(total_default_tokens, total_default_tokens / total_num_examples)
    print(total_num_tokens, total_num_tokens / total_num_examples)
    print(total_num_examples)

    