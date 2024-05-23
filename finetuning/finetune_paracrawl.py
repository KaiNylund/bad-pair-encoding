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

# for "split" strategy
def retokenize(tokens, new_tokenizer, new_tokenizer_vocab):
    re_tokenized = []
    for i in range(len(tokens)):
        if tokens[i] not in new_tokenizer_vocab:
            #print(tokens[i])
            new_toks = new_tokenizer.tokenize(tokens[i], add_special_tokens=False)
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

# for "average" strategy
def retokenize_embs_mean_unks(tokens, new_tokenizer, new_token_to_emb):
    re_tokenized_embs = []
    re_tokenized = []
    for i in range(len(tokens)):
        if tokens[i] not in new_token_to_emb:
            #print(tokens[i])
            new_toks = new_tokenizer.tokenize(tokens[i], add_special_tokens=False)
            #print(def_de_tok, de_tokens[i])
            if len(new_toks) > 0:
                if tokens[i][0] != '▁' and new_toks[0] == '▁':
                    new_toks = new_toks[1:]
                elif new_toks[0][0] == '▁' and len(re_tokenized) > 0 and re_tokenized[-1] == '▁':
                    del re_tokenized[-1]
                re_tokenized += new_toks
                re_tokenized_embs.append(torch.mean([new_token_to_emb[tok] for tok in new_toks], dim=0))
        else:
            re_tokenized.append(tokens[i])
            re_tokenized_embs.append(new_token_to_emb[tokens[i]])
    re_tokenized = (new_tokenizer.convert_tokens_to_ids(re_tokenized) + [1])
    re_tokenized_embs = (re_tokenized_embs + [new_token_to_emb["</s>"]])
    return re_tokenized_embs



if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--train_data_path", nargs="+", required=True)
    parser.add_argument("--dev_data_path", nargs="+", required=True)
    parser.add_argument("--test_data_path", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--tokenizer", default="google/mt5-small")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--unk_strategy", type=str, default="split")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=10000)
    parser.add_argument("--overwrite_cache", action="store_true", default=True)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    args = parser.parse_args()

    print(args.model)
    print("--------------------------------------")
    print(args.tokenizer)

    # Load model + tokenizer
    prefix = f"translate English to {args.language}: "
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.tokenizer != args.model:
        model_tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_tokenizer_vocab = set(model_tokenizer.get_vocab().keys())
        # retokenize unknowns with the default tokenizer
        if args.unk_strategy == "split":
            def preprocess(sample):
                inputs = prefix + str(sample["0"])
                targets = str(sample["1"])
                model_inputs = {}
                model_inputs["input_ids"] = tokenizer.tokenize(inputs, max_length=args.max_seq_len, truncation=True)
                model_inputs["input_ids"] = retokenize(model_inputs["input_ids"], model_tokenizer, model_tokenizer_vocab)
                labels = tokenizer.tokenize(targets, max_length=args.max_seq_len, truncation=True)
                labels = retokenize(labels, model_tokenizer, model_tokenizer_vocab)
                model_inputs["attention_mask"] = ([1] * len(model_inputs["input_ids"]))
                model_inputs["labels"] = labels
                #print(model_inputs)
                return model_inputs
        # handle unknowns in the input by averaging the embeddings from the retokenized version
        else:
            model_token_to_emb_idx = model_tokenizer.get_vocab()
            #model_emb_idx_to_token = {v: k for k, v in model_token_to_emb_idx}
            # Assuming we're using mt5 or t5
            model_embeddings = model.state_dict()['shared.weight'].detach().numpy()
            token_to_emb = {}
            for tok, emb_idx in model_token_to_emb_idx.items():
                token_to_emb[tok] = model_embeddings[emb_idx]

            def preprocess(sample):
                inputs = prefix + str(sample["0"])
                targets = str(sample["1"])
                model_inputs = {}
                input_tokens = tokenizer.tokenize(inputs, max_length=args.max_seq_len, truncation=True)
                model_inputs["input_embeds"] = retokenize_embs_mean_unks(input_tokens, model_tokenizer, model_token_to_emb_idx)
                labels = tokenizer.tokenize(targets, max_length=args.max_seq_len, truncation=True)
                labels = retokenize(labels, model_tokenizer, model_tokenizer_vocab)
                model_inputs["attention_mask"] = ([1] * len(model_inputs["input_embeds"]))
                model_inputs["labels"] = labels
                #print(model_inputs)
                return model_inputs
    else:
        # Preprocess + tokenize paracrawl splits
        def preprocess(sample):
            inputs = prefix + str(sample["0"])
            targets = str(sample["1"])
            model_inputs = tokenizer(inputs, max_length=args.max_seq_len, truncation=True)
            labels = tokenizer(text_target=targets, max_length=args.max_seq_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    def load_dataset(file_names):
        all_files_data = []
        for fname in file_names:
            all_files_data.append(pd.read_csv(fname, sep="\t", header=None, on_bad_lines="skip", engine="python"))
        raw_data = pd.concat(all_files_data)
        raw_data = raw_data.dropna()
        hf_raw_data = Dataset.from_pandas(raw_data)
        return hf_raw_data.map(preprocess,
                                batched=True,
                                load_from_cache_file=(not args.overwrite_cache),
                                desc="Preprocessing train dataset")

    train_dataset = load_dataset(args.train_data_path).remove_columns(["0", "1"])
    dev_dataset = load_dataset(args.dev_data_path).remove_columns(["0", "1"])
    test_dataset = load_dataset(args.test_data_path).remove_columns(["0", "1"])

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    bleu = load("bleu")
    chrf = load("chrf")

    all_metrics = []
    # Compute loss, BLEU, and chrf++ scores
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels)
        metrics = {
            "bleu": bleu_score["bleu"] * 100,
            "chrf++": chrf_score["score"]
        }
        all_metrics.append(metrics)
        return metrics

    # training
    # Evaluate on dev dataset every n steps during training
    training_args = Seq2SeqTrainingArguments(
        args.out_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        num_train_epochs=args.train_epochs,
        predict_with_generate=True,
        report_to="none"
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    # Evaluate on full test dataset after training
    trainer.evaluate(eval_dataset=test_dataset)
    # Save all metrics computed during training
    np.save(f"{args.out_dir}_eval_metrics", all_metrics)