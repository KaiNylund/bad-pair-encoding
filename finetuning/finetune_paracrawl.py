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
    parser.add_argument("--train_data_path", nargs="+", required=True)
    parser.add_argument("--dev_data_path", nargs="+", required=True)
    parser.add_argument("--test_data_path", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--tokenizer", default="google/mt5-small")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=10000)
    args = parser.parse_args()

    # Load model + tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Preprocess + tokenize paracrawl splits
    prefix = f"translate English to {args.language}: "
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
        hf_raw_data = Dataset.from_pandas(raw_data)
        return hf_raw_data.map(preprocess)

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