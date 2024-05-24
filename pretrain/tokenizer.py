from datasets import load_from_disk
from transformers import AutoTokenizer

en_data_dir: str = "./data/mc4_en.hf"
fi_data_dir: str = "./data/mc4_fi.hf"


def load_dataset():
    ds_en = load_from_disk(en_data_dir)["train"]["text"]
    ds_fi = load_from_disk(fi_data_dir)["train"]["text"]
    ds_text = ds_en + ds_fi
    for start_idx in range(0, len(ds_text), 1000):
        yield ds_text[start_idx:start_idx + 1000]


def train_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    tokenizer = tokenizer.train_new_from_iterator(load_dataset(), 250112)
    print('Tokenizer Example:', tokenizer.tokenize('yritysten asettamia haasteita'))
    tokenizer.save_pretrained(f"mt5-language-specific")


def main():
    train_tokenizer()


if __name__ == "__main__":
    main()
