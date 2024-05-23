import pandas as pd
import re
from datasets import Dataset
from huggingface_hub import notebook_login
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MT5Model,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MT5TokenizerFast,
    T5TokenizerFast
)
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset


CUR_PATH="/mmfs1/gscratch/ark/knylund/bad-pair-encoding"
language = 'Finnish'
source_lang = "eng"
target_lang = "fin"
# Change this to the languages we want to train on
data_files = [f"{CUR_PATH}/paracrawl_data/{language}/train-1",
              f"{CUR_PATH}/paracrawl_data/{language}/train-2",
              f"{CUR_PATH}/paracrawl_data/{language}/train-3",
              f"{CUR_PATH}/paracrawl_data/{language}/train-4",
              f"{CUR_PATH}/paracrawl_data/{language}/train-5"]
dataset = load_dataset('text',
                       data_files=data_files,
                       split="train")

old_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")


def get_training_corpus():
    train_dataset = dataset["text"]
    for start_idx in range(0, len(train_dataset), 1000):
        yield train_dataset[start_idx : start_idx + 1000]

# For each language, we will have a dict entry mapping lang to [word_to_morphemes, morphemes]
lang_to_data = {}

# Get the morphynet dataset for one language
from collections import Counter
for lang in [source_lang, target_lang]:
    word_to_morphemes = {}
    morphemes = Counter()
    roots = set()

    # DERIVATIONAL

    # Start with adding the derivational morphemes
    derivational_filepath = f'{CUR_PATH}/MorphyNet/{lang}/{lang}.derivational.v1.tsv'
    derivational = pd.read_csv(derivational_filepath, sep='\t', names=["source_word", "target_word", "source_POS", "target_POS", "morpheme", "morpheme_type"])
    for index, row in tqdm(derivational.iterrows(), total=derivational.shape[0]):
      # Map the target word to the morphemes and the root in order
      # E.g.
      # tuo	toinen	V	J	inen	suffix
      # toinen, tuo, inen, suffix
      target_word, source_word, morpheme, morpheme_type = row["target_word"], row["source_word"], row["morpheme"], row["morpheme_type"]
      # The ordering of the morpheme depends on the morpheme type
      # For now let's make sure that the target_word is not in the dictionary
      if target_word in word_to_morphemes or type(source_word) is not str:
        # print(f"Source word {target_word} found in dictionary as {word_to_morphemes[target_word]} instead of {source_word} + {morpheme}")
        continue

      # We now add the word to the dictionary
      if morpheme_type == 'suffix':
        word_to_morphemes[target_word] = [target_word[:-len(morpheme)], morpheme]
        morphemes.update([(morpheme, 'suffix', 'd')])
      elif morpheme_type == 'prefix':
        word_to_morphemes[target_word] = [morpheme, target_word[len(morpheme):]]
        morphemes.update([(morpheme, 'prefix', 'd')])
      else:
        raise Exception(f'Illegal morpheme_type for {row}')

    # Sort the morphemes by prevalence of appearance
    print(morphemes)
    print(word_to_morphemes)


    # INFLECTIONAL


    # Then add the inflectional morphemes
    inflectional_filepath = f'{CUR_PATH}/MorphyNet/{lang}/{lang}.inflectional.v1.tsv'
    inflectional = pd.read_csv(inflectional_filepath, sep='\t', names=["lemma", "inflected_word", "morpheme_features", "morpheme_segmentation"])
    # inflectional = pd.concat([inflectional.head(10000), inflectional.tail(10000)])   # COMMENT THIS OUT LATER THIS IS JUST FOR TESTING
    root_POSs = ["N", "V", "ADJ"]
    for index, row in tqdm(inflectional.iterrows(), total=inflectional.shape[0]):
      # The way that it looks might vary by language
      # fin e.g.
      # koolaus	koolauksemme	N|PSS1P	koolaus|mme
      # eng e.g.
      # stonewall	stonewalls	V|PRS;3;SG	stonewall|s
      lemma, inflected_word, morpheme_features, morpheme_segmentation = row["lemma"], row["inflected_word"], row["morpheme_features"], row["morpheme_segmentation"]

      # Skip if there's no morphology
      if type(morpheme_segmentation) is not str or morpheme_segmentation == "-":
        # print(f'Skipped {morpheme_segmentation}')
        continue

      # We gotta break up the morpheme segmentation into its morphemes
      split_segmentation = morpheme_segmentation.split("|")
      try:
        split_features = [morpheme_feature.split(";") for morpheme_feature in morpheme_features.split("|")]
      except Exception:
        print(f"Had to continue on {lemma}, {morpheme_features}, {morpheme_segmentation}")
        continue
      indices_with_stems = [index for index in range(len(split_features)) for root_POS in root_POSs if root_POS in split_features[index]]

      # Make sure there's only one head in there and that it's first or last for now
      if len(indices_with_stems) != 1:
        raise Exception(f"Inflected word {inflected_word} with morpheme_features {morpheme_features} doesn't have exactly one head")
      stem_index = indices_with_stems[0]

      # For now let's make sure that the target_word is not in the dictionary
      if inflected_word in word_to_morphemes:
        # print(f"Inflected word {inflected_word} found in dictionary as {word_to_morphemes[inflected_word]} instead of {morpheme_segmentation}")
        continue

      # We don't want to handle circumfixes just yet
      if stem_index == 0:
        # If we want to handle underlying form differently here's the place to do so
        # Merge all the suffixes into one word
        root = split_segmentation[0]
        suffixes = "".join(split_segmentation[1:])
        word_to_morphemes[inflected_word] = [root, suffixes]
        # Add the morphemes to the counter
        morphemes.update([(suffixes, 'suffix', 'i')])
      elif stem_index == len(split_features) - 1:
        root = split_segmentation[-1]
        prefixes = "".join(split_segmentation[:-1])
        word_to_morphemes[inflected_word] = [prefixes, root]
        morphemes.update([(prefixes, 'prefix', 'i')])
      else:
        raise Exception(f"Circumfixes not yet handled")

      # We need to figure out which morpheme is the root
      # For the languages we've chosen, there will always be exactly one root per word
      # It will be a N, or a V for sure, other languages may have other options

    # Add it to the lang data
    lang_to_data[lang] = [word_to_morphemes, morphemes]

lang_to_data[source_lang][1] = sorted(lang_to_data[source_lang][1].items(), key=lambda x: x[1], reverse=True)
lang_to_data[target_lang][1] = sorted(lang_to_data[target_lang][1].items(), key=lambda x: x[1], reverse=True)


global word_already_token
global word_found_in_dict
global word_split
global word_unsplit

word_already_token = 0
word_found_in_dict = 0
word_split = 0
word_unsplit = 0

# Split one individual word
def split_word(word, word_to_morphemes, morphemes, preceding_token):
  # print(word)
  global word_already_token
  global word_found_in_dict
  global word_split
  global word_unsplit

  # If the word is already a token, then we return it directly
  # What this does is it adds a space to the start of each sentence (presumeably they don't have it in the dataset already)
  # It then converts the _ that mT5 puts at the start of sentences and before words with a space to check if they're the same
  if preceding_token == "-START-":
    preceding_token = " "
  previous_with_present = preceding_token + word
  old_tokenization = old_tokenizer.tokenize(previous_with_present)
  if len(old_tokenization) == 1 and old_tokenization[0].replace(chr(9601), " ") == previous_with_present:
    word_already_token += 1
    return ["MERGEWITHPREVIOUS", old_tokenization[0]]
  if len(old_tokenization) == 2 and old_tokenization[1] == word:
    word_already_token += 1
    return [old_tokenization[1]]

  # Check to see if the word is in word_to_morphemes, if it is return that
  if word in word_to_morphemes:
    word_found_in_dict += 1
    return word_to_morphemes[word]

  word_split += 1
  # Otherwise we have to find a way to split it
  for morpheme_option in morphemes:
    (morpheme, morpheme_type, d_or_i), count = morpheme_option

    # If the word is too short continue
    if len(word) < len(morpheme):
      continue

    # THIS METHOD JUST RETURNS THE REST OF THE STRING AS A TOKEN
    # See if it's a match
    if morpheme_type == 'suffix':
      if word[-len(morpheme):] == morpheme:
        return [word[:-len(morpheme)], morpheme]
    elif morpheme_type == 'prefix':
      if word[:len(morpheme)] == morpheme:
        return [morpheme, word[len(morpheme):]]
    else:
      raise Exception(f'{morpheme_type} is an illegal morpheme_type')

  word_split -= 1
  word_unsplit += 1
  # This means that there is no morpheme that matches
  return [word]


# Split the words
def split(sentence, source_or_target, word_to_morphemes, morphemes):
  # The first half of a sentence is going to be the English, then the target language
  source, target = sentence.split("\t")
  if source_or_target == "source":
    split = source
  elif source_or_target == "target":
    split = target
  else:
    raise Exception(f'source_or_target must be "source" or "target" not {source_or_target}')

  # Split it into the basic tokens
  tokens = re.findall(r'\w+|[^\w\s]|\s+', split)
  # print(f"***Working with tokens {tokens} \n\n")

  # For each token, if it's only letters, then we process it
  processed_tokens = []
  for token_index, token in enumerate(tokens):
    # If a token isalpha, then the next token will not be.
    if token.isalpha():
      preceding_token = tokens[token_index - 1] if token_index else "-START-"
      split_words_tokens = split_word(token, word_to_morphemes, morphemes, preceding_token)
      if split_words_tokens[0] == "MERGEWITHPREVIOUS":
        if token_index: # We only remove it if it's at the start
          processed_tokens.pop()
        processed_tokens.extend([split_words_tokens[1]])  # This will always be of the form of ["MERGEWITHPREVIOUS", word]
      else:
        processed_tokens.extend(split_words_tokens)
    else:
      processed_tokens.append(token)
  # print(processed_tokens)

  # We need to process the text as if it was a
  cleaned_tokens = []
  processed_token_index = 0
  while processed_token_index < len(processed_tokens):
    # print(processed_tokens[processed_token_index], processed_token_index)
    if processed_token_index == len(processed_tokens) - 1:
      cleaned_tokens.append(processed_tokens[processed_token_index])
      processed_token_index += 1
      continue
    # If any token is a whitespace and the following one doesn't have a leading underscore, then we combine them
    # MERGE WHITESPACES
    if processed_tokens[processed_token_index] == " " or (processed_token_index == 0 and processed_tokens[0][0] != chr(9601)):
      if processed_token_index == 0:
        processed_token_index = -1  # This is done so the next token is the first token
      next_tokenized_word = old_tokenizer.tokenize(processed_tokens[processed_token_index + 1])
      if len(next_tokenized_word) == 1:
        cleaned_tokens.append(next_tokenized_word[0])
        processed_token_index += 2
        continue
      cleaned_tokens.append(chr(9601))
      if processed_token_index == -1:
        cleaned_tokens.append(processed_tokens[0])
        processed_token_index += 1
      processed_token_index += 1
      continue
    # MERGE WORDS ()
    this_and_next_tokenized = old_tokenizer.tokenize(processed_tokens[processed_token_index] + processed_tokens[processed_token_index + 1])
    if len(this_and_next_tokenized) == 2 and this_and_next_tokenized[0] == chr(9601) and processed_tokens[processed_token_index + 1] != " ":  # Merge two words where the first one is preceded not by a space
      # print(this_and_next_tokenized)
      cleaned_tokens.append(this_and_next_tokenized[1])
      processed_token_index += 2
      continue
    cleaned_tokens.append(processed_tokens[processed_token_index])
    processed_token_index += 1
    continue

  # For some reason it gives multiple
  return cleaned_tokens


test_sentence = dataset["text"][10]
source, target = test_sentence.split("\t")
new_tokenized = split(test_sentence, source_or_target="source", word_to_morphemes=lang_to_data[source_lang][0], morphemes=lang_to_data[source_lang][1])
old_tokenized = old_tokenizer.tokenize(source)

print("Source is: ", source)
print("New tokenizer is: ", new_tokenized)
print("Old tokenizer is: ", old_tokenized)
print(new_tokenized)
print(old_tokenized)
print()

total = word_found_in_dict + word_split + word_unsplit + word_already_token
def percent(num):
    return str(round(num * 100, 1)) + "%"

print(f"Of {total} words, {percent(word_found_in_dict/total)} were found in the dict, " +
      f"{percent(word_split/total)} split, and {percent(word_unsplit/total)} unsplit. " +
      f"{percent(word_already_token/total)} were already in it.")

new_tokenizer = Tokenizer(models.BPE())

join_chr = chr(126)
print(join_chr)
new_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.ByteLevel(add_prefix_space=False), pre_tokenizers.CharDelimiterSplit(join_chr)]
)

# This line was copy and pasted, I'm not sure exactly how it works
new_tokenizer.train([data_file], trainer=trainers.BpeTrainer(vocab_size=len(old_tokenizer), special_tokens=["<|endoftext|>"]) )
new_tokenizer.decoder = decoders.ByteLevel()
new_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Wrap it
# from transformers import GPT2TokenizerFast
wrapped_tokenizer = T5TokenizerFast(tokenizer_object=new_tokenizer)
wrapped_tokenizer.save_pretrained(f"gpt2-English-{language}_morpheme_tokenizer")