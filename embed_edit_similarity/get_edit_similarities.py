from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import Levenshtein

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
t5_small = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
t5_vocab = t5_tokenizer.get_vocab()
t5_emb_idx_to_token = {v: k for k, v in t5_vocab.items()}
t5_small_embeddings = t5_small.state_dict()['shared.weight'].detach().numpy()
print(t5_small_embeddings.shape)

leven_dists = []
jw_dists = []
tok1s = []
tok2s = []
emb_sims = []

first_k_tokens = 5000
# skip <pad>, </s>, <unk>
for i in tqdm(range(3, 3 + first_k_tokens)):
    for j in range(3, i):
        tok1 = t5_emb_idx_to_token[i]
        tok2 = t5_emb_idx_to_token[j]
        emb1 = t5_small_embeddings[i]
        emb2 = t5_small_embeddings[j]
        leven_dists.append(Levenshtein.distance(tok1, tok2))
        jw_dists.append(Levenshtein.jaro_winkler(tok1, tok2))
        emb_sims.append(cos_sim(emb1, emb2))
        tok1s.append(tok1)
        tok2s.append(tok2)

np.save("mt5-small_embedding_edit_dist_sims", {
    "tok1": tok1s,
    "tok2": tok2s,
    "leven_dist": leven_dists,
    "jw_dist": jw_dists,
    "emb_sim": emb_sims
})