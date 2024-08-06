import os
from collections import Counter, defaultdict
from tqdm import tqdm
import spacy
import torch

from config import data_path, wkey, nlp

def load_data(split, amount=-1):
    data = defaultdict(dict)
    indices_to_remove = []

    with open(os.path.join(data_path, split + "_text.txt"), "r", encoding="utf-8") as f:
        txt_lines = f.read().strip().splitlines()
        txt_lines = txt_lines[:amount] if amount != -1 else txt_lines
        for i, line in enumerate(tqdm(txt_lines, desc=f"Loading {split} data")):
            data[i]["text"] = line.strip()
            clean_words = []
            words = []
            for token in nlp(line.strip()):
                words.append(token.text)
                if token.is_alpha and not token.is_stop:
                    clean_words.append(token.lemma_.lower())
                elif token.text in ["!", "?", "#", "@"]:
                    clean_words.append(token.text)
            data[i]["clean_words"] = clean_words
            data[i]["words"] = words

            if len(clean_words) == 0:
                indices_to_remove.append(i)

    with open(os.path.join(data_path, split + "_labels.txt"), "r", encoding="utf-8") as f:
        txt_lines = f.read().strip().splitlines()
        txt_lines = txt_lines[:amount] if amount != -1 else txt_lines
        for i, line in enumerate(txt_lines):
            data[i]["label"] = int(line.strip())

    for indice in indices_to_remove:
        del data[indice]

    print(f"Removed {len(indices_to_remove)} samples with no clean words")
    return data

def get_word_freq(data, key="words"):
    c = Counter()
    for data in data.values():
        c.update(data[key])
    return c

def get_k_most_common_words(k, word_freq):
    return [word for word, _ in word_freq.most_common(k)]

def build_word_to_ix(k_most_words):
    word_to_ix = {word: i for i, word in enumerate(k_most_words)}
    word_to_ix["<UNK>"] = len(word_to_ix)
    word_to_ix["<PAD>"] = len(word_to_ix)
    return word_to_ix

class WordTokenizer:
    def __init__(self, word_to_ix):
        self.word_to_ix = word_to_ix
        self.ix_to_word = {v: k for k, v in word_to_ix.items()}
        self.unknown_id = word_to_ix["<UNK>"]
        self.pad_id = word_to_ix["<PAD>"]

    def __len__(self):
        return len(self.word_to_ix)

    def __call__(self, words):
        return [self.word_to_ix.get(word, self.unknown_id) for word in words]

    def pad(self, words, max_len):
        return words + [self.pad_id] * (max_len - len(words))

    def unpad(self, words):
        return words[:words.index(self.pad_id)] if self.pad_id in words else words

    def encode(self, ids):
        return [self.ix_to_word[id] for id in ids]

    def batch_encode(self, batch_ids):
        return [self.encode(ids) for ids in batch_ids]
