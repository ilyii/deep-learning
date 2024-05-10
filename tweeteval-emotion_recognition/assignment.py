import copy
import os
import random
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import util

UNK = "<UNK>"
PAD = "<PAD>"

def load_data(path:os.PathLike, split:str):
    """
    Loading the data. Assuming the data is stored in the following format:
    
    path/
        {split}_text.txt
        {split}_labels.txt

    Args:
        path: Path to the data directory.
        split: The split to load [train, val, test].
    """
    res = []
    with open(os.path.join(path, f'{split}_text.txt'), encoding="utf-8") as text_file, \
         open(os.path.join(path, f'{split}_labels.txt'), encoding="utf-8") as label_file:
        for t, l in zip(text_file, label_file):
            res.append({
                "text": t,
                "label": int(l.strip())
            })

    return res


def load_yaml(path:os.PathLike):
    import yaml
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def clean_data(data, nlp):
    processed_data = []
    for example in data:
        doc = nlp(example["text"].strip())
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        example["tokens"] = cleaned_tokens
        processed_data.append(example)
    return processed_data


def process_data(data, nlp, clean=True):
    if clean:
        return clean_data(data, nlp)
    else:
        processed_data = []
        for example in data:
            doc = nlp(example["text"].strip())
            tokens = [token.text for token in doc]
            example["tokens"] = tokens
            processed_data.append(example)
        return processed_data


def build_vocab(data, min_freq=1, limit=None):
    counter = Counter()
    for example in data:
        counter.update(example["tokens"])
    vocab = {word: idx for idx, (word, freq) in enumerate(counter.items(), 2) if freq >= min_freq}
    if limit and limit < len(vocab):
        vocab = {word: idx for idx, (word, freq) in enumerate(counter.most_common(limit))}
        vocab[UNK] = len(vocab)
        vocab[PAD] = len(vocab)
    return vocab

        
class LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, vocab_size, out_dim, pad_idx):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        embeds = self.embed(x)
        lstm_out, (hidden, cell) = self.lstm(embeds)
        output = self.fc(hidden[-1])
        return output.unsqueeze(0)
    

class GRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, vocab_size, out_dim, pad_idx):
        super(GRU,self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        embeds = self.embed(x)
        gru_out, hidden = self.gru(embeds)
        output = self.fc(hidden[-1])
        return output.unsqueeze(0)
    

def train_single_line(model=None,
                      data=None,
                      vocab=None,
                      criterion=None,
                      optimizer=None,
                      epochs=0,
                      device=None
                    ):
    s_time = time.time()
    model.train()
    stats = defaultdict(list)
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(data), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (features, label) in data.items():
                optimizer.zero_grad()
                sentence_in = torch.tensor([vocab.get(word, vocab[UNK]) for word in features], dtype=torch.long)
                targets = torch.tensor([label], dtype=torch.long)
                sentence_in, targets = sentence_in.to(device), targets.to(device)
                
                output = model(sentence_in)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                stats["loss"].append(loss.item())
                epoch_loss += loss.item()

                predicted = torch.argmax(output)
                stats["correct"].append((predicted == targets).item())
                stats["total"].append(1)
            
                pbar.update(1)


            epoch_loss /= len(data)
            accuracy = sum(stats["correct"]) / sum(stats["total"])
            
            pbar.set_postfix_str(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}")

            
    
    print(f"Done. ({time.time()-s_time:.2f}s)")
    return stats


def test_single_line(model=None,
                     data=None,
                     vocab=None,
                     criterion=None,
                     device=None
                    ):
    s_time = time.time()
    model.eval()
    stats = defaultdict(list)
    with torch.no_grad():
        for i, (features, label) in data.items():
            sentence_in = torch.tensor([vocab.get(word, vocab[UNK]) for word in features], dtype=torch.long)
            targets = torch.tensor([label], dtype=torch.long)            
            sentence_in, targets = sentence_in.to(device), targets.to(device)
            output = model(sentence_in)            
            loss = criterion(output, targets)
            stats["loss"].append(loss.item())

            predicted = torch.argmax(output, dim=1)
            stats["correct"].append((predicted == targets).item())
            stats["total"].append(1)

    loss = sum(stats["loss"]) / len(data)
    accuracy = sum(stats["correct"]) / sum(stats["total"])
    print(f"Test Loss: {loss:.4f} Accuracy: {accuracy:.4f}")
    print(f"Done. ({time.time()-s_time:.2f}s)")
    return stats



def main(opt):
    util.set_seed(opt.seed)
    # Data
    datasets = {"train": load_data(opt.datapath, "train"),
            "val": load_data(opt.datapath, "val"),
            "test": load_data(opt.datapath, "test")}
    
    with open(os.path.join(opt.datapath, "mapping.txt")) as f:
        id2label = {int(line.split()[0]): line.split()[1] for line in f}

    # Spacy
    nlp = spacy.load("en_core_web_sm")
    datasets_raw = {split: process_data(data, nlp, clean=False) for split, data in datasets.items()}
    datasets_optimized = {split: process_data(data, nlp, clean=True) for split, data in datasets.items()}


    # Vocab
    vocab = build_vocab(datasets_optimized["train"], limit=opt.vocab_size) # "word": "idx"
    unk = vocab[UNK]
    pad = vocab[PAD]

    # Model
    model = LSTM(opt.embed_dim, opt.hidden_dim, opt.num_layers, len(vocab), opt.out_dim, pad)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    model.to(opt.device)

    train_stats = defaultdict(dict)

    # Train
    train_stats["LSTM"]["raw"] = train_single_line(model=copy.deepcopy(model),
                                                    data={i: (example["tokens"], example["label"]) for i, example in enumerate(datasets_raw["train"])},
                                                    vocab=vocab,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    epochs=opt.epochs,
                                                    device=opt.device
                                                   )
    
    train_stats["LSTM"]["optimized"] = train_single_line(model=copy.deepcopy(model),
                                                    data={i: (example["tokens"], example["label"]) for i, example in enumerate(datasets_optimized["train"])},
                                                    vocab=vocab,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    epochs=opt.epochs,
                                                    device=opt.device
                                                   )
    
    # Test
    test_stats = defaultdict(dict)
    test_stats["LSTM"]["raw"] = test_single_line(model=copy.deepcopy(model),
                                                data={i: (example["tokens"], example["label"]) for i, example in enumerate(datasets_raw["test"])},
                                                vocab=vocab,
                                                criterion=criterion,
                                                device=opt.device
                                               )
    
    test_stats["LSTM"]["optimized"] = test_single_line(model=copy.deepcopy(model),
                                                data={i: (example["tokens"], example["label"]) for i, example in enumerate(datasets_optimized["test"])},
                                                vocab=vocab,
                                                criterion=criterion,
                                                device=opt.device
                                               )
    
    # Plot

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_stats["LSTM"]["raw"]["loss"], label="Train Raw")
    ax[0].plot(train_stats["LSTM"]["optimized"]["loss"], label="Train Optimized")
    ax[0].set_title("Train Loss")
    ax[0].legend()

    ax[1].plot(test_stats["LSTM"]["raw"]["loss"], label="Test Raw")
    ax[1].plot(test_stats["LSTM"]["optimized"]["loss"], label="Test Optimized")
    ax[1].set_title("Test Loss")
    ax[1].legend()
    plt.show()
    



if __name__ == "__main__":
    config = util.DotDict(load_yaml("config.yaml"))
    
    main(config)