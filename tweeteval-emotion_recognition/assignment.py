

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import spacy

torch.manual_seed(1)

# ------------------------------
#           RNN
# ------------------------------

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use only the last output of the sequence
        return out


# ------------------------------
#           LSTM
# ------------------------------
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        out, (hn, cn) = self.lstm(inputs)
        out = self.fc(out[:, -1, :])
        return out


# ------------------------------
#           GRU
# ------------------------------

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        out, hn = self.gru(inputs)
        out = self.fc(out[:, -1, :])
        return out


class TweetDataset(Dataset):
    def __init__(self, source, split):
        self.tokenizer = spacy.load("en_core_web_sm") #python -m spacy download en_core_web_sm
        self.label2class = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
        self.text = self.preprocess(os.path.join(source, f"{split}_text.txt"))
        self.text = [[token.text for token in self.tokenizer(text)] for text in self.text]
        vocab = self.get_k_most_common(self.text, 5000)
        self.text = self.reduce_vocab(self.text, vocab)
        print(self.text)
        
        

        self.labels = self.preprocess(os.path.join(source, f"{split}_labels.txt"))

        assert len(self.text) == len(self.labels), f"Mismatch between text={len(self.text)} and labels={len(self.labels)} in length."
        self.data = [(text, label) for text, label in zip(self.text, self.labels)]
        


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        return inputs, label
    
    def preprocess(self, text):
        if os.path.exists(text):            
            with open(text, "r", encoding="utf-8") as f:
                text = f.read()
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        
        new_text = " ".join(new_text).split("\n")
        return new_text
    
    def get_k_most_common(self, text, k):
        vocab = {}
        for part in text:
            for word in part:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab = [elem[0] for elem in sorted(vocab.items(), key=lambda x: x[1], reverse=True)]
        return vocab[:k]
    
    def reduce_vocab(self, text, vocab):
        new_text = []
        for part in text:
            for i, word in enumerate(part):
                if word not in vocab:
                    part[i] = "<UNK>"
            new_text.append(part)
        return new_text

        


    
    




# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_dim = 128
hidden_dim = 256
source = "data/"
train_dataset = TweetDataset(source, "train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# lstm = LSTM(embedding_dim, hidden_dim)
# rnn = RNN(embedding_dim, hidden_dim, 2, 4)


print(train_dataset[0])
# def train():
#     model.train()

#     train_stats = {
#         "loss": 0,
#         "accuracy": 0
#     }
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()


