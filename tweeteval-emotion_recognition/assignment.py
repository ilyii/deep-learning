

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import spacy

torch.manual_seed(1)

# ------------------------------
#           RNN
# ------------------------------

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
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
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, input_size)
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        hidden = (torch.randn(1, 1, self.hidden_size), torch.randn(1, 1, self.hidden_size))
        # Step through the sequence one element at a time
        for i in inputs:
            out, hidden = self.lstm(i.view(1, 1, -1), hidden)
        scores = F.log_softmax(out.view(1, -1), dim=1)
        return scores

def preprocess(text):
    if os.path.exists(text):
        
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class TweetDataset(Dataset):
    def __init__(self, source, split, tokenizer):
        # self.tokenizer = tokenizer

        self.label2class = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
        self.text = self.preprocess(os.path.join(source, f"{split}_text.txt"))
        self.labels = self.preprocess(os.path.join(source, f"{split}_labels.txt"))

        assert len(self.text) == len(self.labels), f"Mismatch between text={len(self.text)} and labels={len(self.labels)} in length."
        self.data = [(text, label) for text, label in zip(self.text, self.labels)]
        self.tokenizer = spacy.load("en_core_web_sm")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors="pt")
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
        
        new_text = " ".join(new_text)

        return text.split("\n")




tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_dim = 128
hidden_dim = 256
source = "data/tweeteval-emotion"

train_dataset = TweetDataset(source, "train", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

lstm = LSTMModel(embedding_dim, hidden_dim)
rnn = RNNModel(embedding_dim, hidden_dim, 2, 4)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

