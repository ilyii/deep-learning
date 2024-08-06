import torch
import torch.nn as nn

class LSTMSimple(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(LSTMSimple, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).unsqueeze(0)
        lstm_out, (hidden, _) = self.lstm(embeds)
        dense_output = self.fc(hidden[-1])
        return dense_output

class GRUSimple(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(GRUSimple, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).unsqueeze(0)
        gru_out, hidden = self.gru(embeds)
        dense_output = self.fc(hidden[-1])
        return dense_output

class LSTMBatched(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, num_classes, tokenizer):
        super(LSTMBatched, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(tokenizer), embedding_dim, padding_idx=tokenizer.pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.word_embeddings(x)
        lstm_out, (hidden, _) = self.lstm(embeds)
        dense_output = self.fc(hidden[-1])
        return dense_output

class GRUBatched(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, num_classes, tokenizer):
        super(GRUBatched, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(tokenizer), embedding_dim, padding_idx=tokenizer.pad_id)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.word_embeddings(x)
        gru_out, hidden = self.gru(embeds)
        dense_output = self.fc(hidden[-1])
        return dense_output
