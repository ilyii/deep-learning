import time
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

def train_single_line(epochs, model, train_data, loss_fn, optimizer, tokenizer, wkey="clean_words"):
    model.train()
    train_stats = defaultdict(list)
    start_time = time.time()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        correct = 0
        total = 0
        sum_loss = 0

        for data in train_data.values():
            model.zero_grad()
            sentence_in = torch.tensor(tokenizer(data[wkey]), dtype=torch.long)
            targets = torch.tensor([data["label"]], dtype=torch.long)
            tag_scores = model(sentence_in)
            loss = loss_fn(tag_scores, targets)
            loss.backward()
            optimizer.step()

            train_stats["loss"].append(loss.item())
            sum_loss += loss.item()
            _, predicted = torch.max(tag_scores, 1)
            total += 1
            correct += (predicted == targets).sum().item()

        train_stats["accuracy"].append((correct / total) * 100)
        train_stats["avg_loss"].append(sum_loss / total)

    print(f"Training took {time.time() - start_time:.2f} seconds")
    plt.plot(train_stats["accuracy"])
    plt.title("Training accuracy")
    plt.show()
    plt.plot(train_stats["avg_loss"])
    plt.title("Training loss")
    plt.show()

    return model

def train(pbar, model, train_loader, criterion, optimizer, device="cuda"):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss =
