import os
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim

import config
from data import load_data, TRANSFORM
from model import CNN


def train(num_epochs, device, model, optimizer, criterion, trainloader, ckptpath):
    os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
    times = []
    for epoch in range(num_epochs): 
        s_time = time.time()
        running_loss = 0.0
        trainbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"[TRAIN] Epoch {epoch+1}/{num_epochs}")
        for i, data in trainbar:
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = torch.mean((outputs.argmax(1) == labels).float())            

            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            trainbar.set_postfix(loss=running_loss/(i+1), acc=acc.item())
        times.append(time.time()-s_time)
        print(f"{(epoch+1)/num_epochs*100:.2f}% - Loss: {running_loss/len(trainloader):.4f}", end="\r", flush=True)

    print(f"Training took {sum(times):.2f}s in total ({sum(times)/num_epochs:.2f}s per epoch)")

    torch.save(model.state_dict(), ckptpath)
    return times





if __name__ == "__main__":
    trainloader, testloader = load_data(root=config.DATAPATH, transform=TRANSFORM, batchsize=config.BATCHSIZE)
    model = CNN(in_channels=3, num_classes=10).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9)
    times = train(config.NUM_EPOCHS, config.DEVICE, model, optimizer, criterion, trainloader, config.CKPTPATH)


