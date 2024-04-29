import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from collections import defaultdict
from tqdm import tqdm



class NN(nn.Module):
  """
  A simple feedforward neural network with one hidden layer, ReLU and log-softmax.

  Args:
  - in_dim: input dimension
  - hidden_dims: hidden layer dimensions
  - out_dim: output dimension
  
  """
  def __init__(self, in_dim, hidden_dims, out_dim):
    super().__init__()
    self.in_layer = nn.Linear(in_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList([])

    for i in range(1, len(hidden_dims)):
      self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
    
    self.out_layer = nn.Linear(hidden_dims[-1], out_dim)

    self.apply(self._init_weights)

  def forward(self, x):
    x = F.relu(self.in_layer(x))
    for layer in self.hidden_layers:
      x = F.relu(layer(x))
    x = F.log_softmax(self.out_layer(x), dim=1)
    return x
  
  def _init_weights(self, m):
      """Normal weight initialization."""
      if isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, mean=0, std=0.01)
          if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)


def train(model:nn.Module=None,
          dataloader_train:DataLoader=None,
          dataloader_test:DataLoader=None,
          optimizer:optim.Optimizer=None,
          criterion:nn.Module=None,
          epochs:int=0, 
          device:str = "cpu", 
          val_freq:int = 1, 
          ckpt_freq:int = 0,
          savepath:str = "./output",
          **kwargs # Placeholder for additional arguments
          ):
    
    stats = defaultdict(list)
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"[TRAIN] Epoch {epoch+1}/{epochs}")

        epoch_loss = 0
        epoch_acc = 0

        for idx, data in train_bar:
            img, label = data
            img.to(device)
            label.to(device)
            
            img = img.view(-1, 28*28)
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label)

            _, predicted = torch.max(output, 1)
            acc = predicted.eq(label).sum().item() / len(label)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        epoch_loss /= len(dataloader_train)
        epoch_acc /= len(dataloader_train)

        stats["epoch"].append(epoch)
        stats["loss"].append(epoch_loss)
        stats["acc"].append(epoch_acc)
          
        train_bar.set_postfix_str(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        # Save
        if ckpt_freq > 0 and epoch % ckpt_freq == 0:
            torch.save(model.state_dict(), os.path.join(savepath, f'{model.__class__.__name__.lower()}_{epochs}.pt'))

        if (epoch+1) % val_freq == 0 and dataloader_test is not None:
            val_loss, val_acc = test(model, dataloader_test, criterion, device)
            stats["val_epoch"].append(epoch+1)
            stats["val_loss"].append(val_loss)
            stats["val_acc"].append(val_acc)
            train_bar.set_postfix_str(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    return stats

    
def test(model:nn.Module, dataloader_test:DataLoader, criterion:nn.Module, device:str ="cpu"):
    model.eval()
    with torch.no_grad():
        
        total_loss = 0
        total_acc = 0

        val_bar = tqdm(enumerate(dataloader_test), total=len(dataloader_test), desc=f"[VAL]")
        for idx, data in val_bar:
            img, label = data
            img.to(device)
            label.to(device)

            img = img.view(-1, 28*28)
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = criterion(output, label)

            _, predicted = torch.max(output, 1)
            acc = predicted.eq(label).sum().item() / len(label)

            total_loss += loss.item()
            total_acc += acc
    
    total_loss /= len(dataloader_test)
    total_acc /= len(dataloader_test)
    return total_loss, total_acc


    
