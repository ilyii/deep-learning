import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    x = x.view(x.size(0), -1) # flatten the input
    x = F.relu(self.in_layer(x))
    
    for layer in self.hidden_layers:
      
      x = F.relu(layer(x))
    x = F.log_softmax(self.out_layer(x), dim=1)
    return x
  
  def _init_weights(self, m):
      """Normal weight initialization."""
      if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)



    


    
