from torch import nn
import torch
import math

a = torch.ones(1, 3)
b = torch.tensor([[2], [3]])
c = a * b
print(c)
