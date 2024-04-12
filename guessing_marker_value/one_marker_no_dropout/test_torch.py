import torch.nn as nn
import torch

xd = nn.Embedding(5, 10)
with torch.no_grad():
    xd.weight[-1] = torch.zeros(10)
print(xd.weight)