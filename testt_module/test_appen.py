import torch
import torch.nn as nn
x = nn.Parameter(torch.Tensor(2,3))
nn.init.uniform_(x, -0.001, 0.001)
print(x)
x = x.view(1, -1)
print(x)
