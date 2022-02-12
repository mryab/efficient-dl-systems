import torch
from torch import nn

batch_size, in_size, out_size = 256, 1024, 2048

tensor = torch.randn(batch_size, in_size).to("cuda:0").half()
layer = nn.Linear(in_size, out_size).to("cuda:0").half()
layer(tensor)
