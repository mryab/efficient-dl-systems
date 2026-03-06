import torch

device = "cuda"
dim = 2048

A = torch.randn(dim, dim, device=device)
B = torch.randn(dim, 2, device=device)

torch.cuda.synchronize()

C = torch.matmul(A, B)

torch.cuda.synchronize()
