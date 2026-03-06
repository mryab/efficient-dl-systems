import torch

device = "cuda"
dim = 4096

A = torch.randn(dim, dim, device=device)
B = torch.randn(dim, dim, device=device)

torch.cuda.synchronize()

C = torch.matmul(A, B)

torch.cuda.synchronize()
