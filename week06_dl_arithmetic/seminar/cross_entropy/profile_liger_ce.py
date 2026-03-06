import torch
from liger_kernel.transformers import LigerCrossEntropyLoss

criterion = LigerCrossEntropyLoss()

torch.cuda.memory._record_memory_history(max_entries=100000)

logits = torch.randn(1024, 128000, device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, 128000, (1024,), device="cuda")
logits.requires_grad_(True)

loss = criterion(logits, targets)
loss.backward()

torch.cuda.memory._dump_snapshot("liger_ce_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

print("Saved: liger_ce_snapshot.pickle")
