import torch
import torch.nn.functional as F

torch.cuda.memory._record_memory_history(max_entries=100000)

logits = torch.randn(1024, 128000, device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, 128000, (1024,), device="cuda")
logits.requires_grad_(True)

compiled_ce = torch.compile(F.cross_entropy)

loss = compiled_ce(logits, targets)
loss.backward()

torch.cuda.memory._dump_snapshot("compiled_ce_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

print("Saved: compiled_ce_snapshot.pickle")
