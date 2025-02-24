# Large Models
* Lecture: [slides](lecture.pdf), [source](lecture.odp), [video](https://disk.yandex.ru/i/zpUT2zZorGilMw)
* Practice: [video](https://disk.yandex.ru/i/Bxp_jXdGa011Xw)
* Homework: see below



### Practice / homework
This homework consists of two parts:
- Part 1: [`./practice_part1.ipynb`](./practice_part1.ipynb) [![open in colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/main/week05_large_models/practice_part1.ipynb) - memory-efficient training and inference
- Part 2: [`./practice_part2.ipynb`](./practice_part2.ipynb) [![open in colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/main/week05_large_models/practice_part2.ipynb) - implementing model and sequence parallelism

Part 1 will require you to implement memory-saving techniques such as offloading and gradient checkpointing / accumulation. To implement offloading, you may either write your own low-level code, or use the recommended trick: write your own [autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) (similar to gradient checkpoint function) that moves the requisite modules on device just in time for computation. Our practice video ('25) contains some tips on extending autograd functions, but those are optional.

Part 2 is much more convenient with multiple GPUs - though, it can *potentially* be solved by emulating GPUs with CPU-only code. 
For YSDA and HSE students, you can use either DataSphere or one of the GPU servers available for this course (recommended). If you are an online student, you can try to register for kaggle kernels ([they ley you run on 2x T4](https://www.kaggle.com/discussions/product-feedback/361104)) in jupyter-like interface. That said, implementing assignments B and C in Kaggle is more difficult than intended. For non-enrolled online students, we recommend option A unless you have access to some other multi-GPU-hardware or are intentionally masochistic.


### References

* PyTorch gradient checkpointing - [API reference](https://pytorch.org/docs/stable/checkpoint.html)
* PyTorch native ZeRO - [FullyShardedDataParallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
* GPipe (one good implementation of pipelining) - [arxiv](https://arxiv.org/abs/1811.06965)
* Megatron-LM - one honking great implementation of large-scale training for transformers - [repo](https://github.com/NVIDIA/Megatron-LM)
* DeepSpeed (a library of many tricks) - [repo](https://github.com/microsoft/DeepSpeed)
    * Parameter/Optimizer State Sharding in ZeRO - [arxiv](https://arxiv.org/pdf/1910.02054v3.pdf) [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
    * ZeRO-offload - moving gradients and statistics from GPU into RAM - [arxiv](https://arxiv.org/abs/2101.06840) [blog](https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html)
* Alpa (automated parallelism in Jax - https://github.com/alpa-projects/alpa
    * ICML'22 tutorial: https://sites.google.com/view/icml-2022-big-model
* FairScale - sharded DDP and pipeline from Meta - [repo](https://github.com/facebookresearch/fairscale)
* [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel) - automated tensor parallelism in PyTorch


During the in-class practice, we also had several PyTorch code examples that could come in handy when training large models:

__Automatic tensor parallelism:__
```python
%pip install tensor_parallel
import tensor_parallel as tp

model = create_a_regular_pytorch_model()
model = tp.tensor_parallel(model, ['cuda:0', 'cuda:1'])
outputs_as_usual = model(input_as_usual)
```

Note: [tensor_parallel](https://github.com/BlackSamorez/tensor_parallel) is one of the simplest ways to do this kind of distributed training, but not the fastest one. If you want to squeeze every last bit of performance, use [DeepSpeed](https://github.com/microsoft/DeepSpeed) or similar specialized frameworks (see `./homework_b.md`)

__Gradient checkpointing:__
```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class Checkpoint(nn.Sequential):
  def forward(self, *inputs):
    return checkpoint(super().forward, *inputs)

class Echo(nn.Module):
  def __init__(self, msg: str):
    super().__init__()
    self.msg = msg  # print this message during forward (for debugging)
  def forward(self, x):
    print("forward", self.msg)
    return x

model = nn.Sequential(
    Checkpoint(nn.Linear(1000, 1000), nn.ReLU(), Echo("layer1 done"),
               nn.Linear(1000, 1000), nn.ReLU(), Echo("layer2 done")),
    Checkpoint(nn.Linear(1000, 1000), nn.ReLU(), Echo("layer3 done"),
               nn.Linear(1000, 1000), nn.ReLU(), Echo("layer4 done")),
    nn.Linear(1000, 1000), nn.ReLU(), Echo("layer5 done"),
)

inputs = torch.randn(16, 1000, requires_grad=True)
# note: we must set inptus requires_grad=True because checkpoints require at least one input with grad for backprop
outputs = model(inputs)
outputs.norm().backward()  # Echo layers will print in the following order: 1 2 3 4 5 3 4 1 2
```
