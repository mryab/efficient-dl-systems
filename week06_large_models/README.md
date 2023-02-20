Slides: [pdf](./lecture.pdf), [source](lecture.odp)

Lecture (video) '22 (in russian) - https://disk.yandex.ru/i/8Fx5f4ewhGLeWA

The practice notebook will be added **today(TBA)**

References:
* PyTorch gradient checkpointing - [API reference](https://pytorch.org/docs/stable/checkpoint.html)
* PyTorch native ZeRO# - [FullyShardedDataParallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
* GPipe (one good implementation of pipelining) - [arxiv](https://arxiv.org/abs/1811.06965)
* Megatron-LM - one honking great implementation of large-scale training for transformers - [repo](https://github.com/NVIDIA/Megatron-LM)
* DeepSpeed (a library of many tricks) - [repo](https://github.com/microsoft/DeepSpeed)
    * Parameter/Optimizer State Sharding in ZeRO - [arxiv](https://arxiv.org/pdf/1910.02054v3.pdf) [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
    * ZeRO-offload - moving gradients and statistics from GPU into RAM - [arxiv](https://arxiv.org/abs/2101.06840) [blog](https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html)
* Alpa (automated parallelism in jax - https://github.com/alpa-projects/alpa
    * ICML'22 tutorial: https://sites.google.com/view/icml-2022-big-model
* FairScale - sharded DDP and pipeline from Meta - [repo](https://github.com/facebookresearch/fairscale)
* [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel) - automated tensor parallelism in PyTorch
