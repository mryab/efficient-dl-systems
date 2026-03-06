# Week 6: Arithmetic of Deep Learning

* Lecture: [slides](./lecture.pdf)
* Seminar: [folder](./seminar)
* Homework: see [homework/README.md](homework/README.md)

## Further reading
* [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - Efficient Triton kernels for LLM training
* [Liger Kernel Paper](https://arxiv.org/abs/2410.10989) - Technical report with kernel details and benchmarks
* [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) - Memory-efficient CE for large vocabularies
* [torch.compile API](https://pytorch.org/docs/stable/generated/torch.compile.html) - Official API reference for torch.compile
* [TorchDynamo Deep-Dive](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html) - How TorchDynamo captures and transforms Python bytecode
* [Torch Compiler Troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html) - Debugging graph breaks and recompilations
* [torch.compile: The Missing Manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/) - Comprehensive guide with practical tips
* [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul) - Deep dive into GPU architecture and CUDA matmul optimization
* [GPU Mode: Optimizing Optimizers](https://www.youtube.com/watch?v=hIop0mWKPHc) - Video on fusing optimizer operations with foreach and torch.compile
* [Essential AI: Blog on efficient communications for Muon](https://www.essential.ai/research/infra) - Layer sharding for large‑scale training with Muon
* [Efficient Muon implementation in Dion library](https://github.com/microsoft/dion/) - Efficient implementations of orthonormal optimizers for distributed ML training
