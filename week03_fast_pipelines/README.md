# Week 3: Training optimizations, profiling DL code

* Lecture: [slides](./lecture.pdf)
* Seminar: [folder](./seminar)
* Homework: see [homework/README.md](homework/README.md)

## Further reading
* [Blog post about reduced precision FP formats](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)
* NVIDIA blog posts about [mixed precision training with Tensor Cores](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/), [Tensor Core performance tips](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/), [TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
* Presentations about Tensor Cores: [one](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf), [two](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21929-tensor-core-performance-on-nvidia-gpus-the-ultimate-guide.pdf), [three](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf)
* [Tensor Core Requirements](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) and [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#mptrain) sections of the [NVIDIA DL performance guide](https://docs.nvidia.com/deeplearning/performance/index.html)
* [Automatic Mixed Precision in PyTorch](https://pytorch.org/docs/stable/amp.html)
* [TF32 section of PyTorch CUDA docs](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
* [FP8 Formats for Deep Learning paper](https://arxiv.org/abs/2209.05433)
* [Float8 in PyTorch discussion](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)
* [AMP](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options), [FP16](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) and [BF16](https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options) in DeepSpeed
* [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#)
* [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html)
* [Pillow Performance benchmarks](https://python-pillow.org/pillow-perf/)
* [Faster Image Processing](https://fastai1.fast.ai/performance.html#faster-image-processing) tips from fastai docs
* [Rapid Data Pre-Processing with NVIDIA DALI](https://developer.nvidia.com/blog/rapid-data-pre-processing-with-nvidia-dali/)
* General-purpose Python profilers: [builtins (cProfile and profile)](https://docs.python.org/3/library/profile.html), [pyinstrument](https://github.com/joerick/pyinstrument), [memory_profiler](https://github.com/pythonprofilers/memory_profiler), [py-spy](https://github.com/benfred/py-spy), [Scalene](https://github.com/plasma-umass/scalene)
* [DLProf user guide](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html)
* [How to profile with DLProf](https://tigress-web.princeton.edu/~jdh4/how_to_profile_with_dlprof_may_2021.pdf)
* [Profiling and Optimizing Deep Neural Networks with DLProf and PyProf](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/)
* NVIDIA presentations on [profiling DL networks](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9339-profiling-deep-learning-networks.pdf), [profiling for DL and mixed precision](https://on-demand.gputechconf.com/gtc-cn/2019/pdf/CN9620/presentation.pdf)
* [Profiling Deep Learning Workloads](https://extremecomputingtraining.anl.gov/files/2020/08/ATPESC-2020-Track-8-Talk-7-Emani-ProfilingDLWorkloads.pdf)
* [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and [PyTorch Profiler with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) tutorial
* [torch.utils.bottleneck quick guide](https://pytorch.org/docs/stable/bottleneck.html)
* [PyTorch Autograd profiler tutorial](https://pytorch.org/tutorials/beginner/profiler.html)
* [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and [Nsight Compute](https://docs.nvidia.com/nsight-compute/2022.1/index.html) user guides
* [Video tutorial about speeding up and profiling neural networks](https://www.youtube.com/watch?v=ySGIaOb_RDY)
* [Solving Machine Learning Performance Anti-Patterns: a Systematic Approach](https://paulbridger.com/posts/nsight-systems-systematic-optimization/)