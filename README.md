# Efficient Deep Learning Systems
This repository contains materials for the Efficient Deep Learning Systems course taught at the [Faculty of Computer Science](https://cs.hse.ru/en/) of [HSE University](https://www.hse.ru/en/) and [Yandex School of Data Analysis](https://academy.yandex.com/dataschool/).

# Syllabus
- [__Week 1:__](./week01_intro) __Introduction__
  - Lecture: Course overview and organizational details. Core concepts of the GPU architecture and CUDA API.
  - Seminar: CUDA operations in PyTorch. Introduction to benchmarking.
- [__Week 2:__](./week02_distributed) __Basics of distributed ML__
  - Lecture: Introduction to distributed training. Process-based communication. Parameter Server architecture.
  - Seminar: Multiprocessing basics. Parallel GloVe training.
- [__Week 3:__](./week03_data_parallel) __Data-parallel training and All-Reduce__
  - Lecture: Data-parallel training of neural networks. All-Reduce and its efficient implementations.
  - Seminar: Introduction to PyTorch Distributed. Data-parallel training primitives.
- [__Week 4:__](./week04_large_models) __Memory-efficient and model-parallel training__
  - Lecture: Model-parallel training, gradient checkpointing, offloading
  - Seminar: Gradient checkpointing in practice
- [__Week 5:__](./week05_fast_pipelines) __Training optimizations, profiling DL code__
  - Lecture: Mixed-precision training. Data storage and loading optimizations. Tools for profiling deep learning workloads  
  - Seminar: Automatic Mixed Precision in PyTorch. Dynamic padding for sequence data and JPEG decoding benchmarks. Basics of PyTorch Profiler and cProfile.
- [__Week 6:__](./week06_deployment) __Python web application deployment__
  - Lecture/Seminar: Building and deployment of production-ready web services. App & web servers, Docker containers, Prometheus metrics, API via HTTP and gRPC.
- [__Week 7:__](./week07_inference_software) __Software for serving neural networks__
  - Lecture/Seminar: Different formats for packing NN: ONNX, TorchScript, IR. Inference servers: OpenVINO, Triton. ML on client devices: TfJS, ML Kit, Core ML.
- [__Week 8:__](./week08_compress_models) __Optimizing models for faster inference__
  - Lecture: Knowlenge distillation, Pruning, Quantization, NAS, Efficient Architectures
  - Seminar: Quantization and distillation of Transformers
- [__Week 9:__](./week09_experiment_tracking) __Experiment tracking, model and data versioning, testing DL code in Python__
  - Lecture: Experiment management basics and pipeline versioning. Configuring Python applications. Intro to regular and property-based testing.
  - Seminar: Example DVC+W&B project walkthrough. Intro to testing with pytest.
- [__Week 10:__](./week10_invited_talks) __Invited talks__
  - [Memory Footprint Reduction Techniques for DNN Training: An Overview](./week10_invited_talks/memory_footprint_reduction.pdf). [Gennady Pekhimenko](http://www.cs.toronto.edu/~pekhimenko/), University of Toronto, Vector Institute
  - [Efficient Inference of Deep Learning Models on (GP)GPU](https://disk.yandex.ru/i/Ro8NL38ZzJDC3w). Ivan Komarov, Yandex

## Grading
There will be a total of 3 home assignments (some of them spread over several weeks). 
The final grade is a weighted sum of per-assignment grades. 
Please refer to the course page of your institution for details.

# Staff
- [Max Ryabinin](https://github.com/mryab)
- [Just Heuristic](https://github.com/justheuristic)
- [Alexander Markovich](https://github.com/markovka17)
- [Alexey Kosmachev](https://github.com/ADKosm)
- [Anton Semenkin](https://github.com/topshik/)
