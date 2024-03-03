# Efficient Deep Learning Systems
This repository contains materials for the Efficient Deep Learning Systems course taught at the [Faculty of Computer Science](https://cs.hse.ru/en/) of [HSE University](https://www.hse.ru/en/) and [Yandex School of Data Analysis](https://academy.yandex.com/dataschool/).

__This branch corresponds to the ongoing 2024 course. If you want to see full materials of past years, see the ["Past versions"](#past-versions) section.__

# Syllabus
- [__Week 1:__](./week01_intro) __Introduction__
  - Lecture: Course overview and organizational details. Core concepts of the GPU architecture and CUDA API.
  - Seminar: CUDA operations in PyTorch. Introduction to benchmarking.
- [__Week 2:__](./week02_management_and_testing) __Experiment tracking, model and data versioning, testing DL code in Python__
  - Lecture: Experiment management basics and pipeline versioning. Configuring Python applications. Intro to regular and property-based testing.
  - Seminar: Example DVC+Weights & Biases project walkthrough. Intro to testing with pytest.
- [__Week 3:__](./week03_fast_pipelines) __Training optimizations, profiling DL code__
  - Lecture: Mixed-precision training. Data storage and loading optimizations. Tools for profiling deep learning workloads. 
  - Seminar: Automatic Mixed Precision in PyTorch. Dynamic padding for sequence data and JPEG decoding benchmarks. Basics of profiling with py-spy, PyTorch Profiler, PyTorch TensorBoard Profiler, nvprof and Nsight Systems.
- [__Week 4:__](./week04_distributed) __Basics of distributed ML__
  - Lecture: Introduction to distributed training. Process-based communication. Parameter Server architecture.
  - Seminar: Multiprocessing basics. Parallel GloVe training.
- [__Week 5:__](./week05_data_parallel) __Data-parallel training and All-Reduce__
  - Lecture: Data-parallel training of neural networks. All-Reduce and its efficient implementations.
  - Seminar: Introduction to PyTorch Distributed. Data-parallel training primitives.
- [__Week 6:__](./week06_large_models) __Training large models__
  - Lecture: Model parallelism, gradient checkpointing, offloading, sharding. 
  - Seminar: Gradient checkpointing and tensor parallelism in practice.
- [__Week 7:__](./week07_application_deployment) __Python web application deployment__
  - Lecture/Seminar: Building and deployment of production-ready web services. App & web servers, Docker, Prometheus, API via HTTP and gRPC.
- __Week 8:__ __Software for serving neural networks__
- __Week 9:__ __Efficient model inference__
- __Week 10:__ __Guest lecture__

## Grading
There will be several home assignments (spread over multiple weeks) on the following topics:
- Training pipelines and code profiling
- Distributed and memory-efficient training
- Deploying and optimizing models for production

The final grade is a weighted sum of per-assignment grades.
Please refer to the course page of your institution for details.

# Staff
- [Max Ryabinin](https://github.com/mryab)
- [Just Heuristic](https://github.com/justheuristic)
- [Alexander Markovich](https://github.com/markovka17)
- [Anton Chigin](https://github.com/achigin)
- [Ruslan Khaidurov](https://github.com/newokaerinasai)

# Past versions
- [2023](https://github.com/mryab/efficient-dl-systems/tree/2023)
- [2022](https://github.com/mryab/efficient-dl-systems/tree/2022)
- [2021](https://github.com/yandexdataschool/dlatscale_draft)
