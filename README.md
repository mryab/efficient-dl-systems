# Efficient Deep Learning Systems
This repository contains materials for the Efficient Deep Learning Systems course taught at the [Faculty of Computer Science](https://cs.hse.ru/en/) of [HSE University](https://www.hse.ru/en/) and [Yandex School of Data Analysis](https://academy.yandex.com/dataschool/).

__This branch corresponds to the ongoing 2025 course. If you want to see full materials of past years, see the ["Past versions"](#past-versions) section.__

# Syllabus
- [__Week 1:__](./week01_intro) __Introduction__
  - Lecture: Course overview and organizational details. Core concepts of the GPU architecture and CUDA API.
  - Seminar: CUDA operations in PyTorch. Introduction to benchmarking.
- [__Week 2:__](./week02_management_and_testing) __Experiment tracking, model and data versioning, testing DL code in Python__
  - Lecture: Experiment management basics and pipeline versioning. Configuring Python applications. Intro to regular and property-based testing.
  - Seminar: Example DVC+Weights & Biases project walkthrough. Intro to testing with pytest.
- [__Week 3:__ ](./week03_fast_pipelines) __Training optimizations, FP16/BF16/FP8 formats, profiling deep learning code__
  - Lecture: Measuring performance of GPU-accelerated software. Mixed-precision training. Data storage and loading optimizations. Tools for profiling deep learning workloads. 
  - Seminar: Automatic Mixed Precision in PyTorch. Dynamic padding for sequence data and JPEG decoding benchmarks. Basics of profiling with py-spy, PyTorch Profiler, Memory Snapshot and Nsight Systems.
- [__Week 4:__](./week04_data_parallel) __Data-parallel training and All-Reduce__
  - Lecture: Introduction to distributed training. Data-parallel training of neural networks. All-Reduce and its efficient implementations.
  - Seminar: Introduction to PyTorch Distributed. Data-parallel training primitives.
- [__Week 5:__](./week05_large_models) __Sharded data-parallel training, distributed training optimizations__
- __Week 6:__ __Training large models__
- [__Week 7:__](./week07_application_deployment) __Python web application deployment__
  - Lecture/Seminar: Building and deployment of production-ready web services. App & web servers, Docker, Prometheus, API via HTTP and gRPC.
- __Week 8:__ __LLM inference optimizations and software__
- __Week 9:__ __Efficient model inference__
- __Week 10:__ Guest lecture

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
- [Yaroslav Zolotarev](https://github.com/Q-c7)
- [Maksim Abraham](https://github.com/fdrose)
- [Gregory Leleytner](https://github.com/RunFMe)
- [Antony Frolov](https://github.com/antony-frolov)
- [Anton Chigin](https://github.com/achigin)
- [Alexander Markovich](https://github.com/markovka17)
- [Roman Gorb](https://github.com/rvg77)

# Past versions
- [2024](https://github.com/mryab/efficient-dl-systems/tree/2024)
- [2023](https://github.com/mryab/efficient-dl-systems/tree/2023)
- [2022](https://github.com/mryab/efficient-dl-systems/tree/2022)
- [2021](https://github.com/yandexdataschool/dlatscale_draft)
