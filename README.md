# Efficient Deep Learning Systems
This repository contains materials for the Efficient Deep Learning Systems course, taught at the [Faculty of Computer Science](https://cs.hse.ru/en/) of [HSE University](https://www.hse.ru/en/) and [Yandex School of Data Analysis](https://academy.yandex.com/dataschool/).

__This branch corresponds to the 2026 iteration of the course. If you want to see full materials of past years, see the ["Past versions"](#past-versions) section.__

# Syllabus
- [__Week 1:__](./week01_intro) __Introduction__
  - Lecture: Course overview and logistics. Core concepts of the GPU architecture and the CUDA API.
  - Seminar: CUDA operations in PyTorch. Introduction to benchmarking.
- [__Week 2:__](./week02_fast_pipelines) __General training optimizations, profiling deep learning code__
  - Lecture: Measuring performance of GPU-accelerated software. Mixed-precision training. Data storage and loading optimizations. Tools for profiling deep learning workloads.
  - Seminar: Automatic Mixed Precision in PyTorch. Basics of profiling with py-spy, PyTorch Profiler, Memory Snapshot and Nsight Systems. Dynamic padding for sequence data and JPEG decoding benchmarks.
- [__Week 3:__](./week03_data_parallel) __Data-parallel training and All-Reduce__
  - Lecture: Introduction to distributed training. Data-parallel training of neural networks. All-Reduce and its efficient implementations.
  - Seminar: Introduction to PyTorch Distributed. Data-parallel training primitives.
- __Week 4:__ Methods for training large models
- __Week 5:__ Sharded data-parallel training, distributed training optimizations
- __Week 6:__ Deep learning performance from first principles
- __Week 7:__ Basics of web service deployment
- __Week 8:__ Systems optimizations for inference
- __Week 9:__ Algorithmic optimizations for inference
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
- [Ruslan Khaidurov](https://github.com/newokaerinasai)
- [Maksim Abraham](https://github.com/fdrose)
- [Sergei Vorobyov](https://github.com/vorobyov01)
- [Gregory Leleytner](https://github.com/RunFMe)
- [Antony Frolov](https://github.com/antony-frolov)
- [Mikhail Khrushchev](https://github.com/MichaelEk)
- [Anton Chigin](https://github.com/achigin)
- [Kamil Izmailov](https://github.com/Kamizm)
- [Roman Gorb](https://github.com/rvg77)
- [Mikhail Seleznev](https://github.com/Dont-Care-Didnt-Ask)

# Past versions
- [2025](https://github.com/mryab/efficient-dl-systems/tree/2025)
- [2024](https://github.com/mryab/efficient-dl-systems/tree/2024)
- [2023](https://github.com/mryab/efficient-dl-systems/tree/2023)
- [2022](https://github.com/mryab/efficient-dl-systems/tree/2022)
- [2021](https://github.com/yandexdataschool/dlatscale_draft)
