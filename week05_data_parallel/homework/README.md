# Week 5 home assignment

The assignment for this week consists of four parts: the first three are obligatory, and the fourth is a bonus one.
Include all the files with implemented functions/classes and the report for Tasks 2 and 4 in your submission.

## Task 1 (1 point)

Implement the function for deterministic sequential printing of N numbers for N processes,
using [sequential_print.py](./sequential_print.py) as a template. 
You should be able to test it with `torchrun --nproc_per_node N sequential_print.py`
Pay attention to the output format!

## Task 2 (7 points)

The pipeline you saw in the seminar shows only the basic building blocks of distributed training. Now, let's train
something actually interesting!

### SyncBatchNorm implementation
For this task, let's take the [CIFAR-100](https://pytorch.org/vision/0.8/datasets.html#torchvision.datasets.CIFAR100)
dataset and train a model with **synchronized** Batch Normalization: this version of the layer aggregates 
the statistics **across all workers** during each forward pass.

Importantly, you have to call a communication primitive **only once** during each forward or backward pass; 
if you use it more than once, you will only earn up to 4 points for this task.
Additionally, you are **not allowed** to use internal PyTorch functions that compute the backward pass
of batch normalization: please implement it manually.

### Reducing gradient synchronization
Also, implement a version of distributed training which is aware of **gradient accumulation**:
for every batch that doesn't run `optimizer.step`, you do not need to run All-Reduce for gradients at all.

### Benchmarking the training pipeline
Compare the performance (in terms of speed, memory footprint, and final quality) of your distributed training 
pipeline with the one that uses primitives from PyTorch (i.e., [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) **and** [torch.nn.SyncBatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)). 
You need to compare the implementations by training with **at least two** processes, and your pipeline needs to have 
at least 2 gradient accumulation steps.

### Tests for SyncBatchNorm
In addition, **test the SyncBN layer itself** by comparing the results with standard **BatchNorm1d** and changing 
the number of workers (1 and 4), the size of activations (128, 256, 512, 1024), and the batch size (32, 64). 

Compare the results of forward/backward passes in the following setup: 
* FP32 inputs come from the standard Gaussian distribution;
* The loss function takes the outputs of batch normalization and computes the sum over all dimensions 
for first B/2 samples (B is the total batch size).

A working implementation of SyncBN should have reasonably low `atol` (at least 1e-3) and `rtol` equal to 0.

This test needs to be implemented via `pytest` in [test_syncbn.py](./test_syncbn.py): in particular, all the above 
parameters (including the number of workers) need to be the inputs of that test.
Therefore, you will need to **start worker processes** within the test as well: `test_batchnorm` contains helpful 
comments to get you started.
The test can be implemented to work only on the CPU for simplicity.

### Performance benchmarks
Finally, measure the GPU time (2+ workers) and the memory footprint of standard **SyncBatchNorm** 
and your implementation in the above setup: in total, you should have 8 speed/memory benchmarks for each implementation.

### Submission format
Provide the results of your experiments in a `.ipynb`/`.pdf` report and attach it to your code 
when submitting the homework.
Your report should include a brief experimental setup (if changed), results of all experiments **with the commands/code 
to reproduce them**, and the infrastructure description (version of PyTorch, number of processes, type of GPUs, etc.).

Use [syncbn.py](./syncbn.py) and [ddp_cifar100.py](./ddp_cifar100.py) as a template. 

## Task 3 (2 points)

Until now, we only aggregated the gradients across different workers during training. But what if we want to run
distributed validation on a large dataset as well? In this assignment, you have to implement distributed metric
aggregation: shard the dataset across different workers (with [scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)), compute accuracy for each subset on 
its respective worker and then average the metric values on the master process.

Also, make one more quality-of-life improvement of the pipeline by logging the loss (and accuracy!) 
only from the rank-0 process to avoid flooding the standard output of your training command. 
Submit the training code that includes all enhancements from Tasks 2 and 3.

## Task 4 (bonus, 3 points)

Using [allreduce.py](./allreduce.py) as a template, implement the Ring All-Reduce algorithm
using only point-to-point communication primitives from `torch.distributed`. 
Compare it with the provided implementation of Butterfly All-Reduce
and with `torch.distributed.all_reduce` in terms of CPU speed, memory usage and the accuracy of averaging. 
Specifically, compare custom implementations of All-Reduce with 1–32 workers and compare your implementation of 
Ring All-Reduce with `torch.distributed.all_reduce` on 1–16 processes and vectors of 1,000–100,000 elements.
