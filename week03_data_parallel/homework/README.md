# Week 3 home assignment

The assignment for this week consists of four parts: the first three are obligatory, and the final one is a bonus.
Include all the files with implemented functions/classes and the report for Tasks 2 (and 4) in your submission.

## Task 1 (10% points)

Implement the function for deterministic sequential printing of N numbers for N processes,
using [sequential_print.py](./sequential_print.py) as a template. 
You should be able to test it with `torchrun --nproc_per_node N sequential_print.py`

## Task 2 (60% points)

The pipeline you saw in the seminar shows only the basic building blocks of distributed training. Now, let's train
something actually interesting!

For example, let's take the [CIFAR-100](https://pytorch.org/vision/0.8/datasets.html#torchvision.datasets.CIFAR100)
dataset and train a model with **synchronized** Batch Normalization: we aggregate the statistics across workers during
each forward pass. 

Importantly, you have to call a communication primitive **only once** during each forward or backward pass; 
if you use it more than once, you will only earn up to 40% overall HW points (instead of 60) for this task.

Also, implement a version of distributed training which is aware of gradient accumulation:
for each batch that doesn't run `optimizer.step`, you can avoid the All-Reduce step altogether.

Compare the performance (in terms of both speed, memory footprint and final quality) of your distributed training 
pipeline with [the](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
[primitives](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) provided by PyTorch. 
You need to compare the implementations by training with **at least two** processes.

In addition, test the SyncBN layer itself by comparing the results with standard **BatchNorm1d** and changing 
the number of workers (use at least 1 and 4), the size of activations (128, 256, 512, 1024), and the batch size (32, 64). 
Compare the results of forward/backward passes with the FP32 inputs from a standard Gaussian distribution and 
the loss function that simply sums the outputs over first N/2 samples (N is the total batch size): 
a working implementation should have reasonably high `rtol` and `atol` (at least 1e-3).
Finally, measure the GPU time (2+ workers) and the memory footprint of standard **SyncBatchNorm** 
and your implementation in the same setup as above.


Provide the results of your experiments in a .ipynb/.pdf report and attach it to your code 
when submitting the homework. Your report should include a brief experimental setup (if changed) 
and the infrastructure description (version of PyTorch, number of processes, type of GPUs, etc.).

Use [syncbn.py](./syncbn.py) and [ddp_cifar100.py](./ddp_cifar100.py) as a template. 

## Task 3 (30% points)

Until now, we only aggregated the gradients across different workers during training. But what if we want to run
distributed validation on a large dataset as well? In this assignment, you have to implement distributed metric
aggregation: shard the dataset across different workers (with [scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)), compute accuracy for each subset on 
its respective worker and then average the metric values on the master process.

Also, make one more quality-of-life improvement of the pipeline by logging the loss (and accuracy!) 
only from the rank-0 process to avoid flooding the standard output of your training command. 
Submit the training code that includes all enhancements from Tasks 2 and 3.

## Task 4 (bonus, 30% points)

Using [allreduce.py](./allreduce.py) as a template, implement the Ring All-Reduce algorithm
using only point-to-point communication primitives from `torch.distributed`. 
Compare it with the provided implementation of Butterfly All-Reduce, 
as well as with `torch.distributed.all_reduce` in terms of CPU speed, memory usage and the accuracy of averaging. 
Specifically, compare custom implementations of All-Reduce with 1–32 workers and compare your implementation of 
Ring All-Reduce with `torch.distributed.all_reduce` on 1–16 processes and vectors of 1,000–100,000 elements.