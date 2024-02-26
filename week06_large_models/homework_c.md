# Option C: write your own model parallelism

__Reminder:__ For this setup, you will need two or more GPUs. 
If you're an external student, you can get access to [2x T4 on kaggle](https://www.kaggle.com/code/cpmpml/protein-bert-finetune-with-2-t4) - but it requires phone confirmation (at the time of writing).

You will need to implement two of the popular model parallelism, algorithms: the pipeline and tensor-parallel ones. 
The specifics of each one are described below.

### Pipeline parallelism (5+ points)

__Quest:__ implement pipeline parallelism using [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html) or [`torch.rpc`](https://pytorch.org/docs/stable/rpc.html), your choice.

![img](https://i.imgur.com/Va7GN6x.png)

Please start from [PyTorch CIFAR](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) starter kit. 
We will not judge if you choose something heavier, e.g. ImageNet -- but that is not necessary.
For this assignment, you will need to implement pipeline-parallel training similar to the one defined in [GPipe](https://arxiv.org/abs/1811.06965) (or lecture 4 notes). 
Feel free to use CPU or share one GPU for multiple processes. 
The main victory condition is that the model must show the same convergence rate as training on a single GPU -- which you will need to demosntrate.

- __(1/5 points)__ doesn't have to be parallel, works with at least 2 stages
- __(3/5 points)__ actual parallelism, like in GPipe, 4 or more stages, loss is shown to go down
- __(5/5 points)__ achieve the same covnergence curves on CIFAR (or the task of your choice) as training with a single process
- __(bonus points)__ demonstrate that your implementation can process a very large number of micro-batches (at least 1000 microbatches with 4 stages) without going OOM
- __(bonus points)__ implement [pipedream](https://arxiv.org/pdf/1806.03377.pdf) or [interleaved pipeline](https://openreview.net/pdf?id=cw-EmNq5zfD) (this may be hard!)

__Conditions:__ your implementation should use multiprocessing (i.e. not a single process sending data between devices). 
Existing pipelines (e.g. `torch.distributed.pipeline`) are off limits. 
If your code resembles one of existing pipelines too much, we will brutally interrogate you about the implementation details (as in "we'll make sure it's easier to build your own").

### Tensor Parallelism (5+ points)
_aka AlexNet-style parallelism_

__Quest:__ implement intra-layer model parallelism and make sure it works.

Similarly to the pipeline task, we recommend that you start with the [PyTorch CIFAR](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial, but you can choose to use a more complex starter kit if you're feeling adventurous. 
If you don't have stable access to a multi-GPU setup, feel free to use CPU or share the same GPU across all processes.

The main objective is to implement AlexNet-style model parallelism, wherein each device computes a subset of "neurons" in each layer and then exchanges results with other units. 
We recommend doing this with [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html) with either `all_reduce` or `all_gather`, depending on whether you split matrix rows or columns.

- __(1/5 points)__ a single linear layer can be split between two GPUs
- __(2/5 points)__ parallelize a simple architecture, `Sequential(linear1, relu, linear2)`
- __(3/5 points)__ a more complex architecture, add at least one batch- or layer normalization
- __(5/5 points)__ train a model like this and show learning curves,
- __(+1 bonus)__ parallelize either ResNet50 or some Transformer variant (__warning__, this is hard!),
- __(+1 bonus)__ implement mixed column + row parallelism, run 2 dense layers with a single communication round (see below).

For both options, please attach the learning curves of your model compared to regular single-process training. 
A decent explanation how this type of parallelism works can be found in Section 2.3 of [this paper](https://arxiv.org/pdf/2104.04473.pdf). 
Optimizations such as Figure 5 from that paper or this [weird trick](https://arxiv.org/abs/1404.5997) are welcome, but not required.
