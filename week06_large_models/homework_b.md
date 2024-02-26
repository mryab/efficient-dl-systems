# Option B: a PhD in tuning DeepSpeed

__Reminder:__ For this setup, you will need two - and preferably more - GPUs. 
If you're an external student, you can get access to [2x T4 on kaggle](https://www.kaggle.com/code/cpmpml/protein-bert-finetune-with-2-t4) - but it requires phone confirmation (at the time of writing). 
Option B specifically needs a lot of RAM -- or a lot of pain. 
If you're limited to kaggle free T4s - and if you're not into masochism, we recommend that you choose something else.


__Task description:__ the main objective of this task is to run memory-efficient training algorithms in your multi-GPU setup and compile results in a small report.

You will need to compare two setups:
- Small model: fine-tune a [Bloom-560m](https://huggingface.co/bigscience/bloom-560m) model for text classification
- Large model: either [Bloom-7b1 (3B)](https://huggingface.co/bigscience/bloomz-3b) or the [7B1 version](https://huggingface.co/bigscience/bloom-7b1) on the same task (or choose any other model with 2~8 billion parameters)

Train both models using the Adam optimizer. 
You can write your own training code or follow the [official tutorial](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). 
We recommend that you use the small model for debugging, since larger models may not fit without deepspeed tricks.


First, you will need to [install](https://www.deepspeed.ai/tutorials/advanced-install/) DeepSpeed and [integrate it](https://huggingface.co/docs/transformers/main_classes/deepspeed) into your training code - and make sure you can train with *some* basic deepspeed config(1 point).

Once you got the basic code running, you will need to answer three questions -- for both small and large models:

1. (3pts) How does ZeRO-2 (optimizer & grad sharding) speed compare to ZeRO-3 (full model sharding)?
  - does this difference depend on the training batch size?

2. (3pts) How does single-GPU offloading compare to training without offloading?
  - what happens if you offload only the optimizer state vs. the model as well?
  - does it change if you combine offloading with data parallelism (ZeRO)?

3. (3 pts) How does DeepSpeed compare with native alternatives:
   - Your ZeRO3 config v.s. [FullyShardedDataParallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
   - [DeepSpeed pipeline](https://www.deepspeed.ai/tutorials/pipeline/) v.s. [torch.distributed Pipeline](https://pytorch.org/docs/stable/pipeline.html)
   - [Optional], Tensor Parallel training v.s. automatic tensor parallelism from [tensor_parallel](https://github.com/BlackSamorez/tensor_parallel)
  

Each question is worth 3 points. 
"Answering" each question requires a benchmark code, example run and saved statistics, e.g. total training time.

