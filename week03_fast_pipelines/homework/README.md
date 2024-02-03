# Week 3 home assignment

The assignment for this week consists of three parts: all parts are obligatory, no bonus tasks are given.
Implement your solutions in the folders for the corresponding tasks. 
Create a report for your homework: briefly describe
the structure of your solution for each section, include benchmark results in the tables, and provide explanations of the observed results.
Poorly written reports will give you a reduced grade for the assignment!

Make sure to install needed packages from `requirements.txt` file in the week's folder.

## Submission format:
- For the report, you need to create an `.ipynb` or a `.pdf` file.
- Create a `.zip` archive that contains:
  - Folders with your solutions for each task
  - The report file with instructions on how to run each part, results of running the code and (if necessary) your analysis 
- Upload this archive when submitting the assignment

## Task 1: DIY loss scaling (3 points)
Implement [loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling) for the AMP training mode.
Use the provided semantic segmentation pipeline in [`task1`](./task1).
Your task is to train the model in the AMP mode with loss scaler implemented by you.
You **can use** `torch.cuda.amp.autocast`, and you **cannot use** `torch.cuda.amp.GradScaler()` (you may only for checking your solution).

Let us recall what loss scaling is. 
Loss scaling is used to avoid the gradient underflow problem when computing gradients in FP16 precision. 
The issue here is that while training in full precision, we might acquire rather small values in the gradients, which will vanish when we cast a tensor to a half precision. 
To fix the problem, we use the following solution:

- Make a forward pass for the model and compute the loss
- Multiply the loss value by some factor
- Call `.backward()`
- Update the model's master weights with **unscaled** FP32 gradients

Loss scaling might be done in two different ways: static and dynamic one.
In the static mode, you choose a factor for scaling only once and use it for the whole training procedure.
In the dynamic mode, you recompute the factor each time you scale the loss.

### Task
- Implement static loss scaling (**1.5 points**)
- Implement dynamic loss scaling (**1.5 points**)

The task is done if you manage to stably achieve high accuracy values (0.985+) within 5 training epochs.
Note that you need to implement and successfully train with **both** scaling modes if you want to get a full grade for this task.
As a starting point, you can run the training in a full precision mode, then try to run in an AMP mode with and without PyTorch loss scaler.
You will observe that adding a scaler gives you additional accuracy points.

**Hint:** To make sure that you're doing everything right, you might want to examine gradients' values: (almost) no zeros must be present there.

### Report instructions
When you are done with the code, you can either:
- Run the training function with implemented scaling modes in an `.ipynb` report
- Include training logs AND instructions on how to run your code in a `.pdf` report

## Task 2: efficient batching for language modeling (4 points)
In this part, you need to examine the efficiency of the three batching approaches we discussed during the seminar. 
Let us remind you of them shortly:

**BRAIN**: pad everything to a fixed `max_length`

**BIG BRAIN**: pad only in the `collate_fn`

**BIGGEST BRAIN**: group examples of similar length into buckets, and sample examples for every batch from a single bucket

### Task
More formally, you need to download [WikiText-103 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip) and implement all the mentioned approaches.
Use only the training subset for all the task's subproblems.

- For naive batching, implement a Pytorch `Dataset` class that will parse training data from the source files of the dataset and pad every sample to a fixed `max_length=640`. **(0.5 points)**
- For the second approach, reimplement the `collate_fn` demo from the seminar for this dataset. **(1 point)**
More specifically, you need to pad sequences only up to a maximum sample length in the current batch. 
- For the third approach, implement the `UltraDuperBigBrainDataset` and the `UltraDuperBigBrainBatchSampler` classes. **(2.5 points)**
Objects of the BatchSampler class are iterables and yield a list of indices that correspond to dataset objects, which are put into a batch. 
You can pass this batch sampler to a DataLoader. 
For more information, refer to PyTorch [docs](https://pytorch.org/docs/stable/data.html#automatic-batching-default). 
Objects in each batch should have the same or similar length. 
Sample batches randomly, but ensure that the length difference between the longest and shortest samples is less than or equal to k (try different values of k: 1, 5, 10, 20, 50). 
Note that some batches may be shorter than the specified batch size.
The `__init__` method must work in O(n) time, where n is the length of the dataset. 
The `__iter__` call must work in O(1) time with respect to the size of the dataset (and obviously, in O(batch_size)).
While processing the dataset, put all possible lengths of the samples into a hash table, where keys are lengths and values are containers with the indices of samples of this length.

For each of the implemented methods (and all variations of the third method), mock one training epoch and measure minimum, maximum, mean and median batch processing times.
To mock a training epoch, you need to construct a small GPT-2-like model: use `nn.Embedding` layer, `PositionalEncoding` class from `transformer.py` file and a single `nn.TransformerDecoder` layer with a hidden size of 1024 and 8 heads.
For tokenization, use `torchtext.data.utils.get_tokenizer("basic_english")`.
Run one epoch **without a backward pass**.
Make sure you've [warmed up](https://forums.developer.nvidia.com/t/why-warm-up/48565) the GPU before computing the statistics and do not forget about asynchronous CUDA kernel execution.

Keep in mind that all padding in this task must be **implemented by you**: unlike the seminar, PyTorch’s default collate padding is not allowed.
In every subproblem, for sequences longer than 640 tokens, just truncate the overflowing part.
Feel free to modify the keyword arguments of functions.

**Hint:** In the third subtask, you might want to use a hash table multiple times.
**Hint 2:** In the third subtask, when `k=640`, you should receive the same results as in Subtask 2.

### Report instructions
When you are done with the code, you can either:
- Display the benchmark results in a `pandas.DataFrame` in your `.ipynb` report
- Display the benchmark results in a table in your `.pdf` report

## Task 3 (3 points)
You are given a training script for a [Vision Transformer model](https://huggingface.co/docs/transformers/model_doc/vit) on the [Clothing dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full).
Your task is to examine the bottlenecks of the training pipeline, including the model and the training loop.
The implementation of the model is based on the [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) repository.


### Task
- Profile model performance during training: (**0.5 points**)
   - Forward pass
       - Inspect the Embedding layer
       - Inspect the Attention layer (both self-attention and feed-forward computations)
       - Inspect activation layers: `nn.Softmax`, `nn.GELU`
   - Backward pass
       - How long does it take compared to a forward pass?
  - Several training iterations
- Find deliberate inefficiencies we've left in the code and fix them. There is a total of 6 inefficiencies. (**2.5 points**)

We expect that in your analysis, you will not only examine the time and memory consumption, but also provide explanations of
whether the obtained results are reasonable.

**Hints:**
- PyTorch profiler [recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch TensorBoard profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- Inefficiencies could be anywhere in the code: they may be in data processing, model performance, the training loop — you name it.
- You might want to look at the trace of operations instead of just per-operation profiling, as there is a lot of useful information.

### Report instructions
When you are done with investigations and fixes, you can either:
- Report the profiler output AND its meaningful analysis in your `.ipynb` report file.
Report fixes you made to the code. Be sure to describe how you found them, why the code was inefficient (with profiler screenshots/outputs) and why suggested fixes help.
- The same applies to the `.pdf` file.
