# Week 2 home assignment

The assignment for this week consists of three parts. 
All parts are obligatory, no bonus tasks are given, but you can earn more than 10 points in total.
Implement your solutions in the folders for the corresponding tasks. 
Create a report for your homework: briefly describe
the structure of your solution for each section, include benchmark results in the tables, and provide explanations of the observed results.
Poorly written reports will give you a reduced grade for the assignment!

Make sure to install the necessary packages from `requirements.txt` in the week's folder.

## Submission format
- For the report, you need to create an `.ipynb` or a `.pdf` file.
- Create a `.zip` archive that contains:
  - Folders with your solutions for each task
  - The report file with instructions on how to run each part, results of running the code and (when necessary) your analysis 
- Upload this archive when submitting the assignment

## Task 1: DIY loss scaling (2 points)
Implement [loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling) for the AMP training mode.

Let us recall what loss scaling is. 
Loss scaling is used to avoid the gradient underflow problem when computing gradients in FP16 precision. 
The issue here is that while training in full precision, we might obtain rather small values in the gradients, which will vanish when we cast a tensor to half precision. 

To fix the problem, we use the following solution:
- Run the forward pass for the model and compute the loss
- Multiply the loss value by some factor
- Call `.backward()`
- Update the model's master weights with **unscaled** FP32 gradients

Loss scaling might be done in two different ways: static and dynamic.
In the static mode, you choose a factor for scaling only once and use it for the whole training procedure.
In the dynamic mode, you recompute the factor each time you scale the loss.

### Task
- Implement static loss scaling (**1 point**)
- Implement dynamic loss scaling (**1 point**)

Use the provided semantic segmentation pipeline in [`task1/`](./task1).
Your task is to train the model in the AMP mode with the loss scaler implemented by you.

The task is done if you manage to reliably achieve high accuracy values (0.985+) within 5 training epochs.
Note that you need to implement and successfully train with **both** scaling modes if you want to get a full grade for this task.
As a starting point, you can run training in the full precision mode, then try to run in the AMP mode with and without the PyTorch loss scaler.
You will observe that adding a scaler gives you additional accuracy points.

You **can use** `torch.cuda.amp.autocast`, and you **cannot use** `torch.cuda.amp.GradScaler()` (except for checking your solution).

**Hint:** To make sure that you're doing everything right, you might want to examine the values of gradients: (almost) no zeros should be present there.

### Report instructions
When you are done with the code, you can either:
- Run the training function with implemented scaling modes in an `.ipynb` report
- Include training logs AND instructions on how to run your code in a `.pdf` report

## Task 2: efficient batching for language modeling (5 points)
In this part, you need to examine the efficiency of the four batching approaches we discussed during the seminar. 
Let us remind you of them shortly:

**BRAIN**: pad everything to a fixed `max_length`

**BIG BRAIN**: pad only in `collate_fn`

**ULTRA BIG BRAIN**: group examples of similar length into buckets, and sample examples for every batch from a single bucket

**ULTRA DUPER BIG BRAIN**: pack all sequences into one long sequence and generate metadata that indicates where each original sequence starts and ends

### Task
More formally, you need to download [WikiText-103 dataset (Dropbox)](https://www.dropbox.com/scl/fi/e6oqpx6iuos7kn9m139z7/wikitext-103-raw-v1.zip?rlkey=81evwbaqfkxtckj8zhks7yied&st=6ept2pdm&dl=0), [WikiText-103 dataset (Yandex Disk)](https://disk.yandex.ru/d/xwMXnteHKDqehw) and implement all the mentioned approaches.
Use only the training subset for all the task's subproblems.

1. **(0.5 points)** For naive batching, implement a Pytorch `Dataset` class that will parse the training data from the source files of the dataset and pad every sample to a fixed `max_length=640`.
2. **(0.5 points)** For the second approach, reimplement the `collate_fn` demo from the seminar for this dataset.
More specifically, you need to pad sequences only up to a maximum sample length in the current batch.
3. **(1.5 points)** For the third approach, implement the `UltraBigBrainDataset` and the `UltraBigBrainBatchSampler` classes.

    Objects of the `BatchSampler` class are iterables and yield a list of indices that correspond to dataset objects, which are put into a batch. 
    You can pass this batch sampler to a `DataLoader`. 
    For more information, refer to PyTorch [docs](https://pytorch.org/docs/stable/data.html#automatic-batching-default). 
    Objects in each batch should have the same or similar length. 
    Sample batches randomly, but ensure that the length difference between the longest and shortest samples is less than or equal to `k` (try different values of `k`: 1, 5, 10, 20, 50). 
    Note that some batches may be smaller than the specified batch size.
    
    The `__init__` method must work in O(n) time, where `n` is the length of the dataset. 
    The `__iter__` call must work in O(1) time with respect to the size of the dataset (and obviously, in O(`batch_size`)).
   
    While processing the dataset, put all possible lengths of the samples into a hash table, where keys are lengths and values are containers with the indices of samples of this length.
4. **(2.5 points)** For the fourth approach (sequence packing), you need to implement packing where multiple sequences are concatenated into a single long sequence.
Don't forget that you also need to build a correct attention mask to prevent cross-contamination of training examples!
   - **Basic packing (0.5 points):** Implement packing with a correct attention mask for packed samples. You are allowed to truncate or pad packed sequences to reach the target length. Packing can be done as a preprocessing step: you don't need to pack on-the-fly during iteration.
   - **FFD packing (1 point):** Implement the First-Fit Decreasing (FFD) algorithm, following the paper ["Fewer Truncations Improve Language Modeling"](https://arxiv.org/abs/2404.10830). 
   
     FFD is a bin packing algorithm that first sorts sequences by length in decreasing order, then places each sequence into the first "bin" (packed sample) where it fits. If no existing bin has enough space, a new bin is created. No truncation is allowed: sequences that exceed `max_length` should be skipped. Packed samples can be padded to `max_length`. O(N * M) complexity is acceptable for this subtask, where N is the number of sequences and M is the number of bins. You can also implement an O(N log M) approach following the paper.
   - **OBFD packing (1 point):** Implement the Optimized Best-Fit Decreasing (OBFD) algorithm with a segment tree as described in the same paper. OBFD improves upon FFD by efficiently finding the best-fitting bin, i.e. the one with the least remaining space that still fits the sequence. The complexity should be O(N log L), where L is the bin size (max sequence length).

For each of the implemented methods (and all variations of the third and fourth methods), mock one training epoch and measure minimum, maximum, mean and median batch processing times.
To mock a training epoch, you need to construct a small GPT-2-like model: use `nn.Embedding` layer, `PositionalEncoding` class from `transformer.py` file and a single `nn.TransformerDecoder` layer with a hidden size of 1024 and 8 heads.
For tokenization, use the `.tokenize()` method of `AutoTokenizer.from_pretrained("bert-base-uncased")`.
Run one training epoch to measure the iteration time.
Make sure you've [warmed up](https://forums.developer.nvidia.com/t/why-warm-up/48565) the GPU before computing the statistics and do not forget about asynchronous CUDA kernel execution.

Keep in mind that all padding in this task must be **implemented by you**: unlike the seminar, PyTorch’s default collation padding is not allowed.
In every subproblem, for sequences longer than 640 tokens, just truncate the overflowing part.
Feel free to modify the keyword arguments of functions.

**Hint:** In the third subtask, you might want to use a hash table multiple times.

**Hint 2:** In the third subtask, when `k=640`, you should receive the same results as in Subtask 2.

### Report instructions
When you are done with the code, you can either:
- Display the benchmark results in a `pandas.DataFrame` in your `.ipynb` report
- Display the benchmark results in a table in your `.pdf` report

## Task 3 (4 points)
You are given a training script for a [Vision Transformer model](https://huggingface.co/docs/transformers/model_doc/vit) on the [Clothing dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full).
In this task, you need to implement a custom profiler to measure the performance of PyTorch models at the layer level, and find inefficiencies in the provided training pipeline.
The implementation of the model is based on the [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) repository.

### Task
- Implement a custom profiler (**1.5 points**):
   - Implement a [context manager](https://book.pythontips.com/en/latest/context_managers.html) to collect execution times for each layer during forward and backward passes. You have a skeleton of the `Profile` class, feel free to modify or extend it. We are doing **layer-level** profiling here (not kernel-level).
   - Support **profiling schedule phases** (e.g., wait, warmup, active), similar to the [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).
   - Implement a `to_perfetto` method that exports data in the [trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw) compatible with [Perfetto](https://ui.perfetto.dev/).

- Measure the training performance (**0.5 points**):
  - Profile a ViT model for several training iterations using your custom profiler. 
  - Visualize the results in the Perfetto UI. 
  - Compare your profiler's layer timings with those from the native PyTorch profiler. **Report** any differences you observe in the measured times.

- Find and fix inefficiencies (**2 points**):
   - We have left 6 deliberate inefficiencies in the code. You will get 0.33 points for each one you find and fix.
   - You can use any profiling tools for this subtask (PyTorch profiler, your custom profiler, etc.).
   - Inefficiencies could be anywhere: data processing, model architecture, training loop, etc.
   - In your analysis, examine both time and memory consumption, and explain whether the results are reasonable.

**Hints:**
- Use PyTorch's forward and backward hooks to collect execution times for each module.
- Use `torch.cuda.synchronize()` and `torch.cuda.Event()` correctly to ensure GPU kernels complete before recording times, since GPU operations are asynchronous ([Asynchronous Execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)).
- Look at the trace of operations, not just per-operation profiling — the timeline contains useful information.

### Report instructions
When you are done with investigations and fixes, you can either:
- Report the profiler output AND its meaningful analysis in your `.ipynb` report file.
List the fixes you made to the code. Be sure to describe how you found them, why the code was inefficient (with profiler screenshots/outputs), and why the suggested fixes help.
- The same applies to the `.pdf` file, if you decide to submit your report in that format.
