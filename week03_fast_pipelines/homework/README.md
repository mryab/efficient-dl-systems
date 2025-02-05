# Week 3 home assignment

The assignment for this week consists of three parts: all parts are obligatory, no bonus tasks are given, but you can earn more than 10 points in total.
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
- Implement static loss scaling (**1 point**)
- Implement dynamic loss scaling (**1 point**)

The task is done if you manage to stably achieve high accuracy values (0.985+) within 5 training epochs.
Note that you need to implement and successfully train with **both** scaling modes if you want to get a full grade for this task.
As a starting point, you can run the training in the full precision mode, then try to run in the AMP mode with and without the PyTorch loss scaler.
You will observe that adding a scaler gives you additional accuracy points.

**Hint:** To make sure that you're doing everything right, you might want to examine the values of gradients: (almost) no zeros should be present there.

### Report instructions
When you are done with the code, you can either:
- Run the training function with implemented scaling modes in an `.ipynb` report
- Include training logs AND instructions on how to run your code in a `.pdf` report

## Task 2: efficient batching for language modeling (4 points)
In this part, you need to examine the efficiency of the four batching approaches we discussed during the seminar. 
Let us remind you of them shortly:

**BRAIN**: pad everything to a fixed `max_length`

**BIG BRAIN**: pad only in the `collate_fn`

**ULTRA BIG BRAIN**: group examples of similar length into buckets, and sample examples for every batch from a single bucket

**ULTRA DUPER BIG BRAIN**: pack all sequences into one long sequence and generate metadata that indicates where each original sequence starts and ends

### Task
More formally, you need to download [WikiText-103 dataset (dropbox)](https://www.dropbox.com/scl/fi/e6oqpx6iuos7kn9m139z7/wikitext-103-raw-v1.zip?rlkey=81evwbaqfkxtckj8zhks7yied&st=6ept2pdm&dl=0), [WikiText-103 dataset (yandex disk)](https://disk.yandex.ru/d/xwMXnteHKDqehw) and implement all the mentioned approaches.
Use only the training subset for all the task's subproblems.

- For naive batching, implement a Pytorch `Dataset` class that will parse training data from the source files of the dataset and pad every sample to a fixed `max_length=640`. **(0.5 points)**
- For the second approach, reimplement the `collate_fn` demo from the seminar for this dataset. **(0.5 points)**
More specifically, you need to pad sequences only up to a maximum sample length in the current batch.
- For the third approach, implement the `UltraBigBrainDataset` and the `UltraBigBrainBatchSampler` classes. **(1.5 points)**
Objects of the `BatchSampler` class are iterables and yield a list of indices that correspond to dataset objects, which are put into a batch. 
You can pass this batch sampler to a `DataLoader`. 
For more information, refer to PyTorch [docs](https://pytorch.org/docs/stable/data.html#automatic-batching-default). 
Objects in each batch should have the same or similar length. 
Sample batches randomly, but ensure that the length difference between the longest and shortest samples is less than or equal to k (try different values of k: 1, 5, 10, 20, 50). 
Note that some batches may be shorter than the specified batch size.
The `__init__` method must work in O(n) time, where n is the length of the dataset. 
The `__iter__` call must work in O(1) time with respect to the size of the dataset (and obviously, in O(batch_size)).
While processing the dataset, put all possible lengths of the samples into a hash table, where keys are lengths and values are containers with the indices of samples of this length.
- For the fourth approach, we recommend to use `IterableDataset`, which is a good choice when we don't know how many samples we need to create a batch. **(1.5 points)** 
If the last sample is too long, you can either truncate it or drop it from the dataset. 
Don't forget that you also need to build a correct attention mask to prevent cross-contamination of training examples and pass it to the model!

For each of the implemented methods (and all variations of the third method), mock one training epoch and measure minimum, maximum, mean and median batch processing times.
To mock a training epoch, you need to construct a small GPT-2-like model: use `nn.Embedding` layer, `PositionalEncoding` class from `transformer.py` file and a single `nn.TransformerDecoder` layer with a hidden size of 1024 and 8 heads.
For tokenization, use `.tokenize()` method of `AutoTokenizer.from_pretrained("bert-base-uncased")`.
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

## Task 3 (5 points)
You are given a training script for a [Vision Transformer model](https://huggingface.co/docs/transformers/model_doc/vit) on the [Clothing dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full).
In this task, you need to implement a custom profiler to measure the performance of PyTorch models at the layer level. 
The profiler should track the execution time of each layer during the forward and backward passes and output results in a trace event format.
You also need to examine the bottlenecks of the training pipeline, including the model and the training loop (you can use any profilers you want here).
The implementation of the model is based on the [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) repository.

### Task
- Implement a basic profiler: (**2.5 points**)
   - Implement a [context manager](https://book.pythontips.com/en/latest/context_managers.html) to collect execution times for each layer. You have a skeleton of the `Profile` class, feel free to modify or extend it. We are only doing **layer-level** profiling here (not kernel-level).
   - Support **profiling schedule phases** (e.g., wait, warmup, active), similar to the [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).  
   - Implement a `to_perfetto` method that exports data in the [trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw) which is compatible with [Perfetto](https://ui.perfetto.dev/).
   - Profile a ViT model for several training iterations using your custom profiler. Visualize the results in the Perfetto UI. Compare your profiler's layer timings with those from the native PyTorch profiler (Don’t forget a warm-up phase!). **Report** any differences you observe in the measured times.

- Profile CUDA kernels now: (**1 point**)
   - Update your profiler: insert **NVTX markers** via `torch.cuda.nvtx`. This will let you see **individual CUDA kernels** in the timeline when using Nsight Systems. **Remove any explicit synchronization**, because Nsight Systems can capture kernel timings directly from the GPU.  
   - Run your script with **Nsight Systems**:
     ```bash
     nsys profile --env-var CUDA_VISIBLE_DEVICES="YOUR_GPU_ID" -o trace python3 main.py
     ```
   - Open the resulting **`.nsys-rep`** file in Nsight Systems. Examine kernel-level details in the GPU timeline. **Report** whether you see any timing differences compared to your earlier runs. If you see any difference, can you explain the reasons?
  
- Profile model performance during training, find deliberate inefficiencies we've left in the code, and fix them: (**1.5 points**)
   - There is a total of 6 inefficiencies, you will get 0.25 points for each one you find
   - We expect that in your analysis, you will not only examine the time and memory consumption, but also provide explanations of
whether the obtained results are reasonable.

**Hints:**
- Use PyTorch's forward and backward hooks to collect execution times for each module in the model.
- Use `torch.cuda.synchronize()` and `torch.cuda.Event()` correctly to ensure GPU kernels complete before recording events, since all GPU operations are asynchronous ([Asynchronous Execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)).
- Inefficiencies could be anywhere in the code: they may be in data processing, model performance, the training loop — you name it.
- You might want to look at the trace of operations instead of just per-operation profiling, as there is a lot of useful information.

### Report instructions
When you are done with investigations and fixes, you can either:
- Report the profiler output AND its meaningful analysis in your `.ipynb` report file.
List the fixes you made to the code. Be sure to describe how you found them, why the code was inefficient (with profiler screenshots/outputs), and why suggested fixes help.
- The same applies to the `.pdf` file, if you decide to submit your report in that format.
