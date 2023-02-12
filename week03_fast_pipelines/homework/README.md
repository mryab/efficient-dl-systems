# Week 3 home assignment

The assignment for this week consists of three parts: all parts are obligatory, no bonus tasks are provided.
Implement your solutions in the folders for the corresponding sections. Create a report for your homework: briefly describe
the structure of your solution for each section, include benchmark results in the tables, provide explanations of the observed results.

Make sure to install needed packages from `requirements.txt` file in the week's folder.

## Submission format:
- For the report you might create `.ipynb` file or `.pdf` file.
- Create an archive that contains:
  - Folder with each section with your solution
  - Report file

## Task 1: DIY loss scaling (3 points)
Implement loss scaling for AMP training mode.
Use provided semantic segmentation pipeline for this section.
Your task is to train the model in the AMP mode with loss scaler implemented by you.
You **can use** `torch.cuda.amp.autocast` and you **cannot use** `torch.cuda.amp.GradScaler()` (you may only for checking your solution).

Let us remind what loss scaling is. Loss scaling is used to avoid the gradient underflow problem, when computing gradients in FP16 precision. The issue here is that while training in full precision, we might acquire rather small values in the gradients, which will vanish when we cast a tensor to a half precision. To fix the problem the following solution is used:

- make a forward pass for the model and compute the loss
- multiply loss value to some factor
- call `.backward()`
- update model's master weights with **unscaled** FP32 gradients

Loss scaling might be done in two different ways: static and dynamic ones.
In static mode, you choose a factor for scaling only once and use it for the whole training procedure.
In dynamic mode, you recompute the factor each time you scale the loss.

### Task
- Implement static loss scaling (**1.5 points**)
- Implement dynamic loss scaling  (**1.5 points**)

The task is done if you managed to stably achieve high accuracy values (0.985+) within 5 training epochs.
For a start, you can run the training in a full precision mode, then try to run in an AMP mode with and without PyTorch loss scaler.
You will observe that adding a scaler gives you additional accuracy points.

**Hint.** To make sure that you're doing everything right, you might want to examine gradients' values: (almost) no zeros must be present there.

### Report instructions
After you are done with a code, you can either:
- Run training function with implemented scaling modes in `.ipynb` report
- Include training logs AND instructions how to run your code in the `.pdf` report

## Task 2: efficient batching for language modeling (4 points)

In this part you need to examine the efficiency of the three batching approaches we discussed during the seminar. Let us remind you shortly:

**BRAIN**: pad everything to a fixed `max_length`

**BIG BRAIN**: pad only in the `collate_fn`

**ULTRA DUPER BIG BRAIN**: presort data to sample sequences smartly, preserving similar examples length in the batch

### Task
More formally, you need to download [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) and implement all the mentioned approaches.
Use only the training subset for all the task's sub-problems.

- For naive batching, implement a Pytorch Dataset class that will parse training data from the source files of the dataset and pad every sample to a fixed `max_length=640`. **(0.5 points)**
- For the second approach, reimplement `collate_fn` demo from the seminar for this dataset.
More specifically, you need to pad sequences only up to a maximum sample length in the current batch. **(1 point)**
- Finally, for the third approach, implement the following trick.
While processing the dataset, split it into the several groups (bins) by sample length.
you need to uniformly split the samples list sorted by sample length. Conduct experiments for 1, 5, 10, 25, 50 bins.
While calling a `__getitem__` method, you firstly sample a bin number, then sample the needed examples number form the bin and pad them with collator from the second subtask. **(2.5 points)**

In every sub-problem, for sequences longer than 640 tokens just truncate the overflowing part.

For each of the implemented methods mock one training epoch and provide min, max, mean and median batch processing times.
To mock a training epoch you need to construct a small GPT-2-like model: use `nn.Embedding` layer, `PositionalEncoding` class from `transformer.py` file and a single `nn.TransformerDecoder` layer with hidden size 1024 and 8 heads.
For tokenization use `torchtext.data.utils.get_tokenizer("basic_english")`.
Run one epoch **without a backward pass**. Make sure you've [warmed up](https://forums.developer.nvidia.com/t/why-warm-up/48565) GPU before computing the statistics and do not forget about asynchronous CUDA kernels execution.

**Hint 1.** In the third subtask you might want to use (not obligatory) a `batch_sampler` in the data loader.
For that, you need to inspect the corresponding Pytorch docs [section](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler).

**Hint 2.** Third approach with 1 bin must replicate the results of the second approach.

### Report instructions
After you are done with a code, you can either:
- Display benchmark results in `pandas.DataFrame` in your `.ipynb` file
- Display benchmark results in the table in your `.pdf` file

## Task 3: ViT profiling (3 points)
In this section, you're given a training script for a [Vision Transformer model](https://huggingface.co/docs/transformers/model_doc/vit) on [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).
Your task is to examine the bottlenecks of the model.
You can find the model script in the `vit.py` file.
The implementation is based on [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) repository.

**Important**: it is necessary to download the dataset — `train.zip` — and to put it into the `./data` folder.

### Task
- Profile model performance during training (**1 point**)
   - Forward pass
       - Inspect the Embedding layer
       - Inspect the Attention layer (both self-attention and feed-forward computations)
       - Inspect activations layers: `nn.Softmax`, `nn.GELU`
   - Backward pass
       - How long does it take compared to a forward pass?
- Find deliberate inefficiencies we've left in the implementation and fix them (**2 points**)

We expect that during inspections you will not only examine time and memory consumptions but also provide explanations
whether acquired results are reasonable.

**Hints:**
- PyTorch profiler [recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- To provide meaningful report of the profiler output, be sure to describe main blocks of the model, analyse if profiler output is expected
and spotlight model's bottlenecks
- Performance tuning [guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- To debug easier you may want to use `extract_dataset_globs(half=True)` while you extract images
- There are hints in the ViT code to ease your analysis. Beware! One of the hints is misleading

### Report instructions
After you are done with investigation and fixes, you can either:
- Report profiler output AND its meaningful analysis in your `.ipynb` report file.
Report fixes you made to ViT. Be sure to describe how you found them, why the code was inefficient and why suggested fixes help.
- The same applies for `.pdf` file


Good luck and have 59 funs!
