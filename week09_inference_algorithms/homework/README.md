# Homework

This week homework is devoted to weight and activation quantization and speculative decoding. 
The details are described in the [notebook](./hw-w8a8-specdec.ipynb).

## W8A8 quantization (6 points + 3 bonus points)

Here you will implement and benchmark kernels for quantization and matrix multiplication in `int8`. 
As a bonus, you can try to implement SmoothQuant algorithm and get Roman Gorb's personal respect.

## Speculative decoding (4 points)

You will implement the simplest version of speculative decoding, measure the speedup and also try `huggingface` implementation.

## Sparse KV-cache (1 bonus point)

You will reproduce the method for detecting streaming and retrieval attention heads from the RazorAttention paper: https://arxiv.org/pdf/2407.15891. After that, you'll visualize the attention maps -- to see, how sparse the attention typically is.
