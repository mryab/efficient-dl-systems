# Homework

This week homework is devoted to weight and activation quantization and speculative decoding. 
The details are described in the [notebook](./hw-w8a8-specdec.ipynb).

## W8A8 quantization (6 points + 4 bonus points)

Here you will implement and benchmark kernels for quantization and matrix multiplication in `int8`. 
As a bonus, you can try to implement SmoothQuant algorithm and get Roman Gorb's personal respect.

## Speculative decoding (4 points + 1 bonus point)

You will implement the simplest version of speculative decoding, measure the speedup and also try `huggingface` implementation. If that's too easy (which it should be), you can support appropriate KV-cache computation for the large model.

## Sparse KV-cache (2 bonus points)

You will reproduce the method for detecting streaming and retrieval attention heads from the RazorAttention paper: https://arxiv.org/pdf/2407.15891. After that, you'll visualize the attention maps -- to see, how sparse the attention typically is.
