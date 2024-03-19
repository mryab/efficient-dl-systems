# Week 9: Efficient model inference

* Lecture: [slides](./lecture_compressed.pdf)
* Seminar: [notebook](./practice.ipynb)
* Homework: see [homework/README.md](homework/README.md)

### Setup for the seminar notebook
You can use [conda](https://docs.anaconda.com/free/miniconda/), [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to create the environment.

```
conda create -n inference \
    python=3.10 \
    pytorch=2.2.1 \
    torchvision=0.17.1 \
    torchaudio=2.2.1 \
    pytorch-cuda=11.8 \
    matplotlib=3.8.0 \
    seaborn=0.12.2 \
    numpy=1.26.4 \
    ipywidgets=8.1.2 \
    jupyterlab_widgets=3.0.10 \
    jupyterlab=4.0.11 \
    tqdm=4.65.0 \
    -c pytorch -c nvidia -y

conda activate inference

# To run part with auto-gptq
pip install auto-gptq==0.7.1 accelerate==0.28.0
pip install --upgrade git+https://github.com/huggingface/transformers.git

# To run part with Smoothquant
cd ~
git clone git@github.com:mit-han-lab/smoothquant.git
cd smoothquant
python setup.py install
cd path/to/efficient-dl-systems/week09_compression

# Finally, running notebook
jupyter lab --no-browser
```

## Further reading
### Efficient architectures
* https://arxiv.org/pdf/1704.04861.pdf
* https://arxiv.org/pdf/2101.03697.pdf
* https://arxiv.org/pdf/2206.04040.pdf
* https://arxiv.org/pdf/2006.04768.pdf
* https://arxiv.org/pdf/1909.11942.pdf
* https://arxiv.org/pdf/2006.16236.pdf

### Knowledge distillation
* https://arxiv.org/pdf/2106.05237.pdf
* https://arxiv.org/pdf/1910.01108.pdf
* https://arxiv.org/pdf/1909.10351.pdf

### Pruning
* https://arxiv.org/abs/2302.04089
* https://arxiv.org/abs/2301.00774

### Matrices decompositions
* https://arxiv.org/pdf/1906.11755.pdf
* https://arxiv.org/pdf/2004.09031.pdf
* https://arxiv.org/pdf/2009.13977.pdf
* https://arxiv.org/pdf/2004.04124.pdf
* https://arxiv.org/pdf/2111.06312.pdf

### Quantization
* https://arxiv.org/abs/2206.09557
* https://arxiv.org/abs/2208.07339
* https://huggingface.co/blog/hf-bitsandbytes-integration
* https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
