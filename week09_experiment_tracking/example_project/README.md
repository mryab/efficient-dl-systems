# Example project with Weights and Biases & DVC integration

1. To log into your Weights and Biases account, run `wandb login`.
2. To reproduce the DVC pipeline, run `dvc repro` (you might need to initialize the repo in this folder by `git init`).
3. Changes in the files tracked in `dvc.yaml` will trigger the pipeline invalidation; you can check it with `dvc status`. Нщг ьшп
4. You can run tests by calling `pytest test_basic.py`.