# Week 9 home assignment

## Submission format
Implement models, training procedures and benchmarks in .py files, run all code in the Jupyter Notebook and convert it to .pdf.
Include your implementations and report file into a .zip archive and submit it.


## Task 1: knowledge distillation for image classification (6 points)

0. Finetune ResNet101 on CIFAR10: change only the classification linear layer [*]; don't freeze other weights (**0 points**)

Then take untrained ResNet101 model, remove layer3 (except one conv block that creates correct number of channels for the 4-th layer) block out of it and implement 3 training setups:
1. Train the model on data only (**1 point**)
2. Train the model on data and add soft cross-entropy between the student (truncated ResNet101) and the teacher (finetuned full ResNet101) (**2 points**)
3. Train the model as in the 2nd subtask but also add MSE loss between corresponding layer1, layer2 and layer 4 features between the student and the teacher (**3 points**)
4. Report test accuracy for each of the models

[\*] Vanilla ResNet is not very well suited for CIFAR: it downsamples the image by x32, while images in CIFAR are 32x32 pixels. So you can
- upsample images (easiest to implement, but you will perform more computations)
- slightly change the first layers (e.g. make `model.conv1` a 3x3 convolution with stride 1 and remove `model.maxpool`)

Feel free to use dataset and model implementation from torch. For losses in 2nd and 3rd subtasks use simple average of all summands.
For the 3rd subtask you will need to return not only model's outputs but also intermediate feature maps.

### Training setup
- Use standard Adam optimizer without scheduler.
- Use any suitable batch size from 128 to 512.
- Training stopping criterion: accuracy (measured from 0 to 1) stabilizes in the second digit after decimal during at least 2 epochs on test set.
That means that you must satisfy condition `torch.abs(acc - acc_prev) < 0.01` for at least two epochs in a row.

## Task 2: use `deepsparse` to prune & quantize your model (4 points)

0. Please, read the whole task description before starting it.
1. Install `deepsparse==1.7.0` and `sparseml==1.7.0`. Note: they might not work smoothly with last torch versions. If so, you can downgrade to `torch==1.12.1`.
2. Take your best trained model from subtasks 1.1-1.3 and run pruning + quantization-aware-training, adapting the following [example](./example_train_sparse_and_quantize.py). You will need to change/implement what is marked by #TODO and report test accuracy of both models. (**3 points**)
3. Take `onnx` baseline (best trained model from subtask 1.1 - 1.3) and pruned-quantized version and benchmark both models on cpu using `deepsparse.benchmark` at batch sizes 1 and 32. (**1 point**) 

For 2.3, you may find [this page](https://web.archive.org/web/20240319095504/https://docs.neuralmagic.com/user-guides/deepsparse-engine/benchmarking/) helpful.

You shouldn't use training stopping criterion in this part, since sparsification recipe relies on having certain amount of epochs.

### Tips: 
- debug your code with resnet18 to iterate faster
- don't forget `model.eval()` before onnx export
- don't forget `convert_qat=True` in `sparseml.pytorch.utils.export_onnx` after you trained the model with quantization
- to visualize `onnx` models, you can use [netron](https://netron.app/)
- explicitely set the amount of cores in `deepsparse.benchmark`
- if you are desperate and don't have time to train bigger models, submit this part with resnet18

Good luck and have 59 funs!
