# Week 9 home assignment

## Submission format
Implement models, training procedures and benchmarks in .py files, run all code in the Jupyter Notebook and convert it to .pdf.
Include your implementations and report file into a .zip archive and submit it.


## Task 1: knowledge distillation for image classification (6 points)
0. Finetune ResNet101 on CIFAR10: change only the classification linear layer; don't freeze other weights (**0 points**)

Then take untrained ResNet101 model, remove layer3 (except one conv block that creates correct number of channels for the 4-th layer) block out of it and implement 3 training setups:
1. Train the model on data only (**1 point**)
2. Train the model on data and add soft cross-entropy between the student (truncated ResNet101) and the teacher (finetuned full ResNet101) (**2 points**)
3. Train the model as in the 2nd subtask but also add MSE loss between corresponding layer1, layer2 and layer 4 features between the student and the teacher (**3 points**)
4. Report accuracy for each of the models

Feel free to use dataset and model implementation from torch. For losses in 2nd and 3rd subtasks use simple average of all summands.
For the 3rd subtask you will need to return not only model's outputs but also intermediate feature maps.

### Training setup
- Use standard Adam optimizer without scheduler.
- Use any suitable batch size from 128 to 512.
- Training stopping criterion: accuracy (measured from 0 to 1) stabilizes in the second digit after decimal during at least 2 epochs on test set.
That means that you must satisfy condition `torch.abs(acc - acc_prev) < 0.01` for at least two epochs in a row.

## Task 2: implement quantization aware training for ResNet (4 points)
1. Take your best trained model from subtasks 1.1-1.3 and implement quantization aware training for it with the same loss function that was used for training (**3 points**)
2. Report accuracy as well as speed benchmarks for FP32 and INT8 models on x86_64 architecture (**1 point**)

Feel free to reuse code from the seminar for these subtasks. Use the same training setup as in the Task 1.

Good luck and have 59 funs!
