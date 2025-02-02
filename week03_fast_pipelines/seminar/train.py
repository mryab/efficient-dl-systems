from typing import Literal

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def model_provider():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4),
        nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 10, 128),
        nn.ReLU(),
        nn.Linear(128, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )
    return model


def train(
    model: nn.Module,
    loss_fn: nn.modules.loss,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 3,
    device: torch.device = torch.device("cuda:0"),
    precision: Literal["full", "half"] = "full",
) -> None:
    if precision == "half":
        model.half()
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        for x_train, y_train in tqdm(train_dataloader, desc=f"Epoch {epoch}: "):
            if precision == "half":
                x_train = x_train.half()
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_pred = model(x_train)

            loss = loss_fn(y_pred, y_train)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print("Starting validation...")
            model.eval()
            val_loss = torch.empty(len(val_dataloader))
            val_accuracy = torch.empty(len(val_dataloader))

            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_dataloader):
                    if precision == "half":
                        x_val = x_val.half()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)
                    loss = loss_fn(y_pred.float(), y_val)
                    val_loss[i] = loss
                    val_accuracy[i] = (
                        (torch.argmax(y_pred, dim=-1) == y_val).float().mean()
                    )

            print(
                f"Epoch: {epoch}, loss: {val_loss.mean().detach().cpu()}, "
                f"accuracy: {val_accuracy.mean().detach().cpu()}"
            )
    model.eval()


def train_amp(
    model: nn.Module,
    loss_fn: nn.modules.loss,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 3,
    device: torch.device = torch.device("cuda:0"),
    precision: Literal["fp16", "bf16"] = "bf16",
    loss_scaling: bool = False,
) -> None:
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)

    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        ValueError("Unsupported precision for amp.")

    for epoch in range(n_epochs):
        model.train()

        for x_train, y_train in tqdm(train_dataloader, desc=f"Epoch {epoch}: "):
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                x_train, y_train = x_train.to(device), y_train.to(device)
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)

            if loss_scaling:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print("Starting validation...")
            model.eval()
            val_loss = torch.empty(len(val_dataloader))
            val_accuracy = torch.empty(len(val_dataloader))

            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_dataloader):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)
                    loss = loss_fn(y_pred, y_val)
                    val_loss[i] = loss
                    val_accuracy[i] = (
                        (torch.argmax(y_pred, dim=-1) == y_val).float().mean()
                    )

            print(
                f"Epoch: {epoch}, loss: {val_loss.mean().detach().cpu()}, "
                f"accuracy: {val_accuracy.mean().detach().cpu()}"
            )
    model.eval()
