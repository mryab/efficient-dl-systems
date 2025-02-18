import argparse

import torch
import torch.nn as nn
import torchvision

from tqdm.auto import tqdm

from train import model_provider


def get_loaders(
    transforms_level: int = 1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if transforms_level == 1:
        # no transforms
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif transforms_level == 2:
        # modest transforms
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomPerspective(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        # heavy transforms
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomPerspective(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.GaussianBlur(5),
                torchvision.transforms.RandomAdjustSharpness(2),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomAdjustSharpness(1),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomAdjustSharpness(0.5),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomEqualize(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    mnist_train = torchvision.datasets.MNIST(
        "./mnist/", train=True, download=True, transform=transform
    )
    mnist_val = torchvision.datasets.MNIST(
        "./mnist/", train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        mnist_train, batch_size=1024, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        mnist_val, batch_size=1024, shuffle=False
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-nvtx", type=int, default=1)
    parser.add_argument("-t", "--transforms-level", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    train_dataloader_, val_dataloader_ = get_loaders(args.transforms_level)
    model_ = model_provider()
    optimizer = torch.optim.Adam(model_.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()

    epoch = 0
    i = 0
    model_.to(device)
    model_.train()

    if args.emit_nvtx:
        with torch.autograd.profiler.emit_nvtx():
            for x_train, y_train in tqdm(train_dataloader_, desc=f"Epoch {epoch}: "):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    x_train, y_train = x_train.to(device), y_train.to(device)
                    y_pred = model_(x_train)
                    loss = loss_fn(y_pred, y_train)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                i += 1
                if i == 3:
                    break
    else:
        torch.cuda.nvtx.range_push("Train Loop")
        for x_train, y_train in tqdm(train_dataloader_, desc=f"Epoch {epoch}: "):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                x_train, y_train = x_train.to(device), y_train.to(device)
                torch.cuda.nvtx.range_push("Forward")
                y_pred = model_(x_train)
                loss = loss_fn(y_pred, y_train)
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Backward")
            scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("Optimizer Step")
            scaler.step(optimizer)
            torch.cuda.nvtx.range_pop()
            scaler.update()
            optimizer.zero_grad()
            i += 1
            if i == 3:
                break
        torch.cuda.nvtx.range_pop()
