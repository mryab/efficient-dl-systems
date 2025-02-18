import argparse

import torch
import torch.nn as nn
import torchvision

from train import model_provider, train, train_amp


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
    parser.add_argument("-t", "--transforms-level", type=int, default=1)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--n-epochs", type=int, default=100)
    args = parser.parse_args()

    train_dataloader_, val_dataloader_ = get_loaders(args.transforms_level)
    model_ = model_provider()
    optimizer = torch.optim.Adam(model_.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    if args.amp:
        train_amp(model_, loss_fn, optimizer, train_dataloader_, val_dataloader_, n_epochs=args.n_epochs)
    else:
        train(model_, loss_fn, optimizer, train_dataloader_, val_dataloader_, n_epochs=args.n_epochs)
